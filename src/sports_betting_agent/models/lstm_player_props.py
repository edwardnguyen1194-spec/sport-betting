"""LSTM neural-network model for player-prop betting.

Design goals
============

* **Target win rate:** 65-75% on top-confidence picks (top decile).
* **Inputs:** a rolling window of the last ``N`` games for a single
  player -- counting stats, minutes/innings, opponent defensive
  rating, rest days, home/away, game pace, and the sportsbook line
  for the prop we're modelling.
* **Output:** probability the player **goes over** the posted line
  for that prop.
* **Training:** standard BCE loss, Adam optimiser, early stopping.
  The model is deliberately small (2 LSTM layers, 64 hidden) so it
  can train on a few thousand player-game rows in minutes on CPU
  and keep inference latency << 10 ms per prop.
* **PyTorch optional.** If PyTorch isn't available at import time
  we fall back to a lightweight NumPy implementation that mirrors
  the same API. This keeps the dashboard usable even on minimal
  deploys (e.g. fly.io micro VMs without CUDA wheels).

Example usage::

    model = LSTMPlayerPropsModel(LSTMPlayerPropsConfig(feature_dim=12))
    model.fit(training_logs, epochs=20)
    prob_over = model.predict_over(recent_log_for_lebron, line=26.5)
    if prob_over >= 0.65:
        bet("LeBron James OVER 26.5 points")
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)


try:  # pragma: no cover - exercised in deployments that have torch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:  # torch missing or CPU-only wheel error
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config and data types
# ---------------------------------------------------------------------------


@dataclass
class LSTMPlayerPropsConfig:
    feature_dim: int = 12
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    sequence_length: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    early_stop_patience: int = 5
    device: str = "cpu"


@dataclass
class PlayerGameLog:
    """A player's historical game log, one row per game, newest last.

    Every ``features`` entry must have the same length == ``feature_dim``.
    ``stat`` is the ground-truth value of the prop (e.g. points scored).
    """

    player_id: str
    prop: str
    features: List[List[float]]
    stats: List[float]
    lines: List[float]          # posted line at each historical game
    meta: Dict = field(default_factory=dict)

    def training_pairs(
        self, sequence_length: int
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Yield (window, line, target_over) tuples for training."""

        pairs: List[Tuple[np.ndarray, float, float]] = []
        n = len(self.features)
        if n <= sequence_length:
            return pairs
        for i in range(sequence_length, n):
            window = np.asarray(self.features[i - sequence_length : i], dtype=np.float32)
            line = float(self.lines[i])
            stat = float(self.stats[i])
            target = 1.0 if stat > line else 0.0
            pairs.append((window, line, target))
        return pairs


class PlayerPropsDataset(Dataset):  # type: ignore[misc]
    """PyTorch dataset wrapping :class:`PlayerGameLog` objects."""

    def __init__(self, logs: Sequence[PlayerGameLog], sequence_length: int) -> None:
        self.samples: List[Tuple[np.ndarray, float, float]] = []
        for log in logs:
            self.samples.extend(log.training_pairs(sequence_length))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, line, target = self.samples[idx]
        if _TORCH_AVAILABLE:
            return (
                torch.from_numpy(window),
                torch.tensor([line], dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
        return window, np.float32(line), np.float32(target)


# ---------------------------------------------------------------------------
# Torch network
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):  # type: ignore[misc]
        def __init__(self, cfg: LSTMPlayerPropsConfig) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=cfg.feature_dim,
                hidden_size=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=cfg.bidirectional,
            )
            out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
            self.head = nn.Sequential(
                nn.Linear(out_dim + 1, cfg.hidden_dim),  # +1 for the posted line
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )

        def forward(self, x: "torch.Tensor", line: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            merged = torch.cat([last, line], dim=-1)
            logits = self.head(merged).squeeze(-1)
            return logits


# ---------------------------------------------------------------------------
# Public model wrapper
# ---------------------------------------------------------------------------


class LSTMPlayerPropsModel:
    """High-level wrapper used by the rest of the package.

    Falls back to a logistic-regression over the flattened window if
    PyTorch isn't installed -- still useful, still trains, still
    ships probabilities; just less expressive.
    """

    def __init__(self, config: Optional[LSTMPlayerPropsConfig] = None) -> None:
        self.config = config or LSTMPlayerPropsConfig()
        self._torch_net = None
        self._np_weights: Optional[np.ndarray] = None
        self._np_bias: float = 0.0
        self._fitted = False
        if _TORCH_AVAILABLE:
            self._torch_net = _LSTMNet(self.config).to(self.config.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, logs: Sequence[PlayerGameLog], *, epochs: Optional[int] = None) -> Dict:
        if not logs:
            raise ValueError("LSTMPlayerPropsModel.fit: no training logs provided")
        if _TORCH_AVAILABLE:
            return self._fit_torch(logs, epochs=epochs or self.config.epochs)
        return self._fit_numpy(logs, epochs=epochs or self.config.epochs)

    def _fit_torch(self, logs: Sequence[PlayerGameLog], *, epochs: int) -> Dict:
        assert _TORCH_AVAILABLE and self._torch_net is not None
        ds = PlayerPropsDataset(logs, self.config.sequence_length)
        if len(ds) == 0:
            raise ValueError(
                "LSTMPlayerPropsModel.fit: not enough history per player for the configured sequence_length"
            )
        # Very small holdout; this class is expected to be used inside
        # a proper training harness with CV elsewhere.
        split = max(1, int(len(ds) * 0.2))
        val_samples = ds.samples[-split:]
        train_samples = ds.samples[:-split] or val_samples

        ds.samples = train_samples
        val_ds = PlayerPropsDataset.__new__(PlayerPropsDataset)
        val_ds.samples = val_samples  # type: ignore[attr-defined]

        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size)

        opt = torch.optim.Adam(self._torch_net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_state = None
        patience = 0
        history: List[Dict] = []

        for epoch in range(epochs):
            self._torch_net.train()
            total = 0.0
            count = 0
            for win, line, target in loader:
                win = win.to(self.config.device)
                line = line.to(self.config.device)
                target = target.to(self.config.device)
                opt.zero_grad()
                logits = self._torch_net(win, line)
                loss = loss_fn(logits, target)
                loss.backward()
                opt.step()
                total += loss.item() * win.size(0)
                count += win.size(0)
            train_loss = total / max(count, 1)

            self._torch_net.eval()
            with torch.no_grad():
                total = 0.0
                count = 0
                for win, line, target in val_loader:
                    win = win.to(self.config.device)
                    line = line.to(self.config.device)
                    target = target.to(self.config.device)
                    logits = self._torch_net(win, line)
                    loss = loss_fn(logits, target)
                    total += loss.item() * win.size(0)
                    count += win.size(0)
                val_loss = total / max(count, 1)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            logger.debug("LSTM epoch %d train=%.4f val=%.4f", epoch, train_loss, val_loss)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self._torch_net.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.config.early_stop_patience:
                    break

        if best_state is not None:
            self._torch_net.load_state_dict(best_state)
        self._fitted = True
        return {"backend": "torch", "history": history, "best_val_loss": best_val}

    def _fit_numpy(self, logs: Sequence[PlayerGameLog], *, epochs: int) -> Dict:
        """Tiny NumPy fallback: logistic regression on flattened windows."""

        X: List[np.ndarray] = []
        y: List[float] = []
        lines: List[float] = []
        seq = self.config.sequence_length
        for log in logs:
            for window, line, target in log.training_pairs(seq):
                flat = window.flatten().astype(np.float32)
                X.append(flat)
                lines.append(float(line))
                y.append(float(target))
        if not X:
            raise ValueError("LSTMPlayerPropsModel.fit: not enough history per player")

        X_arr = np.stack(X, axis=0)
        lines_arr = np.asarray(lines, dtype=np.float32).reshape(-1, 1)
        X_arr = np.hstack([X_arr, lines_arr])
        y_arr = np.asarray(y, dtype=np.float32)

        # Standard scale.
        self._np_mean = X_arr.mean(axis=0)
        self._np_std = X_arr.std(axis=0) + 1e-6
        X_scaled = (X_arr - self._np_mean) / self._np_std

        dim = X_scaled.shape[1]
        rng = np.random.default_rng(42)
        w = rng.normal(0.0, 0.01, size=dim).astype(np.float32)
        b = 0.0
        lr = self.config.learning_rate

        for epoch in range(epochs):
            logits = X_scaled @ w + b
            preds = 1.0 / (1.0 + np.exp(-logits))
            grad_w = X_scaled.T @ (preds - y_arr) / len(y_arr)
            grad_b = float(np.mean(preds - y_arr))
            w -= lr * grad_w
            b -= lr * grad_b

        self._np_weights = w
        self._np_bias = b
        self._fitted = True
        return {"backend": "numpy", "samples": int(len(y_arr))}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_over(self, log: PlayerGameLog, line: float) -> float:
        """Return probability the player goes OVER ``line`` in their next game."""

        if not self._fitted:
            raise RuntimeError("LSTMPlayerPropsModel: call fit() before predict_over()")

        seq = self.config.sequence_length
        if len(log.features) < seq:
            # Pad with zeros if we haven't seen enough games yet.
            pad = [[0.0] * self.config.feature_dim] * (seq - len(log.features))
            window = np.asarray(pad + log.features, dtype=np.float32)
        else:
            window = np.asarray(log.features[-seq:], dtype=np.float32)

        if _TORCH_AVAILABLE and self._torch_net is not None:
            self._torch_net.eval()
            with torch.no_grad():
                t_window = torch.from_numpy(window).unsqueeze(0).to(self.config.device)
                t_line = torch.tensor([[float(line)]], dtype=torch.float32).to(self.config.device)
                logits = self._torch_net(t_window, t_line)
                prob = torch.sigmoid(logits).item()
                return float(max(0.0, min(1.0, prob)))

        # NumPy fallback.
        flat = np.concatenate([window.flatten(), np.asarray([line], dtype=np.float32)])
        scaled = (flat - self._np_mean) / self._np_std
        logit = float(scaled @ self._np_weights + self._np_bias)  # type: ignore[operator]
        return 1.0 / (1.0 + math.exp(-logit))

    def predict_recommendation(
        self,
        log: PlayerGameLog,
        line: float,
        player_name: str,
        *,
        over_price: float = -110,
        under_price: float = -110,
    ) -> Optional[Dict]:
        """Convenience helper used by the props strategy.

        Returns a dict with confidence/edge for whichever side the
        model prefers, or ``None`` if the prediction doesn't clear the
        configured 0.60 confidence floor (below which props lose to
        vig historically).
        """

        p_over = self.predict_over(log, line)
        side = "Over" if p_over >= 0.5 else "Under"
        confidence = p_over if side == "Over" else (1.0 - p_over)
        if confidence < 0.60:
            return None
        price = over_price if side == "Over" else under_price
        from ..models_schema import american_to_decimal, american_to_implied

        decimal = american_to_decimal(price)
        implied = american_to_implied(price)
        edge = confidence - implied
        if edge <= 0:
            return None
        return {
            "player": player_name,
            "prop": log.prop,
            "selection": side,
            "line": line,
            "confidence": round(confidence, 4),
            "edge": round(edge, 4),
            "american": price,
            "decimal": decimal,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if _TORCH_AVAILABLE and self._torch_net is not None:
            torch.save(
                {
                    "config": self.config.__dict__,
                    "state_dict": self._torch_net.state_dict(),
                    "backend": "torch",
                },
                path,
            )
        else:
            payload = {
                "config": self.config.__dict__,
                "weights": self._np_weights.tolist() if self._np_weights is not None else None,
                "bias": self._np_bias,
                "mean": getattr(self, "_np_mean", None).tolist() if hasattr(self, "_np_mean") else None,
                "std": getattr(self, "_np_std", None).tolist() if hasattr(self, "_np_std") else None,
                "backend": "numpy",
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Synthetic-data helpers (used by tests / demos)
# ---------------------------------------------------------------------------


def make_synthetic_log(
    *,
    player_id: str = "demo",
    prop: str = "points",
    games: int = 60,
    feature_dim: int = 12,
    seed: int = 0,
) -> PlayerGameLog:
    """Produce a deterministic fake player-game log for tests."""

    rng = random.Random(seed)
    features: List[List[float]] = []
    stats: List[float] = []
    lines: List[float] = []
    base = rng.uniform(12.0, 28.0)
    for _ in range(games):
        row = [rng.gauss(0.0, 1.0) for _ in range(feature_dim)]
        # Stat is a noisy function of the first feature + base.
        stat = max(0.0, base + 5.0 * row[0] + rng.gauss(0.0, 3.0))
        line = round(base + rng.gauss(0.0, 1.0) * 0.5, 1)
        features.append(row)
        stats.append(stat)
        lines.append(line)
    return PlayerGameLog(player_id=player_id, prop=prop, features=features, stats=stats, lines=lines)
