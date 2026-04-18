(() => {
  const $ = (sel) => document.querySelector(sel);
  const oddsList = $("#odds-list");
  const recList = $("#rec-list");
  const chatLog = $("#chat-log");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const sportSelect = $("#sport-select");

  const fmtAmerican = (n) => (n > 0 ? `+${Math.round(n)}` : `${Math.round(n)}`);

  const marketMap = { moneyline: "Tỷ lệ thắng", spread: "Kèo chấp", total: "Tổng điểm" };
  const fmtMarket = (m) => marketMap[m] || m;

  const stratMap = {
    value_bets: "Giá trị",
    heavy_favorite: "Đội mạnh",
    "heavy_favorite+value_bets": "Đội mạnh + Giá trị",
    spread_value: "Kèo chấp giá trị",
    total_value: "Tổng điểm giá trị",
    contrarian: "Ngược dòng",
    "spread_value+value_bets": "Kèo chấp + Giá trị",
    "total_value+value_bets": "Tổng điểm + Giá trị",
    "contrarian+value_bets": "Ngược dòng + Giá trị",
    "heavy_favorite+spread_value": "Đội mạnh + Kèo chấp",
    middle_detector: "Kèo giữa",
    situational: "Tình huống",
    elo_edge: "Elo Power",
    pythagorean: "Pythagorean",
    "elo_edge+value_bets": "Elo + Giá trị",
    "pythagorean+value_bets": "Pyth + Giá trị",
    "situational+value_bets": "Tình huống + Giá trị",
    "middle_detector+spread_value": "Kèo giữa + Kèo chấp",
    ai_analysis: "AI phân tích",
    player_props: "Chỉ số cầu thủ",
    ensemble: "Tổng hợp",
  };
  const fmtStrategy = (s) => stratMap[s] || s;

  const viReasoning = (text) => text
    .replace(/\bSpread value:/g, "Kèo chấp giá trị:")
    .replace(/\bTotal value:/g, "Tổng điểm giá trị:")
    .replace(/\bContrarian:/g, "Ngược dòng:")
    .replace(/\bpublic on/g, "công chúng đặt")
    .replace(/\bbut sharps disagree/g, "nhưng nhà cái sắc bén không đồng ý")
    .replace(/\bBetting/g, "Đặt cược")
    .replace(/\bValue:/g, "Giá trị:")
    .replace(/\bHeavy favorite/g, "Đội mạnh")
    .replace(/\bbest price/g, "giá tốt nhất")
    .replace(/\bconsensus no-vig prob/g, "xác suất đồng thuận")
    .replace(/\bacross/g, "qua")
    .replace(/\bbooks/g, "nhà cái")
    .replace(/\bTarget WR band/g, "Mục tiêu tỷ lệ thắng")
    .replace(/\bsharp fair prob/g, "xác suất thực")
    .replace(/\bsharp no-vig fair prob/g, "xác suất thực")
    .replace(/\bvs implied/g, "so với kỳ vọng")
    .replace(/\bedge/gi, "lợi thế")
    .replace(/\bpaying/g, "trả")
    .replace(/\bat\b/g, "tại")
    .replace(/\bin\b/g, "trong");

  const allSports = ["baseball_ncaa","baseball_mlb","basketball_nba","basketball_ncaab","football_nfl","football_ncaaf","hockey_nhl"];

  async function loadOdds() {
    oddsList.innerHTML = "Đang tải...";
    const sport = sportSelect.value;
    const sports = sport === "all" ? allSports : [sport];
    try {
      // Fetch all selected sports in parallel
      const results = await Promise.all(sports.map(s => fetch(`/api/odds?sport=${encodeURIComponent(s)}`).then(r => r.json()).catch(() => ({games:[]}))));
      const allGames = results.flatMap(d => d.games || []);
      if (allGames.length === 0) {
        oddsList.innerHTML = "<div class='card'>Không có trận nào.</div>";
        return;
      }
      oddsList.innerHTML = "";
      for (const game of allGames.slice(0, 50)) {
        const lines = game.lines || [];
        const ml = lines.filter((l) => l.market === "moneyline");
        const spreads = lines.filter((l) => l.market === "spread");
        const totals = lines.filter((l) => l.market === "total");
        const homeLines = ml.filter((l) => l.selection.toLowerCase() === game.home_team.toLowerCase());
        const awayLines = ml.filter((l) => l.selection.toLowerCase() === game.away_team.toLowerCase());
        const bestHome = homeLines.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const bestAway = awayLines.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const homeSpreads = spreads.filter((l) => l.selection.toLowerCase() === game.home_team.toLowerCase());
        const bestSpread = homeSpreads.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const overs = totals.filter((l) => l.selection.toLowerCase() === "over");
        const unders = totals.filter((l) => l.selection.toLowerCase() === "under");
        const bestOver = overs.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const bestUnder = unders.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const sportLabel = sportLabels[game.sport] || game.sport;
        const div = document.createElement("div");
        div.className = "card";
        const gameDate = game.commence_time ? new Date(game.commence_time) : null;
        const dateStr = gameDate ? gameDate.toLocaleString("en-US", {
          weekday: "short", month: "short", day: "numeric",
          hour: "numeric", minute: "2-digit", timeZoneName: "short"
        }) : "";
        div.innerHTML = `
          <div class="best-pick-sport">${sportLabel}</div>
          <div class="teams">${game.away_team} @ ${game.home_team}</div>
          ${dateStr ? `<div class="meta" style="color: #7ecbff; font-weight: 500;">🗓 ${dateStr}</div>` : ""}
          <div class="meta">
            ${bestAway ? `${game.away_team} ${fmtAmerican(bestAway.american)}` : ""}
            ${bestHome ? ` • ${game.home_team} ${fmtAmerican(bestHome.american)}` : ""}
            ${bestAway || bestHome ? ` (${(bestHome || bestAway).book})` : ""}
          </div>
          ${bestSpread ? `<div class="meta">Kèo chấp: ${game.home_team} ${bestSpread.line > 0 ? '+' : ''}${bestSpread.line} (${fmtAmerican(bestSpread.american)})</div>` : ""}
          ${bestOver ? `<div class="meta">Tổng: O/U ${bestOver.line} — Trên ${fmtAmerican(bestOver.american)}${bestUnder ? ` • Dưới ${fmtAmerican(bestUnder.american)}` : ""} (${bestOver.book})</div>` : ""}
          <div class="meta">${(game.meta?.sources || []).join(", ") || game.source}</div>
        `;
        oddsList.appendChild(div);
      }
    } catch (exc) {
      oddsList.innerHTML = `<div class='card'>Lỗi: ${exc}</div>`;
    }
  }

  async function loadRecs() {
    recList.innerHTML = "Đang phân tích...";
    try {
      const sport = sportSelect.value;
      const sportsParam = sport === "all" ? allSports.join(",") : sport;
      const res = await fetch(`/api/recommendations?sports=${encodeURIComponent(sportsParam)}`);
      const data = await res.json();
      if (!data.recommendations || data.recommendations.length === 0) {
        recList.innerHTML = "<div class='card'>Chưa tìm thấy bet phù hợp.</div>";
        return;
      }
      recList.innerHTML = "";
      for (const rec of data.recommendations) {
        const conf = rec.confidence;
        const stars = conf >= 0.85 ? "★★★★★" : conf >= 0.75 ? "★★★★☆" : conf >= 0.65 ? "★★★☆☆" : conf >= 0.55 ? "★★☆☆☆" : "★☆☆☆☆";
        const lockLabel = conf >= 0.80 ? "🔒 LOCK" : conf >= 0.70 ? "🔥 HOT" : "";
        const div = document.createElement("div");
        div.className = "card rec";
        const recSportLabel = sportLabels[rec.sport] || rec.sport;
        div.innerHTML = `
          <div class="best-pick-sport">${recSportLabel}</div>
          <div class="teams">${rec.selection} ${lockLabel ? `<span class="lock-badge">${lockLabel}</span>` : ""}</div>
          <div class="meta">
            ${rec.away_team} @ ${rec.home_team} • ${fmtMarket(rec.market)}${rec.line ? ' ' + rec.line : ''} • ${fmtAmerican(rec.american)} @ ${rec.book}
          </div>
          <div class="meta">
            <span class="stars">${stars}</span>
            Chiến lược: <strong>${fmtStrategy(rec.strategy)}</strong> •
            Độ tin ${Math.round(conf * 100)}% • Lợi thế ${(rec.edge * 100).toFixed(1)}%
          </div>
          <div class="reason">${viReasoning(rec.reasoning || "")}</div>
        `;
        recList.appendChild(div);
      }
    } catch (exc) {
      recList.innerHTML = `<div class='card'>Lỗi: ${exc}</div>`;
    }
  }

  function appendChat(role, text) {
    const div = document.createElement("div");
    div.className = `msg ${role === "user" ? "user" : "bot"}`;
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // Chat form removed (replaced by live agent activity feed).
  // Null-guard so existing handler-wiring code doesn't crash on
  // missing #chat-form.
  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const message = chatInput.value.trim();
      if (!message) return;
      appendChat("user", message);
      chatInput.value = "";
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
        });
        const data = await res.json();
        appendChat("bot", data.content || "(trống)");
      } catch (exc) {
        appendChat("bot", `Lỗi: ${exc}`);
      }
    });
  }

  const sportLabels = {
    baseball_ncaa: "NCAA Baseball",
    baseball_mlb: "MLB",
    basketball_nba: "NBA",
    basketball_ncaab: "NCAA Basketball",
    football_nfl: "NFL",
    football_ncaaf: "NCAA Football",
    hockey_nhl: "NHL",
  };

  async function loadBestPicks() {
    const list = $("#best-picks-list");
    list.innerHTML = "Đang phân tích tất cả môn thể thao...";
    try {
      const res = await fetch("/api/best-picks");
      const data = await res.json();
      if (!data.picks || data.picks.length === 0) {
        list.innerHTML = "<div class='card'>Chưa tìm thấy kèo tốt.</div>";
        return;
      }
      list.innerHTML = "";
      for (const rec of data.picks) {
        const sportLabel = sportLabels[rec.sport] || rec.sport;
        const potentialWin = rec.american > 0
          ? (100 * rec.american / 100).toFixed(0)
          : (100 * 100 / Math.abs(rec.american)).toFixed(0);
        const div = document.createElement("div");
        div.className = "card best-pick-card";
        div.innerHTML = `
          <div class="best-pick-sport">${sportLabel}</div>
          <div class="teams">${rec.selection}</div>
          <div class="meta">${rec.away_team} @ ${rec.home_team}</div>
          <div class="bet-details">
            <span>${fmtMarket(rec.market)}${rec.line ? ' ' + rec.line : ''}</span>
            <span>Kèo: <strong>${fmtAmerican(rec.american)}</strong></span>
            <span>$100 → <strong>+$${potentialWin}</strong></span>
          </div>
          <div class="meta">
            Chiến lược: <strong>${fmtStrategy(rec.strategy)}</strong> •
            Độ tin ${Math.round(rec.confidence * 100)}% • Lợi thế ${(rec.edge * 100).toFixed(1)}%
          </div>
          <div class="reason">${viReasoning(rec.reasoning || "")}</div>
        `;
        list.appendChild(div);
      }
    } catch (exc) {
      list.innerHTML = `<div class='card'>Lỗi: ${exc}</div>`;
    }
  }

  // --------------------------------------------------------------
  // Risk panel — bankroll, peak, drawdown, halt status.
  // Yellow when drawdown >10%, red border + halt chip when halted.
  // --------------------------------------------------------------
  async function loadRisk() {
    const panel = $("#risk-panel");
    if (!panel) return;
    try {
      const res = await fetch("/api/risk");
      const r = await res.json();
      const dd = Number(r.drawdown_pct || 0);
      const halted = !!r.halted;
      const bankroll = Number(r.bankroll || 0);
      const peak = Number(r.peak_bankroll || 0);

      const bankrollEl = $("#risk-bankroll");
      const peakEl = $("#risk-peak");
      const ddEl = $("#risk-drawdown");
      const stateEl = $("#risk-state");
      const chip = $("#risk-status-chip");

      if (bankrollEl) bankrollEl.textContent = `$${bankroll.toFixed(2)}`;
      if (peakEl) peakEl.textContent = `$${peak.toFixed(2)}`;
      if (ddEl) {
        ddEl.textContent = `${dd.toFixed(2)}%`;
        ddEl.classList.toggle("warn", !halted && dd > 10);
        ddEl.classList.toggle("halt", halted);
      }

      let panelState = "ok";
      let chipLabel = "An toàn";
      let chipCls = "risk-chip-ok";
      if (halted) {
        panelState = "halted";
        chipLabel = "Tạm dừng — Sụt giảm quá 20%";
        chipCls = "risk-chip-halt";
      } else if (dd > 10) {
        panelState = "warn";
        chipLabel = "Cảnh báo — Sụt giảm >10%";
        chipCls = "risk-chip-warn";
      }
      panel.dataset.state = panelState;
      if (chip) {
        chip.className = `risk-chip ${chipCls}`;
        chip.textContent = chipLabel;
      }
      if (stateEl) {
        stateEl.textContent = halted ? "Tạm dừng" : (dd > 10 ? "Cảnh báo" : "An toàn");
        stateEl.classList.toggle("warn", !halted && dd > 10);
        stateEl.classList.toggle("halt", halted);
      }
    } catch (exc) {
      const chip = $("#risk-status-chip");
      if (chip) {
        chip.className = "risk-chip";
        chip.textContent = "Không tải được";
      }
    }
  }

  // --------------------------------------------------------------
  // Per-strategy scoreboard — W-L, ROI, avg CLV.
  // CLV + by_strategy come from /api/clv; ROI from roi_by_strategy
  // which we added to the handler (derived from closed ledger).
  // --------------------------------------------------------------
  async function loadStrategyScoreboard() {
    const body = $("#strategy-scoreboard-body");
    if (!body) return;
    try {
      const res = await fetch("/api/clv");
      const data = await res.json();
      const byStrat = data.by_strategy || {};
      const roiByStrat = data.roi_by_strategy || {};
      const names = Array.from(new Set([
        ...Object.keys(byStrat),
        ...Object.keys(roiByStrat),
      ])).filter(n => n && n !== "unknown");

      if (names.length === 0) {
        body.innerHTML = `<tr><td colspan="5" class="scoreboard-empty">Chưa có đủ dữ liệu. Cần ít nhất vài cược đã kết thúc.</td></tr>`;
        return;
      }

      // Sort by bet count so the most-used strategies surface first.
      names.sort((a, b) => {
        const ba = (byStrat[a]?.bets || 0) + (roiByStrat[a]?.bets || 0);
        const bb = (byStrat[b]?.bets || 0) + (roiByStrat[b]?.bets || 0);
        return bb - ba;
      });

      body.innerHTML = "";
      for (const name of names) {
        const clv = byStrat[name] || {};
        const roi = roiByStrat[name] || {};
        const bets = Math.max(clv.bets || 0, roi.bets || 0);
        const record = clv.record || "0-0";
        const roiPct = roi.roi_pct;
        const avgClv = clv.average_clv_pct;
        const roiCls = roiPct === undefined ? "score-neutral" : (roiPct > 0 ? "score-pos" : (roiPct < 0 ? "score-neg" : "score-neutral"));
        const clvCls = avgClv === undefined ? "score-neutral" : (avgClv > 0 ? "score-pos" : (avgClv < 0 ? "score-neg" : "score-neutral"));
        const roiText = roiPct === undefined ? "—" : `${roiPct > 0 ? "+" : ""}${roiPct.toFixed(2)}%`;
        const clvText = avgClv === undefined ? "—" : `${avgClv > 0 ? "+" : ""}${avgClv.toFixed(2)}%`;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${fmtStrategy(name)}</strong></td>
          <td>${record}</td>
          <td class="${roiCls}">${roiText}</td>
          <td class="${clvCls}">${clvText}</td>
          <td>${bets}</td>
        `;
        body.appendChild(tr);
      }
    } catch (exc) {
      body.innerHTML = `<tr><td colspan="5" class="scoreboard-empty">Lỗi tải bảng điểm: ${exc}</td></tr>`;
    }
  }

  // --------------------------------------------------------------
  // Market-mix panel — spread vs total breakdown of OPEN bets.
  // Uncle wants BOTH markets firing; flag yellow if either is <20%.
  // --------------------------------------------------------------
  async function loadMarketMix() {
    const panel = $("#market-mix-panel");
    if (!panel) return;
    const totalEl = $("#mix-total");
    const spreadEl = $("#mix-spread");
    const spreadPctEl = $("#mix-spread-pct");
    const totalMarketEl = $("#mix-total-market");
    const totalPctEl = $("#mix-total-pct");
    const ouEl = $("#mix-ou");
    const ouSubEl = $("#mix-ou-sub");
    const chip = $("#market-mix-chip");
    try {
      const res = await fetch("/api/ledger");
      const data = await res.json();
      const open = data.open || [];
      const total = open.length;
      let spreadCount = 0;
      let totalCount = 0;
      let overCount = 0;
      let underCount = 0;
      for (const b of open) {
        const m = (b.market || "").toLowerCase();
        if (m === "spread") {
          spreadCount += 1;
        } else if (m === "total") {
          totalCount += 1;
          const sel = (b.selection || "").toLowerCase();
          if (sel.includes("over")) overCount += 1;
          else if (sel.includes("under")) underCount += 1;
        }
      }
      const spreadPct = total > 0 ? (spreadCount / total) * 100 : 0;
      const totalPct = total > 0 ? (totalCount / total) * 100 : 0;

      if (totalEl) totalEl.textContent = String(total);
      if (spreadEl) spreadEl.textContent = String(spreadCount);
      if (spreadPctEl) spreadPctEl.textContent = total > 0 ? `${spreadPct.toFixed(0)}%` : "—";
      if (totalMarketEl) totalMarketEl.textContent = String(totalCount);
      if (totalPctEl) totalPctEl.textContent = total > 0 ? `${totalPct.toFixed(0)}%` : "—";
      if (ouEl) ouEl.textContent = `${overCount} / ${underCount}`;
      if (ouSubEl) ouSubEl.textContent = totalCount > 0 ? `Over vs Under` : "—";

      // Flag yellow if either market is under-represented — Uncle
      // wants BOTH spreads and totals firing in parallel.
      const imbalanced = total >= 5 && (spreadPct < 20 || totalPct < 20);
      panel.dataset.state = imbalanced ? "warn" : "ok";
      if (chip) {
        if (total === 0) {
          chip.className = "risk-chip";
          chip.textContent = "Chưa có kèo mở";
        } else if (imbalanced) {
          chip.className = "risk-chip risk-chip-warn";
          chip.textContent = "Phân bổ chưa đều";
        } else {
          chip.className = "risk-chip risk-chip-ok";
          chip.textContent = "Cân bằng";
        }
      }
      spreadEl?.classList.toggle("warn", imbalanced && spreadPct < 20);
      totalMarketEl?.classList.toggle("warn", imbalanced && totalPct < 20);
    } catch (exc) {
      if (chip) {
        chip.className = "risk-chip";
        chip.textContent = "Không tải được";
      }
    }
  }

  $("#btn-refresh").addEventListener("click", loadOdds);
  $("#btn-recs").addEventListener("click", loadRecs);
  if ($("#btn-best-picks")) $("#btn-best-picks").addEventListener("click", loadBestPicks);
  sportSelect.addEventListener("change", () => { loadOdds(); loadRecs(); });

  // Agent activity feed — shows what the 10 Claude sub-agents are
  // currently doing/thinking in real time. Replaces the old chat
  // panel so Uncle can watch the AI brain live.
  const AGENT_VI = {
    pick_reviewer:      "Chuyên gia xét kèo",
    news_triage:        "Lọc tin tức",
    post_mortem:        "Phân tích thua",
    game_analyst:       "Phân tích trận",
    opportunity_scout:  "Săn cơ hội",
    strategy_auditor:   "Kiểm tra chiến lược",
    skills_learner:     "Học kỹ năng mới",
    mcp_discovery:      "Tìm MCP",
    self_reflection:    "Tự kiểm điểm",
    hooks_discovery:    "Tìm hooks",
  };

  async function loadAgentActivity() {
    try {
      const res = await fetch("/api/agent-log?n=30");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const entries = data.entries || [];
      const tokens = data.today_tokens || {};
      $("#agent-activity-count").textContent = entries.length;
      $("#agent-activity-tokens").textContent = (
        (tokens.total || 0).toLocaleString()
      );
      const container = $("#agent-activity");
      if (!entries.length) {
        container.innerHTML = '<div class="agent-activity-empty">Chưa có hoạt động. Các sub-agent đang chờ trigger.</div>';
        return;
      }
      // Most recent first
      const recent = [...entries].reverse().slice(0, 15);
      container.innerHTML = recent.map(e => {
        const ts = e.ts ? new Date(e.ts).toLocaleTimeString('vi-VN', {hour: '2-digit', minute: '2-digit'}) : "";
        const agent = e.agent || "unknown";
        const agentVi = AGENT_VI[agent] || agent;
        const dec = e.decision || {};
        const approved = dec.approved !== false;
        const err = dec.error;
        const reasoning = (dec.reasoning || "").slice(0, 220);
        const ctx = (e.context_summary || "").slice(0, 60);
        const stakeMult = dec.stake_multiplier;
        const confDelta = dec.confidence_delta;
        const latency = e.latency_ms || 0;
        const tokIn = e.tokens_in || 0;
        const tokOut = e.tokens_out || 0;
        let statusIcon, statusClass;
        if (err) { statusIcon = "⚠️"; statusClass = "err"; }
        else if (!approved) { statusIcon = "🚫"; statusClass = "veto"; }
        else { statusIcon = "✅"; statusClass = "ok"; }
        let detailChips = "";
        if (stakeMult !== undefined && stakeMult !== 1.0 && stakeMult !== null) {
          detailChips += `<span class="chip">stake ×${stakeMult.toFixed(2)}</span>`;
        }
        if (confDelta !== undefined && confDelta !== 0 && confDelta !== null) {
          const sign = confDelta > 0 ? "+" : "";
          detailChips += `<span class="chip">conf ${sign}${(confDelta * 100).toFixed(1)}%</span>`;
        }
        if (latency) {
          detailChips += `<span class="chip chip-muted">${(latency / 1000).toFixed(1)}s</span>`;
        }
        if (tokIn || tokOut) {
          detailChips += `<span class="chip chip-muted">${tokIn}+${tokOut} tok</span>`;
        }
        return `
          <div class="agent-entry agent-${statusClass}">
            <div class="agent-entry-head">
              <span class="agent-icon">${statusIcon}</span>
              <span class="agent-name">${agentVi}</span>
              <span class="agent-time">${ts}</span>
            </div>
            <div class="agent-ctx">${ctx}</div>
            ${reasoning ? `<div class="agent-reasoning">${reasoning}${(dec.reasoning || "").length > 220 ? "…" : ""}</div>` : ""}
            ${err ? `<div class="agent-err">${err}</div>` : ""}
            ${detailChips ? `<div class="agent-chips">${detailChips}</div>` : ""}
          </div>
        `;
      }).join("");
    } catch (err) {
      $("#agent-activity").innerHTML = `<div class="agent-activity-empty">Không tải được (${err.message}).</div>`;
    }
  }

  // Provider names in Vietnamese for the chip row.
  const PROVIDER_VI = {
    gemini:     "Gemini",
    openrouter: "OpenRouter",
    groq:       "Groq",
    legacy:     "cũ",
    none:       "không có",
  };

  async function loadCostBreakdown() {
    try {
      const res = await fetch("/api/cost-breakdown");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const today = data.today || {};
      const configured = data.configured_providers || [];
      // Sort providers by call count descending so the top one is obvious.
      const rows = Object.entries(today).sort((a, b) => b[1].calls - a[1].calls);
      const totalCalls = rows.reduce((s, [, v]) => s + v.calls, 0) || 1;
      const chipsEl = $("#provider-chips");
      if (!configured.length) {
        chipsEl.innerHTML = '<span class="provider-chip provider-chip-warn">⚠ Chưa cấu hình provider free</span>';
        return;
      }
      const chipBits = configured.map(p => {
        const stats = today[p];
        if (!stats) {
          return `<span class="provider-chip provider-chip-idle">${PROVIDER_VI[p] || p} · sẵn sàng</span>`;
        }
        const pct = Math.round((stats.calls / totalCalls) * 100);
        const errTag = stats.errors ? ` · ${stats.errors} lỗi` : "";
        return `<span class="provider-chip provider-chip-active">${PROVIDER_VI[p] || p} · ${stats.calls} (${pct}%)${errTag}</span>`;
      });
      chipsEl.innerHTML = chipBits.join("");
    } catch (err) {
      $("#provider-chips").innerHTML = `<span class="provider-chip provider-chip-warn">${err.message}</span>`;
    }
  }

  // Auto-load on page open so Uncle doesn't have to click Làm mới /
  // Đề xuất từ AI every time. Both fire in parallel — odds from
  // the 60s cache is instant, recs take a couple seconds.
  loadOdds();
  loadRecs();
  loadRisk();
  loadStrategyScoreboard();
  loadMarketMix();
  loadAgentActivity();
  loadCostBreakdown();

  // Also auto-refresh both every 90s so the dashboard stays live
  // without manual clicking. Risk refresh is faster (30s) so halt
  // events surface promptly without a full page reload.
  setInterval(() => { loadOdds(); loadRecs(); loadStrategyScoreboard(); }, 90000);
  setInterval(loadRisk, 30000);
  setInterval(loadMarketMix, 45000);
  // Agent activity refreshes faster (10s) so Uncle sees live decisions.
  setInterval(loadAgentActivity, 10000);
  // Cost + provider-mix refresh every 20s.
  setInterval(loadCostBreakdown, 20000);
})();
