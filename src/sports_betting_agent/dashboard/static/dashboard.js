(() => {
  const $ = (sel) => document.querySelector(sel);
  const oddsList = $("#odds-list");
  const recList = $("#rec-list");
  const chatLog = $("#chat-log");
  const chatForm = $("#chat-form");
  const chatInput = $("#chat-input");
  const sportSelect = $("#sport-select");

  const fmtAmerican = (n) => (n > 0 ? `+${Math.round(n)}` : `${Math.round(n)}`);

  async function loadOdds() {
    oddsList.innerHTML = "Đang tải...";
    const sport = sportSelect.value;
    try {
      const res = await fetch(`/api/odds?sport=${encodeURIComponent(sport)}`);
      const data = await res.json();
      if (!data.games || data.games.length === 0) {
        oddsList.innerHTML = "<div class='card'>Không có trận nào.</div>";
        return;
      }
      oddsList.innerHTML = "";
      for (const game of data.games) {
        const ml = (game.lines || []).filter((l) => l.market === "moneyline");
        const homeLines = ml.filter((l) => l.selection.toLowerCase() === game.home_team.toLowerCase());
        const awayLines = ml.filter((l) => l.selection.toLowerCase() === game.away_team.toLowerCase());
        const bestHome = homeLines.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const bestAway = awayLines.sort((a, b) => (b.decimal || 0) - (a.decimal || 0))[0];
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="teams">${game.away_team} @ ${game.home_team}</div>
          <div class="meta">
            ${bestAway ? `${game.away_team} ${fmtAmerican(bestAway.american)} (${bestAway.book})` : ""}
            ${bestHome ? ` • ${game.home_team} ${fmtAmerican(bestHome.american)} (${bestHome.book})` : ""}
          </div>
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
      const res = await fetch("/api/recommendations");
      const data = await res.json();
      if (!data.recommendations || data.recommendations.length === 0) {
        recList.innerHTML = "<div class='card'>Chưa tìm thấy bet phù hợp.</div>";
        return;
      }
      recList.innerHTML = "";
      for (const rec of data.recommendations) {
        const div = document.createElement("div");
        div.className = "card rec";
        div.innerHTML = `
          <div class="teams">${rec.selection}</div>
          <div class="meta">
            ${rec.away_team} @ ${rec.home_team} • ${rec.market} • ${fmtAmerican(rec.american)} @ ${rec.book}
          </div>
          <div class="meta">
            Chiến lược: <strong>${rec.strategy}</strong> •
            Tự tin ${Math.round(rec.confidence * 100)}% • Edge ${(rec.edge * 100).toFixed(1)}%
          </div>
          <div class="reason">${rec.reasoning || ""}</div>
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

  $("#btn-refresh").addEventListener("click", loadOdds);
  $("#btn-recs").addEventListener("click", loadRecs);
  sportSelect.addEventListener("change", loadOdds);

  loadOdds();
})();
