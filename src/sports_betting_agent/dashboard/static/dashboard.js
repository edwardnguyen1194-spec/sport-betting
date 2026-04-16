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

  $("#btn-refresh").addEventListener("click", loadOdds);
  $("#btn-recs").addEventListener("click", loadRecs);
  if ($("#btn-best-picks")) $("#btn-best-picks").addEventListener("click", loadBestPicks);
  sportSelect.addEventListener("change", loadOdds);

  // Show ready message instead of auto-loading (which is slow)
  oddsList.innerHTML = "<div class='card'>Nhấn \"Làm mới\" để tải kèo.</div>";
})();
