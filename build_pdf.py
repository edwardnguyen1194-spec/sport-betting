"""Generate sports_betting_report.pdf combining codebase strategies + GitHub/MCP research."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_LEFT

OUT = "/home/user/sport-betting/sports_betting_report.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=12,
                    textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15, spaceBefore=14,
                    spaceAfter=8, textColor=colors.HexColor("#2c5282"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#2d3748"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14,
                      alignment=TA_LEFT, spaceAfter=6)
CAPTION = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=8,
                         textColor=colors.HexColor("#718096"), spaceAfter=6)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=11,
                      backColor=colors.HexColor("#f7fafc"), borderPadding=4,
                      textColor=colors.HexColor("#1a202c"))

def table(data, col_widths=None, header_bg="#2c5282"):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t

def p(text, style=BODY):
    return Paragraph(text, style)

story = []

# ========== COVER ==========
story += [
    Spacer(1, 1.2*inch),
    p("Sports Betting Intelligence Report", H1),
    p("Strategy Engine + GitHub/MCP Ecosystem Reference", H2),
    Spacer(1, 0.2*inch),
    p("Prepared for: <b>Uncle Phung</b>", BODY),
    p("Source codebase: <b>sports-betting-ai-agent</b> (fly.io deployment)", BODY),
    p("Report date: 2026-04-21", BODY),
    Spacer(1, 0.4*inch),
    p("""<b>Reality check up front:</b> no legitimate strategy produces a
       "super high" win rate. Professional bettors hit 54-58% long-term.
       Anyone promising more is running arbitrage, promo-hunting, or lying.
       The tools and strategies in this report reflect that reality —
       they focus on <b>edge discovery, closing line value (CLV), and
       bankroll math</b>, not magic.""", BODY),
    PageBreak(),
]

# ========== PART 1: YOUR CODEBASE ==========
story += [
    p("Part 1 — Your Codebase: Strategy Engine", H1),
    p("Analysis of the 9 strategies currently implemented in <i>src/sports_betting_agent/strategies/</i>.", CAPTION),
]

story += [p("Strategy Summary Table", H2)]
strat_data = [
    ["Strategy", "Market", "Min Edge", "Conf Cap", "Base Rate"],
    ["Elo Spread",          "Spread",        "2.5%",   "58%", "–"],
    ["Public Fade",         "Spread/Total",  "2.0%",   "58%", "63.8% ATS (NFL)"],
    ["Price Dispersion",    "Spread/Total",  "2.5%",   "62%", "–"],
    ["MLS Travel Fatigue",  "Spread",        "2.0%",   "56%", "3% dog edge"],
    ["Spread Value",        "Spread",        "1.5%",   "–",   "–"],
    ["Total Value",         "Total",         "2.5%",   "–",   "–"],
    ["Steam Follow",        "Spread/Total",  "2.0%",   "53%", "~57-60%"],
    ["Total Projection",    "Total",         "sport-dep.", "57%", "–"],
    ["Reverse Line Mvmt",   "Spread/Total",  "2.0%",   "60%", "56-58% ATS"],
    ["Ensemble (combined)", "Spread/Total",  "positive", "72%", "varies"],
]
story += [table(strat_data, col_widths=[1.5*inch, 1.1*inch, 0.9*inch, 0.7*inch, 1.4*inch])]
story += [Spacer(1, 0.15*inch)]

# ---- Strategies in detail ----
story += [p("Tier 1 — Strongest Historical Signal", H2)]

story += [
    p("1. Reverse Line Movement (RLM) — 56-58% ATS", H3),
    p("""Highest-conviction single strategy. Requires two signals firing
       simultaneously: (a) 60%+ of public money on one side, and
       (b) the line moves the opposite direction (sharps disagree with
       the crowd). When both align, confidence reaches 60% — the highest
       cap of any single strategy.""", BODY),

    p("2. Public Fade — 63.8% ATS (NFL dogs, 4-year avg)", H3),
    p("""Fades teams with 65%+ of public bets. Strongest on NFL
       underdogs getting &lt;40% of tickets. The market systematically
       overvalues popular/heavily-bet favorites.""", BODY),
]

story += [p("Tier 2 — Quantitative Sharp-Book Signals", H2)]
story += [
    p("3. Steam Follow — ~57-60%", H3),
    p("""Follows when 3+ sharp books (Pinnacle, Circa, BookMaker) move
       the same direction. Confidence cap is deliberately conservative
       (53%) because the edge decays fast — the opportunity closes
       quickly after sharp action.""", BODY),

    p("4. Elo Spread", H3),
    p("""In-house Elo power ratings vs. market spread. Min gap 1.5 pts
       (NFL), 2.0 pts (NBA), 0.5 runs (MLB), 0.4 goals (soccer).
       Only bets sharp books. Confidence cap 58%.""", BODY),

    p("5. Price Dispersion", H3),
    p("""Universal cross-sport strategy. Groups lines by
       (market, selection, line number) across all books. Fires when one
       book is 2.5%+ cheaper than the 3+ book consensus. Cap 62% —
       highest of any single strategy.""", BODY),
]

story += [p("Tier 3 — Model-Based Projection", H2)]
story += [
    p("6. Total Projection", H3),
    p("""Blends rolling team scoring averages with park/weather/umpire
       adjustments, then regresses 50% toward the book line. Per-sport
       edge thresholds: 1.5 runs (MLB), 6.25 pts (NBA), 0.75 goals (NHL),
       3.0 pts (NFL), 4.25 pts (NCAAF), 0.5 goals (MLS), 0.25 (EPL).""", BODY),

    p("7. Spread Value / Total Value", H3),
    p("""Devigs sharp books to find true fair price, then checks if any
       soft book offers 1.5-2.5%+ better. Pure cross-book edge.""", BODY),

    p("8. MLS Home Travel Fatigue — ~3% ATS underperformance for travelers", H3),
    p("""Niche but real: MLS away teams crossing 2+ timezones on &lt;4
       days rest cover at a measurable deficit. Bets home spreads at -0.5
       or -1 only.""", BODY),
]

story += [p("The Ensemble (Combined Engine)", H2)]
story += [
    p("""When multiple strategies independently agree on the same
       selection, confidences are combined via <b>Bayesian log-odds</b>
       (not simple averaging).""", BODY),
]
ensemble_data = [
    ["Agreeing strategies", "Effective confidence"],
    ["1 strategy at 55%",   "~55%"],
    ["2 independent at 55%", "~60%"],
    ["3 independent at 55%", "~65%"],
    ["Hard cap",            "72%"],
]
story += [table(ensemble_data, col_widths=[2.5*inch, 2.0*inch]),
          Spacer(1, 0.1*inch),
          p("""Any single strategy is marginal. Two or three agreeing
             independently on the same play is a strong signal — those
             are the ensemble's 65-72% confidence bets.""", BODY)]

story += [p("Design Rules the System Enforces", H2)]
rules = [
    ["Rule", "Why"],
    ["Spreads & totals only — no moneylines", "Moneylines have larger vig and are harder to beat consistently"],
    ["Sharp books only for line-reading", "Pinnacle/Circa/BookMaker set the true market; square books lag"],
    ["Min 2.5% edge before betting", "Below this, the vig eats the edge"],
    ["Market regression 40-60% on projection models", "Pure model predictions overfit; shrinking toward line is more robust"],
    ["Per-strategy confidence caps (53-72%)", "Prevents overconfidence; keeps Kelly bet sizing conservative"],
]
story += [table(rules, col_widths=[2.4*inch, 4.0*inch])]

story += [PageBreak()]

# ========== PART 2: GITHUB & MCP ECOSYSTEM ==========
story += [
    p("Part 2 — GitHub & MCP Ecosystem", H1),
    p("Top repositories, MCP servers, and tools to extend or benchmark against.", CAPTION),
]

story += [p("ML-Based Prediction Models", H2)]
ml_data = [
    ["Repo", "Stars", "What it does"],
    ["kyleskom/NBA-Machine-Learning-Sports-Betting", "1,630", "NBA ML with Keras/TensorFlow + GPT analysis"],
    ["georgedouzas/sports-betting", "691", "Scikit-learn framework for backtesting strategies"],
    ["NBA-Betting/NBA_AI", "100", "NBA predictions with XGBoost/PyTorch"],
    ["day-mon/sports-betting-ai", "101", "Deep-learning NBA winner predictor (Rust)"],
    ["jkrusina/SoccerPredictor", "112", "LSTM soccer time-series model"],
]
story += [table(ml_data, col_widths=[2.8*inch, 0.6*inch, 3.0*inch])]

story += [p("Analytical / Math Libraries", H2)]
math_data = [
    ["Repo", "Stars", "What it does"],
    ["nautechsystems/nautilus_trader", "22,137", "Rust event-driven engine (supports sports betting)"],
    ["martineastwood/penaltyblog", "157", "Poisson / Dixon-Coles / Elo / Pi-ratings for football"],
    ["sedemmler/WagerBrain", "300", "Essential betting math: Kelly, vig removal, EV"],
    ["gotoConversion/goto_conversion", "108", "Shin / power devigging (Kaggle gold-medal winner)"],
]
story += [table(math_data, col_widths=[2.8*inch, 0.6*inch, 3.0*inch])]

story += [p("Arbitrage & Line-Shopping", H2)]
arb_data = [
    ["Repo", "Stars", "What it does"],
    ["pretrehr/Sports-betting", "503", "Multi-bookmaker arb (Pinnacle, Betfair, Bwin, …)"],
    ["personal-coding/Live-Sports-Arbitrage-Bet-Finder", "275", "Live arb across FanDuel / DK / William Hill"],
    ["ryankrumenacker/sports-betting-arbitrage-project", "195", "Statistical arbitrage notebooks"],
    ["daankoning/ArbitrageFinder", "92", "Simple Python arb detector"],
    ["carterlasalle/SportsArbFinder", "10", "Odds-API-powered multi-region arb tool"],
]
story += [table(arb_data, col_widths=[2.8*inch, 0.6*inch, 3.0*inch])]

story += [p("Kelly / Bankroll", H2)]
kelly_data = [
    ["Repo", "Stars", "What it does"],
    ["chrisgillam/polymarket_gambot", "24", "Pinnacle vs Polymarket probabilistic + Kelly sizing bot"],
]
story += [table(kelly_data, col_widths=[2.8*inch, 0.6*inch, 3.0*inch])]

story += [p("Data Scrapers & APIs", H2)]
data_data = [
    ["Repo", "Stars", "What it does"],
    ["jordantete/OddsHarvester", "154", "Playwright scraper for oddsportal.com"],
    ["cvidan/bet365-scraper", "154", "Selenium-based Bet365 scraper"],
    ["dos-2/oddshub", "119", "Terminal UI for odds analysis (Go)"],
    ["sportsdataverse/oddsapiR", "8", "R wrapper for The Odds API"],
]
story += [table(data_data, col_widths=[2.8*inch, 0.6*inch, 3.0*inch])]

story += [p("MCP Servers for Claude / AI Agents", H2)]
mcp_data = [
    ["Server", "Purpose"],
    ["Cloudbet Sports MCP",       "Live markets, odds, events from Cloudbet"],
    ["Wagyu Sports MCP",          "Bridge to The Odds API"],
    ["Apify Sports Analytics",    "12 tools: ESPN data, DFS projections, lineup optimization, 25+ leagues"],
    ["FootballBin",               "Premier League + Champions League predictions"],
    ["sports-leader-mcp",         "Free: scores, odds, injuries across 17 sports / 139 leagues"],
    ["kitchenchem Odds API MCP",  "Odds comparison across bookmakers"],
]
story += [table(mcp_data, col_widths=[2.0*inch, 4.4*inch])]

story += [PageBreak()]

# ========== PART 3: THE REAL PLAYBOOK ==========
story += [p("Part 3 — The Real Playbook", H1)]

story += [p("Strategies Ranked by Real Win Rate", H2)]
wr_data = [
    ["Strategy", "Real Win Rate", "Profit Source"],
    ["Arbitrage",                   "~100% per bet", "Multi-book price mismatches (not prediction)"],
    ["Bonus/Promo hunting",         "~99% EV",       "Free-bet conversion"],
    ["Market-making (exchanges)",   "52-55%",        "Spread capture on Betfair/Polymarket"],
    ["CLV edge vs Pinnacle close",  "54-58%",        "Beating Pinnacle's close by 2-5%"],
    ["Line shopping at sharp books","+3-5% ROI",     "-105 vs -110 compounds"],
    ["Reverse Line Movement",       "56-58%",        "Already in your system"],
    ["Public Fade (NFL dogs)",      "58-64% ATS",    "Already in your system"],
    ["Steam Following",             "57-60%",        "Already in your system"],
]
story += [table(wr_data, col_widths=[2.2*inch, 1.2*inch, 3.0*inch])]

story += [p("What Actually Wins Long-Term", H2)]
story += [
    p("1. <b>Beat Pinnacle's closing line by 2-5%</b> → 15-25% annual ROI", BODY),
    p("2. <b>Shop every bet across 6+ books</b> (Pinnacle, Circa, BookMaker, DK, FD, MGM)", BODY),
    p("3. <b>Bet early when lines open soft</b>, before sharp money shapes them", BODY),
    p("4. <b>Devig sharp lines</b> to find true probability — use goto_conversion or Crazy Ninja Odds devigger", BODY),
    p("5. <b>Use fractional Kelly (¼ or ½)</b> — never full Kelly; pros cap at 2.5% per bet", BODY),
    p("6. <b>Focus on liquid markets</b>: NFL spreads, NBA totals, MLB moneylines, Soccer Asian handicaps", BODY),
]

story += [p("Suggested Hook Additions for Your Repo", H2)]
story += [
    p("<b>PreToolUse on Bash(settle)</b> — auto-check CLV before settlement", BODY),
    p("<b>PostToolUse on bet placement</b> — log CLV to the tracker", BODY),
    p("<b>SessionStart hook</b> — fetch overnight line moves + steam alerts", BODY),
    p("<b>Stop hook</b> — daily P&amp;L summary with CLV histogram", BODY),
]

story += [p("Honest Caveats", H2)]
story += [
    p("• Anyone promising \"90%+ win rate\" is selling a scam.", BODY),
    p("• Pinnacle, Circa, BookMaker, BetOnline tolerate winners. "
      "FanDuel / DraftKings / MGM <b>limit sharps fast</b>.", BODY),
    p("• The edge comes from <b>process</b> — line shopping, CLV tracking, "
      "bankroll discipline — not from one magic formula.", BODY),
]

story += [PageBreak()]

# ========== APPENDIX: SOURCES ==========
story += [p("Appendix — Sources", H1)]
sources = [
    ("Cloudbet Sports MCP Server",       "https://cloudbet.github.io/wiki/en/docs/sports/api/mcp_integration/"),
    ("Awesome MCP Servers - Sports",     "https://mcpservers.org/servers/apify-com-nexgendata-sports-mcp-server"),
    ("Wagyu Sports MCP Market",          "https://mcpmarket.com/server/wagyu-sports"),
    ("CLV Guide 2026 (XCLSV)",           "https://xclsvmedia.com/closing-line-value-clv-explained-the-complete-guide-for-sports-bettors-in-2026/"),
    ("OddsJam CLV Tracker",              "https://oddsjam.com/betting-education/closing-line-value"),
    ("Unabated CLV Calculator",          "https://unabated.com/betting-calculators/closing-line-value-calculator"),
    ("Boyd's Bets - Beating CLV",        "https://www.boydsbets.com/closing-line-value/"),
    ("Sharp Football CLV Guide",         "https://www.sharpfootballanalysis.com/sportsbook/clv-betting/"),
    ("Kelly Criterion (Wikipedia)",      "https://en.wikipedia.org/wiki/Kelly_criterion"),
    ("OddsJam Kelly Calculator",         "https://oddsjam.com/betting-calculators/kelly-criterion"),
    ("betstamp Kelly Guide",             "https://betstamp.com/education/kelly-criterion"),
    ("Sports Betting Dime Kelly Guide",  "https://www.sportsbettingdime.com/guides/strategy/kelly-criterion/"),
    ("TipMaster Kelly Guide",            "https://tipmaster.ai/blog/kelly-criterion-sports-betting-mathematical-edge"),
    ("XCLSV Devigging Guide 2026",       "https://xclsvmedia.com/how-to-devig-betting-lines-in-2026-the-complete-guide-to-finding-true-odds/"),
    ("XCLSV Line Shopping 2026",         "https://xclsvmedia.com/line-shopping-guide-2026-how-to-find-the-best-odds-and-save-thousands/"),
    ("Pikkit - Sharpest Sportsbooks",    "https://pikkit.com/blog/which-sportsbooks-are-sharp"),
    ("Unabated - Market Makers",         "https://unabated.com/articles/who-sets-the-sports-betting-line-market-makers"),
    ("Bet Hero - Pinnacle as Sharp Ref", "https://betherosports.com/blog/how-to-use-pinnacle"),
    ("PickTheOdds - Sharp Edges",        "https://picktheodds.app/en/blog/sharp-sportsbooks-what-they-are-and-how-to-use-them-to-find-edges"),
]
for title, url in sources:
    story.append(p(f"• {title} — <link href='{url}' color='blue'>{url}</link>", BODY))

# ========== BUILD ==========
doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.7*inch, rightMargin=0.7*inch,
    topMargin=0.7*inch, bottomMargin=0.7*inch,
    title="Sports Betting Intelligence Report",
    author="sports-betting-ai-agent",
)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#718096"))
    canvas.drawString(0.7*inch, 0.4*inch,
        "Sports Betting Intelligence Report  •  For Uncle Phung  •  2026-04-21")
    canvas.drawRightString(letter[0]-0.7*inch, 0.4*inch, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}")
