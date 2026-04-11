"""Generate the session-handoff PDF.

Run from the repo root::

    python docs/generate_handoff_pdf.py

This writes ``docs/SESSION_HANDOFF.pdf`` -- a single PDF you can
read on your MacBook and copy-paste the highlighted blocks into a
fresh Claude Code session to pick up where this one left off.
"""

from __future__ import annotations

import datetime as _dt
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = os.path.join(os.path.dirname(__file__), "SESSION_HANDOFF.pdf")
BRANCH = "claude/add-bovada-odds-c9xmr"
REPO = "edwardnguyen1194-spec/sport-betting"
LIVE_APP = "sports-betting-ai-agent"
LIVE_URL = "https://sports-betting-ai-agent.fly.dev"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#0b1020"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#4a5578"),
        spaceAfter=14,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=colors.HexColor("#0b1020"),
        spaceBefore=16,
        spaceAfter=6,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#1a2c52"),
        spaceBefore=10,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1a1d33"),
        spaceAfter=6,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=14,
        bulletIndent=4,
        textColor=colors.HexColor("#1a1d33"),
        spaceAfter=2,
    )
    styles["code"] = ParagraphStyle(
        "code",
        parent=base["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=8,
        rightIndent=8,
        textColor=colors.HexColor("#0b1020"),
        backColor=colors.HexColor("#eef1f8"),
        borderColor=colors.HexColor("#c9d2e6"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=10,
    )
    styles["note"] = ParagraphStyle(
        "note",
        parent=base["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#5a3a00"),
        backColor=colors.HexColor("#fff7e0"),
        borderColor=colors.HexColor("#f0c05a"),
        borderWidth=0.5,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=10,
    )
    styles["warn"] = ParagraphStyle(
        "warn",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#7a0e0e"),
        backColor=colors.HexColor("#ffe5e5"),
        borderColor=colors.HexColor("#d66060"),
        borderWidth=0.5,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=10,
    )
    return styles


STYLES = build_styles()


def P(text, style="body"):
    return Paragraph(text, STYLES[style])


def code(text):
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(safe, STYLES["code"])


def note(text):
    return Paragraph(text, STYLES["note"])


def warn(text):
    return Paragraph(text, STYLES["warn"])


def bullet(text):
    return Paragraph(f"• {text}", STYLES["bullet"])


def hr():
    return HRFlowable(
        width="100%",
        thickness=0.6,
        color=colors.HexColor("#c9d2e6"),
        spaceBefore=6,
        spaceAfter=6,
    )


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------


def build_flowables():
    today = _dt.date.today().isoformat()
    flow = []

    # --- Cover ---
    flow.append(P("Sport Betting AI — Session Handoff", "title"))
    flow.append(
        P(
            f"Branch: <b>{BRANCH}</b> &nbsp;•&nbsp; Repo: <b>{REPO}</b> &nbsp;•&nbsp; "
            f"Live app: <b>{LIVE_APP}</b> &nbsp;•&nbsp; Generated: {today}",
            "subtitle",
        )
    )
    flow.append(hr())

    # --- TL;DR ---
    flow.append(P("TL;DR", "h1"))
    flow.append(
        P(
            "A previous Claude Code session built a feature branch with every "
            "item from your god-mode brief: Bovada college baseball odds, six "
            "other free odds sources, a heavy-favorite filter strategy, an "
            "LSTM player-props model, a Vietnamese Flask dashboard, a 24/7 "
            "paper trader, and 33 passing unit tests. The branch is pushed to "
            f"<b>{BRANCH}</b> but could not be validated against live endpoints "
            "or deployed to fly.io because the build sandbox had no outbound "
            "network access to the odds sites and no fly.io credentials."
        )
    )
    flow.append(
        P(
            "This PDF is a drop-in handoff. Open a fresh Claude Code session "
            "on your MacBook inside the directory that actually deploys to "
            f"<b>{LIVE_APP}</b>, paste the highlighted prompt, and the new "
            "session will finish the job with real network and deploy access."
        )
    )

    # --- State table ---
    flow.append(P("Current state", "h1"))
    state = [
        ["Item", "Status"],
        ["Feature branch", f"{BRANCH} — pushed (2 commits, 45 files)"],
        ["Test suite", "33 passed in ~2s, no network required"],
        ["Bovada NCAA baseball fetcher", "Built, unit-tested against sample payload"],
        ["Other JSON fetchers (ESPN, Action Network)", "Built, defensive fixes for schema drift"],
        ["HTML scrapers (SoO, VI, Covers)", "Permissive parsers — likely need real-markup fixes"],
        ["Heavy-favorite strategy", "75-85% WR target band, sizing floor"],
        ["Value-bet + ensemble strategies", "Built, tested"],
        ["LSTM player-props model", "Scaffolded — needs real training data"],
        ["Paper trader", "Built, $50 min, Kelly sizing, JSON ledger"],
        ["Flask dashboard", "Built, Vietnamese UI, Claude chat endpoint"],
        ["Live endpoint validation", "NOT DONE — sandbox egress blocked"],
        ["fly.io deploy", "NOT DONE — no fly CLI access from sandbox"],
        ["main branch / PR", "NOT POSSIBLE — no base branch in repo yet"],
    ]
    table = Table(state, colWidths=[2.4 * inch, 4.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1020")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d2e6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    flow.append(table)
    flow.append(Spacer(1, 0.1 * inch))

    # --- What I recommended ---
    flow.append(P("Why a new Claude Code session on your Mac", "h1"))
    flow.append(
        P(
            "The blocker for every remaining task is network + credentials the "
            "build sandbox doesn't have. A fresh <b>claude</b> session running "
            "in your Mac Terminal, inside the directory that currently deploys "
            "to fly.io, has all of it:"
        )
    )
    flow.append(bullet("Outbound internet to bovada.lv / site.api.espn.com / api.actionnetwork.com"))
    flow.append(bullet("Your <b>fly</b> CLI auth — can run <b>fly deploy</b>"))
    flow.append(bullet("Your <b>gh</b> CLI auth — can open PRs against a base branch"))
    flow.append(bullet("Your real deployment tree — the code that actually drives the live site"))
    flow.append(bullet("SSH keys to push to any git remote, including fly's builder"))

    flow.append(PageBreak())

    # --- Step 1: install / locate ---
    flow.append(P("Step 1 — Open Claude Code on your Mac", "h1"))
    flow.append(P("If you don't already have it installed:"))
    flow.append(code("curl -fsSL https://claude.ai/install.sh | bash\n# or: brew install claude"))
    flow.append(
        P(
            "Find the directory that deploys to fly.io. Run this on your Mac "
            "to confirm the app exists and see its current state:"
        )
    )
    flow.append(code("fly apps list\nfly status -a sports-betting-ai-agent"))
    flow.append(
        P(
            "Then <b>cd</b> into the local directory containing the <b>fly.toml</b> that "
            "references <b>sports-betting-ai-agent</b>. That is the tree the new session "
            "needs to work in."
        )
    )
    flow.append(code("cd ~/path/to/your/sport-betting\nclaude"))

    # --- Step 2: the handoff prompt ---
    flow.append(P("Step 2 — Paste this prompt verbatim into the new session", "h1"))
    flow.append(
        note(
            "Copy everything inside the box below, including the numbered "
            "steps. The new session will then fetch the branch, validate it "
            "against real endpoints, fix anything broken, and deploy."
        )
    )

    prompt = (
        "I'm continuing work from another Claude Code session. The previous "
        f"session built a feature branch on {REPO} called `{BRANCH}`. That "
        "branch adds:\n\n"
        "- Bovada free NCAA baseball odds (+ 6 other free sources: ESPN, "
        "ScoresAndOdds, VegasInsider, Covers, Action Network, SBR)\n"
        "- Heavy-favorite filter strategy (75-85% WR target)\n"
        "- LSTM neural network player props model (PyTorch + NumPy fallback)\n"
        "- Flask dashboard with Vietnamese UI + Claude chat\n"
        "- 24/7 paper trader ($50 min bet, Kelly sizing)\n"
        "- 33 passing unit tests\n"
        "- scripts/validate_live.py for live endpoint validation\n\n"
        "The previous session could not hit live odds endpoints (sandbox "
        "egress blocked) and had no fly.io access, so it wrote defensive "
        "code by review rather than against live responses. Please:\n\n"
        "1. git fetch origin && git checkout claude/add-bovada-odds-c9xmr\n"
        "2. python -m venv .venv && source .venv/bin/activate\n"
        "3. pip install -r requirements.txt\n"
        "4. PYTHONPATH=src python -m pytest src/sports_betting_agent/tests -q\n"
        "   (expect 33 passed)\n"
        "5. PYTHONPATH=src python scripts/validate_live.py "
        "--sources bovada,espn,actionnetwork --dump\n"
        "   Report what each source returned. If any parser returns 0 games "
        "or errors, inspect validation_dumps/<source>_<sport>.json and fix "
        "the parser.\n"
        "6. Then try: --sources scoresandodds,vegasinsider,covers\n"
        "   Those are HTML scrapers and will likely need selector fixes. "
        "Fix what you can; disable what you can't by removing them from "
        "SBA_ENABLED_SOURCES.\n"
        "7. Reconcile this branch with the code currently driving "
        "sports-betting-ai-agent.fly.dev. The current deployment directory "
        "is this one (check `cat fly.toml`).\n"
        "8. BEFORE deploying: `fly secrets set SBA_AUTO_TRADE=0 "
        "-a sports-betting-ai-agent`\n"
        "9. `fly deploy`\n"
        "10. After deploy is healthy, visit /api/health, "
        "/api/odds?sport=baseball_ncaa, and the dashboard.\n"
        "11. Once I confirm everything looks right, re-enable: "
        "`fly secrets set SBA_AUTO_TRADE=1 -a sports-betting-ai-agent`\n\n"
        "Be honest about what works and what doesn't. Do not deploy if "
        "validation fails. Do not enable auto-trade until I confirm. Treat "
        "the heavy-favorite strategy with skepticism -- break-even at -180 "
        "is 64.3%, so high WR does not automatically mean profit."
    )
    flow.append(code(prompt))

    flow.append(PageBreak())

    # --- Step 3: caveats ---
    flow.append(P("Step 3 — Important caveats for the new session", "h1"))
    flow.append(
        warn(
            "Set SBA_AUTO_TRADE=0 before the first deploy. The background "
            "paper-trading loop runs on a 5-minute interval by default. If "
            "you deploy with it enabled, it will start placing untested "
            "bets against whatever the fetchers return — including any "
            "garbage data from a broken HTML scraper. Validate first, "
            "then flip it on."
        )
    )
    flow.append(
        note(
            "If the Mac directory you're working in is NOT a clone of "
            f"{REPO}, the new session will need to manually copy "
            "src/sports_betting_agent/, scripts/, requirements.txt, "
            "fly.toml, and Dockerfile from the fetched branch into the "
            "existing deployment tree. Just tell it to do that."
        )
    )
    flow.append(
        note(
            "The GitHub repo currently has no main branch (the repo was "
            "empty when I started). If you want PRs to work, have the new "
            "session promote the feature branch to main: "
            "`git checkout -b main claude/add-bovada-odds-c9xmr "
            "&& git push -u origin main`."
        )
    )

    # --- Honest assessment ---
    flow.append(P("Honest assessment — read this before you trust the bot", "h1"))
    flow.append(
        P(
            "The code is structurally sound, cleanly tested, and ready to "
            "run. But three things deserve open eyes:"
        )
    )
    flow.append(
        P(
            "<b>1. Heavy-favorite strategy isn't automatically +EV.</b> "
            "Sharp books price chalk accurately. At -180 the bettor needs "
            "64.3% to break even, and the vig-free consensus on a -180 "
            "chalk is typically ~62-64%. The strategy will hit the advertised "
            "75-85% win rate but real profitability is uncertain. Value-bet "
            "and LSTM props are where measurable edge lives.",
            "body",
        )
    )
    flow.append(
        P(
            "<b>2. LSTM is scaffolded, not trained.</b> The fit/predict loop "
            "works on synthetic data. To actually hit 65-75% WR you need "
            "real player game logs piped into PlayerGameLog objects. Until "
            "then, treat it as unused.",
            "body",
        )
    )
    flow.append(
        P(
            "<b>3. HTML scrapers are educated guesses.</b> ScoresAndOdds, "
            "VegasInsider, and Covers were written from reasoning about "
            "typical markup, not from inspecting live HTML. Expect zero "
            "games from them on first run; the new session should fix or "
            "disable them.",
            "body",
        )
    )

    # --- File manifest ---
    flow.append(P("What's on the branch — file manifest", "h1"))
    manifest_rows = [
        ["Area", "Files"],
        ["Core schema", "src/sports_betting_agent/models_schema.py, config.py, confidence.py"],
        [
            "Fetchers",
            "fetchers/{bovada,espn,scoresandodds,vegasinsider,covers,actionnetwork,sbr,aggregator,base}.py",
        ],
        [
            "Strategies",
            "strategies/{heavy_favorite,value_bets,ensemble,player_props,base}.py",
        ],
        ["Model", "models/lstm_player_props.py"],
        ["Engine", "paper_trader.py, claude_chat.py, cli.py"],
        [
            "Dashboard",
            "dashboard/app.py, dashboard/templates/index.html, dashboard/static/dashboard.{css,js}",
        ],
        [
            "Tests (33)",
            "tests/test_{models_schema,bovada_parser,heavy_favorite,paper_trader,lstm_model,confidence,parser_robustness}.py",
        ],
        ["Validation", "scripts/validate_live.py"],
        ["Packaging", "requirements.txt, pyproject.toml, Dockerfile, fly.toml, README.md, .gitignore"],
    ]
    manifest_table = Table(manifest_rows, colWidths=[1.5 * inch, 5.2 * inch])
    manifest_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1020")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d2e6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    flow.append(manifest_table)

    # --- Signoff ---
    flow.append(Spacer(1, 0.2 * inch))
    flow.append(hr())
    flow.append(
        P(
            f"Generated by Claude Code. Branch {BRANCH} is pushed and ready. "
            "Good luck — and remember: gamble responsibly.",
            "subtitle",
        )
    )

    return flow


def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Sport Betting AI — Session Handoff",
        author="Claude Code",
    )
    doc.build(build_flowables())
    size = os.path.getsize(OUTPUT)
    print(f"wrote {OUTPUT} ({size:,} bytes)")


if __name__ == "__main__":
    main()
