# AURUM V2

**An AI-native quantitative research firm that discovers, validates, debates, deploys, and learns from its own investment research — including its own mistakes.**

Most backtesting projects show "here's a model that predicts returns." AURUM V2 demonstrates something harder to fake: a system that generates a hypothesis, has it picked apart by an adversarial AI committee, gets it backtested, watches the proposed fix fail in a different way, writes that failure down, and produces a *better* hypothesis next time — all with full provenance in Postgres.

This is not a claim. It's a traceable chain across four hypothesis generations, three independent committee debates, and two rounds of empirical validation, documented below with the actual output.

---

## The thesis

V1 (a separate, completed project — [`aurum`](#)) answers *"what should my portfolio do today?"*

V2 answers a fundamentally different question: *"what investment research should exist?"*

Every component in this system exists to support one loop:

```
observe market → generate hypothesis → backtest → validate statistically
→ debate (bull / bear / risk / judge) → deploy to paper trading
→ monitor for decay → extract lessons from failure → repeat, smarter
```

The loop is the product. Everything else is infrastructure in service of it.

---

## What makes this different

Most AI-assisted trading research projects stop at "here's a backtest with a good Sharpe ratio." AURUM V2 is built around the assumption that **a single backtest number proves almost nothing**, and the system is structured to actively work against its own optimism:

- A **statistical validator** requires t-test significance, Sharpe significance, sub-period stability, and adequate sample size before any hypothesis can reach committee — 3 of 4 checks must pass, or the hypothesis is held.
- A **multi-agent debate committee** (Bull, Bear, Risk, Judge) argues over every hypothesis that passes validation. The Judge's decision is only valid if it explicitly engages with the Bear's strongest objection — it cannot simply side with the better Sharpe ratio.
- A **Research Memory** layer extracts structured lessons from debates and failed backtests, and those lessons are retrieved and applied the next time the Research Scientist generates a hypothesis — provably, not just in theory.
- A **Continuous Learning** module walks deployed hypotheses forward through real historical data to check whether the committee's specific concerns actually materialized.

The result is a system that caught its own design flaw — a fixed-threshold volatility circuit-breaker — independently through three different methods (theoretical debate, historical backtest, and forward paper-trading simulation), and used that finding to produce a structurally different next-generation hypothesis.

---

## Proof: a real self-correction chain

This is the actual lineage traced through the system, not a hypothetical example.

**Hypothesis #3 and #4** (large-cap momentum + earnings revision, VIX&lt;15 gate, 21–42 day holding periods) were independently debated. In both cases, the **Bear Agent** identified the same structural flaw: a VIX&lt;15 entry gate cannot function as an exit signal, because historical vol spikes (Feb 2018, Mar 2020, 2022) move from sub-15 to 25–40+ within 3–10 trading days — faster than the holding period can react. The committee's decisions diverged (REQUEST_REVISION for #3, conditional APPROVE for #4) based on whether a statistical review had been completed — the AI committee penalized the hypothesis that skipped a governance stage.

This finding was extracted into two **Research Memory** entries (`timing_mismatch`, `selection_bias`).

**Hypothesis #7** was generated next, with those two memories retrieved and supplied to the Research Scientist. The resulting hypothesis explicitly decoupled entry and exit logic: a 10-day holding period (down from 21–42) plus a dedicated `intraperiod_vix_circuit_breaker` component, with the generated rationale stating the holding period was "deliberately shortened... below the 15-day threshold identified in prior failure analysis."

Backtesting the circuit-breaker honestly (simulating actual daily VIX checks against historical data, not just the entry signal) revealed it fired in **69.5% of holding periods** — collapsing Sharpe from 1.14 to 0.31 while only marginally improving drawdown (-27.0% → -23.6%). The fix was real, but miscalibrated.

This was extracted into a third memory (`circuit_breaker_overcalibration`), with an explicit constraint: *trigger frequency must be backtested before deployment; if it exceeds ~15–20% of holding periods, the threshold is too sensitive.*

**Hypothesis #8** was generated with all three memories available. It replaced the fixed VIX=18 threshold with a dynamically recalibrated 88th-percentile threshold, explicitly pre-computing and stating its expected trigger frequency (~12%) against the memory's stated acceptable range — and added a fifth signal component (`momentum_crowding_risk_score`) addressing crowding risk the Bear had raised independently in *both* prior debates, without that lesson having been formalized into memory at all.

Debating #8 immediately surfaced the next layer of the problem: the Bear Agent identified that the 88th-percentile calibration was circular (the threshold and its acceptable frequency range were derived from the same historical sample, with zero out-of-sample validation), and that the trailing 504-day lookback would have been anchored to calm 2018–2020 data right as the March 2020 crisis began — meaning the breaker would have been *systematically too slow* exactly when needed. The Judge built this into mandatory paper-trading conditions: walk-forward validation of the firing frequency before any live capital consideration.

Separately, **forward paper-trading simulation** of Hypothesis #4 over a real 756-day window — one that happened to contain an actual VIX spike to 52.3 — cross-validated the circuit-breaker finding through a completely independent mechanism: drawdown was contained to -8.5% (vs. -26% unprotected), but Sharpe collapsed 61.5% and win rate fell below breakeven. Two unrelated code paths, one backtest simulation and one forward walk, converged on the identical structural lesson.

**The chain in one view:**

```
H3 + H4 debates  ──►  Memory: timing_mismatch, selection_bias
                              │
                              ▼
H7 (decoupled exit, fixed VIX=18 breaker)
                              │
                  backtest: 69.5% trigger rate, Sharpe 1.14 → 0.31
                              │
                              ▼
Memory: circuit_breaker_overcalibration
                              │
                              ▼
H8 (percentile-based dynamic threshold, self-validating frequency claim)
                              │
                  debate: circular calibration, no out-of-sample evidence
                              │
                              ▼
Paper trading mandated, walk-forward validation required before live capital
```

Every arrow in that chain is a real database write, a real Claude API call, and a real (sometimes uncomfortable) empirical result — not a scripted demo.

---

## Architecture

The build follows a five-layer dependency chain, ordered by what each layer needs to exist before it's useful.

```
Layer 1 — Core Research Loop       Research Scientist, Backtester, Governance
Layer 2 — Research Intelligence    Debate Engine, Continuous Learning, Research Memory
Layer 3 — Knowledge                (planned — relational first, graph-backed later)
Layer 4 — Research Operations      Streamlit dashboard, governance workflow
Layer 5 — Portfolio Lab            (planned)
```

### Layer 1 — Core Research Loop

**Research Scientist** (`src/agents/research_scientist.py`)
Generates structured, testable hypotheses from market observations using Claude. Before generation, it retrieves relevant entries from Research Memory and is instructed — not just permitted — to adapt the hypothesis if a retrieved memory describes a directly applicable past failure. Every hypothesis is written with a full schema: signal components, macro conditions, expected holding period, and risk factors.

**Backtester** (`src/agents/backtester.py`)
Builds a composite signal from a hypothesis's components (momentum, earnings-revision proxy, volatility compression, institutional flow proxy) against real S&P 500 price/volume data via `yfinance`, then runs a non-overlapping, sequential-rebalance backtest — Sharpe, Sortino, Calmar, max drawdown, win rate, and an 80/20 out-of-sample split.

*A note on rigor*: an early version of this backtester computed forward returns on every trading day rather than at actual rebalance points, which silently compounded heavily overlapping windows and produced a -99.8% max drawdown on an otherwise-reasonable strategy. The fix (sampling returns only at non-overlapping `holding_days` intervals) is now standard, and the broken historical result was re-run and corrected rather than left in the database — a small but deliberate choice about what "governance" means in practice.

**Governance** (`src/models/governance.py`)
Every hypothesis gets a governance record the moment it's generated, tracking its full lifecycle: idea → statistical review → risk review → committee → paper trading → production → monitoring → retired. Every stage transition is logged with a timestamp and notes. Nothing skips a stage silently.

### Layer 2 — Research Intelligence

**Statistical Validator** (`src/agents/statistical_validator.py`)
Four independent checks before a hypothesis can reach committee: a one-sample t-test against zero return, a Sharpe-ratio significance test (Lo, 2002 asymptotic approximation), a sub-period stability check (the return series is split into three chunks; at least two must be individually positive), and a minimum sample-size floor (30 non-overlapping observations). Three of four checks must pass.

**Debate Engine** (`src/agents/debate_engine.py`)
Four sequential agents, each a separate Claude call with a distinct adversarial role:

- **Bull** — builds the strongest case for deployment using the backtest and statistical data
- **Bear** — reads the Bull's thesis and writes a targeted rebuttal aimed specifically at its weakest claims, not generic risk disclaimers
- **Risk** — independent portfolio-level assessment: concentration, tail risk, position limits
- **Judge** — synthesizes all three into a final decision. The system prompt enforces a hard constraint: the decision is only valid if it explicitly restates and engages with the Bear's strongest objection. A Judge that simply prefers the better Sharpe ratio without addressing the Bear's specific argument is, by construction, not following the protocol.

**Research Memory** (`src/models/research_memory.py`, `src/agents/memory_extractor.py`)
After a debate or backtest produces a clear failure mode, an extraction agent converts the full record into a structured memory: failure mode category, the conditions under which it occurred, a human-readable lesson, and a machine-readable structured constraint. These are retrieved (via signal-type and condition matching, with a small-corpus fallback) before every subsequent hypothesis generation and are not optional context — the Research Scientist's system prompt requires it to either adapt the hypothesis, add an explicit mitigation, or document why a given memory doesn't apply.

**Continuous Learning** (`src/agents/continuous_learning.py`)
Walks a deployed hypothesis forward through the most recent N trading days of real market data, simulating what paper trading would have shown, and compares realized performance against the original backtest. Flags decay (Sharpe falls below 50% of backtest) and recommends retire / improve / continue. Critically, it distinguishes *why* a strategy underperformed — a drawdown-protection mechanism that successfully reduces drawdown while taxing returns is flagged for recalibration, not retirement.

**Alpha Registry** (`src/models/alpha_registry.py`, `src/agents/alpha_registrar.py`)
The system's actual output: hypotheses that survived backtest, statistical validation, *and* committee approval get formally registered with a generated signal-construction summary, full performance metrics, and feature lineage. Registration explicitly checks the committee's decision field, not just the governance stage — a hypothesis the committee sent back for revision is excluded even if its stage hasn't formally regressed.

That last point exists because of a real bug this build caught in itself: an early version of the eligibility check trusted `current_stage` alone, and the `REQUEST_REVISION` branch of the debate engine recorded the decision in history without ever moving `current_stage` off its prior value. The result was that Hypothesis #3 — which the committee explicitly sent back for revision — initially passed the registry's eligibility check and was registered as a validated alpha. It was caught by cross-referencing the stage against the stored `committee_decision`, deregistered, and the eligibility logic was rewritten to treat the decision field as the source of truth. The fix is in the codebase; the bug isn't hidden.

### Layer 4 — Research Operations

**Dashboard** (`src/dashboard/app.py`)
A Streamlit interface with five views:
- **Research Feed** — every hypothesis with current governance stage and validated performance
- **Hypothesis Detail** — signal components, full governance timeline, and the complete bull/bear/risk/judge debate transcript for any hypothesis that's been through committee
- **Research Memory** — every extracted lesson with its structured constraint
- **Memory Lineage** — the causal chain itself, visually: which hypothesis produced which memory, and which later hypothesis applied it
- **Alpha Registry** — the system's validated, committee-approved output, with signal construction logic and full performance metrics

---

## Tech stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL + SQLAlchemy |
| LLM | Claude (Anthropic API) — Research Scientist, Debate Engine, Memory Extractor, Data Quality Debate |
| Market data | `yfinance` (price, volume, VIX) |
| Fundamental data | SEC EDGAR XBRL API (EPS history, quarterly filings) — 48/50 ticker coverage |
| Backtesting | `pandas` / `numpy`, non-overlapping rebalance simulation + circuit-breaker simulation |
| Statistics | `scipy.stats` — t-tests, asymptotic Sharpe significance |
| Interface | Streamlit (8-page research terminal) |
| Scenario analysis | Analytical impact estimation via sector/factor exposure + Claude synthesis |

---

## Project structure

```
aurum-v2/
├── src/
│   ├── core/
│   │   ├── database.py               # SQLAlchemy engine + session
│   │   └── config.py                 # environment configuration
│   ├── models/
│   │   ├── hypothesis.py             # hypothesis schema + lifecycle status enum
│   │   ├── governance.py             # stage tracking, debate record, committee decision
│   │   ├── research_memory.py        # structured failure taxonomy + lineage
│   │   ├── alpha_registry.py         # validated signal registry with live tracking
│   │   └── experiment_queue.py       # scheduling primitives
│   ├── data/
│   │   └── edgar_client.py           # SEC EDGAR XBRL client — real EPS history,
│   │                                 # quarterly deduplication, known-gap registry
│   ├── agents/
│   │   ├── research_scientist.py     # hypothesis generation with memory retrieval
│   │   ├── backtester.py             # composite signal + non-overlapping backtest
│   │   │                             # + circuit-breaker simulation + EDGAR wiring
│   │   ├── statistical_validator.py  # 4-gate significance testing
│   │   ├── debate_engine.py          # Bull/Bear/Risk/Judge + Knowledge Graph context
│   │   ├── memory_extractor.py       # debate to structured Research Memory
│   │   ├── alpha_registrar.py        # governance-integrity-checked registry admission
│   │   ├── data_quality_debate.py    # ad-hoc debate for empirical findings
│   │   ├── continuous_learning.py    # full-registry paper trading evaluation
│   │   ├── governance_actions.py     # stage-gated advancement with prerequisites
│   │   ├── research_copilot.py       # NL search over full research corpus
│   │   └── portfolio_lab.py          # scenario stress testing + AI explainer
│   └── dashboard/
│       └── app.py                    # 8-page Streamlit research terminal:
│                                     # Research Feed | Hypothesis Detail |
│                                     # Research Memory | Memory Lineage |
│                                     # Alpha Registry | Research Copilot |
│                                     # Knowledge Graph | Portfolio Lab
├── tests/                            # one test file per agent/module
├── migrations/                       # Alembic migrations
├── .env                              # DATABASE_URL + ANTHROPIC_API_KEY
├── requirements.txt
└── main.py                           # creates all tables via SQLAlchemy metadata
```

---

## Running it

```bash
# 1. Install dependencies
pip install anthropic psycopg2-binary sqlalchemy alembic pandas numpy \
    python-dotenv streamlit fastapi uvicorn scipy yfinance pydantic requests

# 2. Configure environment (.env at project root)
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/aurum_v2
ANTHROPIC_API_KEY=sk-ant-...

# 3. Create the database and all tables
createdb aurum_v2
python main.py

# 4. Generate a hypothesis (Research Scientist + Governance + Queue)
python -m tests.test_scientist

# 5. Run the full pipeline on a hypothesis (backtest → stats → debate)
python -m tests.test_validator
python -m tests.run_more_debates

# 6. Run continuous learning across the full registry (V2.7)
python -m tests.test_v27_full_registry

# 7. Test the Knowledge Graph
python -m tests.test_knowledge_store

# 8. Run Portfolio Lab stress tests
python -m tests.test_portfolio_lab

# 9. Launch the 8-page dashboard
streamlit run src/dashboard/app.py
```

---

## Build status

All five layers of the V2 roadmap are complete:

```
Layer 1 — Core Research Loop      ✅ Research Scientist, Backtester, Governance
Layer 2 — Research Intelligence   ✅ Debate Engine, Continuous Learning, Data Lake, Research Memory
Layer 3 — Knowledge Graph         ✅ Relational KnowledgeStore (Neo4j-swappable abstraction)
Layer 4 — Research Operations     ✅ Writable Governance Workflow, Research Copilot (NL search)
Layer 5 — Portfolio Lab           ✅ Scenario stress testing across all registered alphas
```

## Honest limitations

This is a research platform, not a production trading system:

- **Institutional flow is still a price-volume proxy.** Earnings revision now uses real SEC EDGAR XBRL data (48/50 ticker coverage, +0.94 OOS Sharpe improvement validated via matched-universe controlled test). Institutional flow replacing it with real dark-pool or 13F data is the next data gap.
- **The Knowledge Graph is relational, not a true graph database.** The `KnowledgeStore` abstraction is in place — migrating to Neo4j requires only implementing the same interface, not touching any consumers. The relational version covers 50 tickers with GICS classification, ETF constituencies, and macro sensitivity tags.
- **The Research Scheduler does not exist.** Relevant once hypothesis generation runs in batches rather than one at a time. With 8 hypotheses, there is no real contention to schedule.
- **The Alpha Registry has two members.** Hypothesis #3 was correctly excluded after the governance integrity bug was caught and fixed. No revised version has been generated — it sits at "needs revision," unresolved, which is realistic: not every research idea gets picked back up.
- **Portfolio Lab scenario impacts are analytical estimates, not mark-to-market P&L.** The system uses sector/factor exposure mappings and historical analog returns to estimate scenario impact — it does not have access to real-time positions or intraday data.

---

## Why this exists

This project is built around a specific, falsifiable claim: that an AI research system is more credible when it's designed to find its own flaws than when it's designed to look good. Every backtest number in this README that looks bad (the 69.5% circuit-breaker trigger rate, the 61.5% Sharpe degradation in paper trading) was kept, not edited out, because the discovery of those numbers — and the system's response to them — is the actual point.