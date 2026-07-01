import anthropic
import json
from src.core.config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

BULL_SYSTEM = """You are the Bull Agent evaluating a specific empirical finding for AURUM's
investment committee: a hypothesis's backtest improved across every metric when the price-volume
earnings proxy was replaced with real SEC EDGAR fundamental data. Your job is to make the
strongest case that this improvement reflects a genuine signal quality gain, not a statistical
artifact of the universe shrinking.

Return ONLY valid JSON, no markdown fences:
{
  "position": "genuine_improvement",
  "thesis_points": ["point 1", "point 2", "point 3"],
  "strongest_argument": "...",
  "confidence": <float 0-1>
}"""

BEAR_SYSTEM = """You are the Bear Agent evaluating the same finding. Your job is to find the
strongest case that this improvement is an artifact, NOT a genuine signal quality gain — most
likely because the universe shrank from 50 to 47 tickers (the 3 dropped tickers had real EDGAR
data gaps, not poor performance), and the improved Sharpe/drawdown could simply reflect removing
3 specific names rather than the earnings data ranking the remaining 47 any better.

Return ONLY valid JSON, no markdown fences:
{
  "position": "likely_artifact",
  "rebuttal_points": ["point 1", "point 2", "point 3"],
  "strongest_objection": "...",
  "confidence": <float 0-1>
}"""

JUDGE_SYSTEM = """You are the Judge. Read the Bull and Bear cases on whether a backtest
improvement from real EDGAR data vs. a price-volume proxy is genuine or an artifact of universe
shrinkage. Your decision must specify EXACTLY what additional test would resolve the ambiguity
— do not just pick a side without specifying the empirical test needed.

Return ONLY valid JSON, no markdown fences:
{
  "verdict": "genuine | artifact | inconclusive",
  "justification": "2-4 sentences engaging with the bear's strongest objection",
  "required_test": "the specific empirical test needed to resolve this (e.g. 're-run proxy backtest excluding the same 3 tickers')",
  "confidence": <float 0-1>
}"""

def _call(system, user, max_tokens=1200):
    resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=max_tokens,
                                    system=system, messages=[{"role": "user", "content": user}])
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

def debate_data_quality_finding(real_results: dict, proxy_results: dict,
                                  dropped_tickers: list, real_n: int, proxy_n: int) -> dict:
    context = f"""Hypothesis #4 backtest comparison:

REAL EDGAR DATA (n={real_n} tickers, dropped: {dropped_tickers}):
{json.dumps(real_results, indent=2)}

OLD PROXY (n={proxy_n} tickers, full universe):
{json.dumps(proxy_results, indent=2)}

The dropped tickers (BRK-B, GS, V) were excluded due to EDGAR data gaps, not poor performance —
BRK-B has non-standard XBRL tagging, GS had an EDGAR API failure, V had insufficient quarterly history.
"""
    bull = _call(BULL_SYSTEM, context)
    bear = _call(BEAR_SYSTEM, context + f"\n\nBull's case:\n{json.dumps(bull, indent=2)}")
    judge = _call(JUDGE_SYSTEM,
                  context + f"\n\nBull:\n{json.dumps(bull, indent=2)}\n\nBear:\n{json.dumps(bear, indent=2)}")

    return {"bull": bull, "bear": bear, "judge": judge}

def print_finding_debate(d: dict):
    print(f"\n{'='*65}\nDATA QUALITY FINDING — IS THE IMPROVEMENT REAL?\n{'='*65}")
    print(f"\nBULL — genuine improvement (confidence {d['bull']['confidence']}):")
    for p in d['bull']['thesis_points']:
        print(f"  • {p}")
    print(f"\n  Strongest: {d['bull']['strongest_argument']}")

    print(f"\nBEAR — likely artifact (confidence {d['bear']['confidence']}):")
    for p in d['bear']['rebuttal_points']:
        print(f"  • {p}")
    print(f"\n  Strongest: {d['bear']['strongest_objection']}")

    print(f"\nJUDGE VERDICT: {d['judge']['verdict'].upper()}")
    print(f"  {d['judge']['justification']}")
    print(f"\n  REQUIRED TEST: {d['judge']['required_test']}")
    print(f"  Confidence: {d['judge']['confidence']}")
    print(f"{'='*65}\n")