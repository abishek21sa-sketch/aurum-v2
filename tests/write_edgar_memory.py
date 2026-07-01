from src.core.database import SessionLocal
from src.models.research_memory import ResearchMemory
from src.models.hypothesis import Hypothesis
import uuid

def main():
    db = SessionLocal()
    hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()

    memory = ResearchMemory(
        id=uuid.uuid4(),
        source_hypothesis_id=hyp.id,
        source_hypothesis_number=4,
        failure_mode="other",
        conditions={"data_source": "proxy", "signal_type": "earnings_revision"},
        lesson=(
            "Price-volume divergence proxies for earnings revision introduce systematic noise "
            "that real SEC EDGAR fundamental data eliminates. Controlled matched-universe test "
            "(identical 47 tickers) confirmed real EDGAR data improved OOS Sharpe by +0.94 "
            "(1.39 vs 2.33) and reduced max drawdown by 37% (-24% vs -15%) vs the proxy — "
            "not from universe composition effects (falsified by the control), but from cleaner "
            "fundamental signal reducing false positives within the top quintile. Win rate "
            "improved only marginally (+2%), suggesting the improvement comes from better "
            "position magnitude ranking rather than better stock selection per se."
        ),
        structured_constraint={
            "applies_when": "A hypothesis uses a price-volume based proxy for any fundamental "
                           "signal (earnings revision, quality, sentiment)",
            "required_action": "Replace with real fundamental data (EDGAR XBRL for EPS/revenue, "
                               "FRED for macro, SEC filings for quality signals) before treating "
                               "backtest results as reliable. Always run a matched-universe "
                               "controlled test to distinguish signal quality from universe "
                               "composition effects when comparing data sources."
        },
        affected_signal_types=["earnings_revision", "fundamental_quality", "sentiment"],
        affected_features=["earnings_revision_proxy", "volume_divergence"],
        created_by="edgar_validation"
    )
    db.add(memory)
    db.commit()
    print(f"Memory written: {memory.id}")
    db.close()

if __name__ == "__main__":
    main()