from src.agents.backtester import fetch_price_data, build_composite_signal, SP500_UNIVERSE
from src.core.database import SessionLocal
from src.models.hypothesis import Hypothesis

def main():
    db = SessionLocal()
    hyp = db.query(Hypothesis).filter_by(hypothesis_number=4).first()

    close, volume = fetch_price_data(SP500_UNIVERSE, "2018-01-01", "2026-06-30")
    print(f"close shape: {close.shape}\n")

    print("Calling build_composite_signal directly (real fix path)...\n")
    composite = build_composite_signal(close, volume, hyp.signal_components, use_real_earnings_data=True)

    print(f"\nFinal composite signal type: {type(composite)}")
    if composite is not None:
        print(f"Final composite signal length: {len(composite)}")
        print(composite.head(10))

    db.close()

if __name__ == "__main__":
    main()