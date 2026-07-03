from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Entity:
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap_tier: str   # large | mid | small
    description: str = ""

@dataclass
class Relationship:
    source: str            # ticker
    target: str            # ticker or ETF or macro factor
    rel_type: str          # sector_peer | etf_constituent | supplier | customer | macro_sensitivity
    weight: float = 1.0   # strength of relationship 0-1
    notes: str = ""

class KnowledgeStore(ABC):
    """
    Abstract interface for the knowledge layer.
    All agents call this — none know or care whether the backend
    is a SQL table, SQLite file, or Neo4j graph.
    """

    @abstractmethod
    def get_entity(self, ticker: str) -> Entity | None:
        pass

    @abstractmethod
    def get_peers(self, ticker: str) -> list[Entity]:
        """Return sector peers of a ticker."""
        pass

    @abstractmethod
    def get_etf_exposure(self, ticker: str) -> list[dict]:
        """Return ETFs that hold this ticker with approximate weight."""
        pass

    @abstractmethod
    def get_sector_tickers(self, sector: str) -> list[str]:
        """Return all tickers in a given GICS sector."""
        pass

    @abstractmethod
    def get_macro_sensitivities(self, ticker: str) -> list[str]:
        """Return macro factors this ticker is sensitive to."""
        pass

    @abstractmethod
    def query(self, natural_language: str) -> str:
        """Answer a natural language question about relationships."""
        pass