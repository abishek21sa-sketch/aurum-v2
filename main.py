from alembic.config import Config
from alembic import command

def init_db():
    print("Running Alembic migrations...")
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    print("Database is at head revision.")

if __name__ == "__main__":
    init_db()