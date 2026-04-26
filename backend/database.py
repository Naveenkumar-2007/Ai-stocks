import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Use PostgreSQL if provided in .env, otherwise fallback to local SQLite for rapid development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_stocks.db")

# Setup SQLAlchemy engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    # Import all modules here that might define models so that
    # they will be registered properly on the metadata.
    import models
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized successfully at {DATABASE_URL}")
