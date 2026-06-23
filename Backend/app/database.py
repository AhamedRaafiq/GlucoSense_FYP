from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from .config import settings

def create_database_if_not_exists():
    if settings.DATABASE_URL:
        # If DATABASE_URL is explicitly set, skip auto-creation as credentials parsing is not safe
        return
        
    try:
        # Connect to the default 'postgres' database to check/create the target database
        conn = psycopg2.connect(
            dbname="postgres",
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            host=settings.DB_HOST,
            port=settings.DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if the target database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{settings.DB_NAME}';")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{settings.DB_NAME}' does not exist. Creating it...")
            cursor.execute(f"CREATE DATABASE {settings.DB_NAME};")
            print(f"Database '{settings.DB_NAME}' created successfully.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not check/create database automatically. Details: {e}")

# Run database checks before connecting
create_database_if_not_exists()

# For psycopg2 connection
engine = create_engine(
    settings.get_database_url(),
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
