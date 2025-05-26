from sqlmodel import text, create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

db_engine = create_engine(DB_URL)

with db_engine.connect() as conn:
    conn.scalar(text("SELECT 1"))
