from sqlalchemy import Column, Integer, String, Float, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String)
    phone = Column(String)
    skills = Column(JSON)             # list of skills
    experience_years = Column(Float)
    education = Column(JSON)          # list of qualifications
    ats_score = Column(Float, nullable=True)
    resume_raw_text = Column(String, nullable=True)


# SQLite for quick local dev - swap to PostgreSQL URL for production
DATABASE_URL = "sqlite:///./careerlens.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
