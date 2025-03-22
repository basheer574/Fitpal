from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime

# Initialize the database engine
DATABASE_URL = "sqlite:///fitpal.db"
engine = create_engine(DATABASE_URL, echo=True)

# Define the base class for models
Base = declarative_base()

# =================== 🔹 Define User Model 🔹 ===================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    height = Column(Integer)
    weight = Column(Integer)
    preferences = Column(String)  # Fitness preferences
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship with interactions
    interactions = relationship("Interaction", back_populates="user")

# =================== 🔹 Define Interaction Model 🔹 ===================
class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    input_text = Column(String, nullable=False)
    response_text = Column(String, nullable=False)
    pdf_url = Column(String, nullable=True)  # ✅ Stores the fitness plan PDF path
    category = Column(String, nullable=True)  # ✅ NEW: Stores fitness category
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")

# =================== 🔹 Initialize Database Function 🔹 ===================
def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully.")

# =================== 🔹 Create Session Factory 🔹 ===================
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()
