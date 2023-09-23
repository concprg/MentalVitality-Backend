from sqlalchemy import Column
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Integer, String, ForeignKey, JSON, Float

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    age = Column(Integer)
    gender = Column(String)

class Screen_Tests(Base):
    __tablename__ = "screen_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    questions = Column(JSON)

class Test_Responses(Base):
    __tablename__ = "user_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    test_id = Column(Integer, ForeignKey("screen_tests.id"))
    responses = Column(JSON)