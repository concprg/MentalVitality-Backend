from app import db, models
from fastapi import Depends

def get_questions(test_id):
    questions = Depends(db.get_db()).query(models.Screen_Tests).filter(models.Screen_Tests.id == test_id).first()
    return questions

def put_question(test_id, questions_as_json):
    sti

def parse_responses(response_as_json):
    user_id = response_as_json["user_id"]
    test_id = response_as_json["test_id"]