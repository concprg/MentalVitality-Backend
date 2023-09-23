from app.models import User


def generate_context(user: User):
    context = f"""
    Patient Profile:
    Username: {user.username}
    Email: {user.email}
    Age: {user.age}
    Name: {user.gender}
    """
    return context
