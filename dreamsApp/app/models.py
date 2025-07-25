# app/models.py

from flask_login import UserMixin
from werkzeug.security import check_password_hash
from flask_login import LoginManager

class User(UserMixin):
    """
    User model that works with MongoDB documents.
    It's not a database model in the SQL sense, but a helper
    class to work with user data from MongoDB.
    """
    def __init__(self, user_data):
        self.id = str(user_data.get('_id'))
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password_hash')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
