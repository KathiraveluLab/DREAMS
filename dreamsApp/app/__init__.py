from flask import Flask
from pymongo import MongoClient
import os
from flask_login import LoginManager
from .models import User  
from bson.objectid import ObjectId 
#creating app
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Default config
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flask.sqlite')
    )

    if test_config is None:
        from . import config
        app.config.from_object(config)
    else:
        app.config.update(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    # MongoDB connection
    client = MongoClient(app.config["MONGO_URI"])
    app.mongo = client[app.config["MONGO_DB_NAME"]]

    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        """Checks if user is logged-in on every page load."""
        if user_id is not None:
            # Query the user by their MongoDB _id
            user_data = app.mongo.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        return None

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from .ingestion.routes import bp as ingestion_bp
    app.register_blueprint(ingestion_bp)

    from .dashboard import bp as dashboard_bp
    app.register_blueprint(dashboard_bp)

    return app