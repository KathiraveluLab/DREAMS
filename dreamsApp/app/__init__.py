from flask import Flask
import os
from flask_login import LoginManager
from dreamsApp.core.config import PipelineConfig
from dreamsApp.core.pipeline import DreamsPipeline
from .models import User  
from bson.objectid import ObjectId 

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

    # Globally attach our core AI pipeline so we don't boot models per request
    app.dreams_pipeline = DreamsPipeline(config=PipelineConfig())

    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        """Checks if user is logged-in on every page load."""
        if user_id is not None:
            import sqlite3
            from dreamsApp.core.database import db_manager
            try:
                with sqlite3.connect(db_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    user_row = cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
                    if user_row:
                        return User(dict(user_row))
            except Exception as e:
                app.logger.error(f"Error loading user: {e}")
        return None

    from .auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from .ingestion.routes import bp as ingestion_bp
    app.register_blueprint(ingestion_bp)

    from .dashboard import bp as dashboard_bp
    app.register_blueprint(dashboard_bp)

    from .analytics import bp as analytics_bp
    app.register_blueprint(analytics_bp)

    return app