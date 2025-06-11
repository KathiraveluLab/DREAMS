from flask import Flask
from pymongo import MongoClient
import os

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Default config
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flask.sqlite')
    )

    if test_config is None:
        from . import config  # import config.py
        app.config.from_object(config)  # load it
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



    from .ingestion.routes import bp as ingestion_bp
    app.register_blueprint(ingestion_bp)

    return app
