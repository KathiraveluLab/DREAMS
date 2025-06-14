from flask import Blueprint

bp = Blueprint('ingestion', __name__, url_prefix='/api')

from ..ingestion import routes