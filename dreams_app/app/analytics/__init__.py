from flask import Blueprint

bp = Blueprint('analytics_api', __name__, url_prefix='/api/analytics')

from . import routes
