from flask import Blueprint

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

from . import main
from . import analytics_routes
