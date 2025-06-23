from flask import render_template, request
from flask import current_app
from . import bp


@bp.route('/', methods =['GET'])
def index():
    return render_template('dashboard/dashboard.html')

