from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required
from .models import User
from werkzeug.security import generate_password_hash

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        mongo = current_app.mongo

        if not username or not email or not password:
            flash('Please fill in all fields.')
            return redirect(url_for('auth.register'))

        if mongo.users.find_one({"username": username}):
            flash('Please use a different username.')
            return redirect(url_for('auth.register'))

        if mongo.users.find_one({"email": email}):
            flash('Please use a different email address.')
            return redirect(url_for('auth.register'))

        
        mongo.users.insert_one({
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password)
        })

        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('auth.login'))
        
    return render_template('register.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        

        mongo = current_app.mongo
        user_data = mongo.users.find_one({"username": username})

        if user_data:
            user = User(user_data)
            
            is_password_correct = user.check_password(password)
            

            if is_password_correct:
                login_user(user, remember=True)
                return redirect(url_for('dashboard.main'))

        flash('Invalid username or password')
        return redirect(url_for('auth.login'))

    return render_template('login.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))