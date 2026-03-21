from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required
from .models import User
from werkzeug.security import generate_password_hash
import sqlite3
from dreamsApp.core.database import db_manager

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('Please fill in all fields.')
            return redirect(url_for('auth.register'))

        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()

            if cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
                flash('Please use a different username.')
                return redirect(url_for('auth.register'))

            if cursor.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                flash('Please use a different email address.')
                return redirect(url_for('auth.register'))

            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password))
            )
            conn.commit()

        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('auth.login'))
        
    return render_template('register.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            user_row = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user_row:
            user = User(dict(user_row))
            
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