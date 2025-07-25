from flask import render_template, request, url_for
from flask import current_app
from . import bp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64

@bp.route('/', methods =['GET'])
def main():
    mongo = current_app.mongo['posts']
    unique_users = mongo.distinct('user_id')
    return render_template('dashboard/main.html', users=unique_users)

        
@bp.route('/user/<string:target>', methods =['GET'])
def profile(target):
    mongo = current_app.mongo['posts']
    

    target_user_id = target
    user_posts = list(
    mongo.find({'user_id': target_user_id}).sort('timestamp', 1)
    )

    for post in user_posts:
        post['sentiment_label'] = post['sentiment']['label']
        post['sentiment_score'] = post['sentiment']['score']

    df = pd.DataFrame(user_posts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    df['score'] = df['sentiment_label'].map(sentiment_map)

    df = df.sort_values("timestamp")
    df["cumulative_score"] = df["score"].cumsum()
    df["rolling_avg"] = df["score"].rolling(window=5, min_periods=1).mean()
    df["ema_score"] = df["score"].ewm(span=5, adjust=False).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["cumulative_score"], label="Cumulative Sentiment", color="blue", marker="o", alpha=0.5)
    plt.plot(df["timestamp"], df["rolling_avg"], label="Rolling Avg (5 days)", color="orange", linestyle="--", marker="x")
    plt.plot(df["timestamp"], df["ema_score"], label="EMA (span=5)", color="green", linestyle="-", marker="s")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Sentiment Trend for User {target_user_id}")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return render_template('dashboard/profile.html', plot_url=plot_data)

