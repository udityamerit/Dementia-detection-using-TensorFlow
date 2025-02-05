from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Database Setup
def init_db():
    conn = sqlite3.connect("system.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT,
            reminder TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/kscar')
def kscar_assessment():
    return render_template("kscar.html")

@app.route('/timeline', methods=['GET', 'POST'])
def timeline():
    conn = sqlite3.connect("system.db")
    cursor = conn.cursor()
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        cursor.execute("INSERT INTO events (title, description) VALUES (?, ?)", (title, description))
        conn.commit()
    cursor.execute("SELECT * FROM events")
    events = cursor.fetchall()
    conn.close()
    return render_template("timeline.html", events=events)

@app.route('/reminders', methods=['GET', 'POST'])
def reminders():
    conn = sqlite3.connect("system.db")
    cursor = conn.cursor()
    if request.method == 'POST':
        task = request.form['task']
        reminder = request.form['reminder']
        cursor.execute("INSERT INTO tasks (task, reminder) VALUES (?, ?)", (task, reminder))
        conn.commit()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    conn.close()
    return render_template("reminders.html", tasks=tasks)

@app.route('/alerts', methods=['GET', 'POST'])
def caregiver_alert():
    conn = sqlite3.connect("system.db")
    cursor = conn.cursor()
    if request.method == 'POST':
        message = request.form['message']
        cursor.execute("INSERT INTO alerts (message) VALUES (?)", (message,))
        conn.commit()
    cursor.execute("SELECT * FROM alerts")
    alerts = cursor.fetchall()
    conn.close()
    return render_template("alerts.html", alerts=alerts)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
