import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    # 用户表增加最后活动时间
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 操作日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 检查是否已存在管理员，若无则创建默认管理员 admin/admin123
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_pw = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       ('admin', hashed_pw, 'admin'))
    
    conn.commit()
    conn.close()

def update_last_active(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def log_action(username, action, details=""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO audit_logs (username, action, details) VALUES (?, ?, ?)", 
                   (username, action, details))
    conn.commit()
    conn.close()

def get_online_users(minutes=5):
    conn = get_connection()
    cursor = conn.cursor()
    # SQLite datetime('now') 是 UTC 时间，确保一致性
    cursor.execute(f"SELECT username, role, last_active FROM users WHERE last_active > datetime('now', '-{minutes} minutes')")
    users = cursor.fetchall()
    conn.close()
    return users

def get_audit_logs(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, action, details, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    logs = cursor.fetchall()
    conn.close()
    return logs

def verify_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password = ?", (username, hashed_pw))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def add_user(username, password, role='user'):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       (username, hashed_pw, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def delete_user(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    # 防止删除最后一个管理员
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if user and user[0] == 'admin':
        return False
    
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True

# 初始化数据库
init_db()
