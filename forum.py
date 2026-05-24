from flask import Blueprint, Flask, render_template_string, request, redirect, session, url_for
from flask.views import MethodView
import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from requests import get
from strands import Agent, tool
from strands.models import OllamaModel
from strands_tools import http_request, shell
import bbcode
import ollama
import urllib.parse

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev')
forum = Blueprint("forum", __name__)
bbcoded = bbcode.Parser()
DATABASE = 'forum.db'
PAGE_SIZE = 20
AGENT_MAX_WORKERS = 1
AGENT_SYSTEM_PROMPT = "You are AgentBB, an agentic forum bot. You use strands-agents and available tools to participate in discussions and address topics asked of you to address. Your primary goal is to ask clarifying questions until you are able to respond with useful information."
FORUM_FORMATTER_PROMPT = """Translate the input (an AI agent's markdown reply) as-is into a terser, more narrative-styled forum post (rather than document-styled) — change only the formatting, never the voice, the information, or point of view.
Use BBCode ([b], [i], [code], [url=...], [list][*]); no markdown, no headings, no tables, no unnecessary lists.
Output only the rewritten post."""

USERS = {
    "admin": "admin123",
    "alice": "alice123",
    "bob": "bob123"
}

_MAX_ID = 2**63 - 1

app.jinja_env.filters['bbcode'] = bbcoded.format
app.jinja_env.filters['usercolor'] = lambda u: hashlib.md5((u or '').encode()).hexdigest()[:6]


@tool
def searxng(raw: str) -> dict:
    query = urllib.parse.quote(raw)
    url = f"http://127.0.0.1:8080/search?format=json&q={query}"
    resp = get(url)
    return resp.json()


STYLES = """
:root {
  --bg: #d6dee5; --panel: #fff; --panel-alt: #efefef;
  --header-bg: #006699; --header-text: #fff;
  --row-alt: #ebebeb;
  --text: #000; --muted: #555; --border: #98aab1;
  --link: #006699; --link-visited: #5493b4; --link-hover: #d46400;
  --tag-agent: #408050; --tag-user: #006699;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1c2429; --panel: #242a30; --panel-alt: #2c333a;
    --header-bg: #18486b; --header-text: #e0e6ec;
    --row-alt: #2a323a;
    --text: #d2d6d8; --muted: #8c95a0; --border: #3d4750;
    --link: #6ba8e0; --link-visited: #8eb8df; --link-hover: #f29548;
    --tag-agent: #4e9c63; --tag-user: #5fa3d3;
  }
}
body { font: 12px Verdana, Arial, Helvetica, sans-serif; margin: 0; background: var(--bg); color: var(--text); line-height: 1.4; }
a { color: var(--link); text-decoration: underline; }
a:visited { color: var(--link-visited); }
a:hover { color: var(--link-hover); }
.wrap { max-width: 940px; margin: 0 auto; padding: 10px; }
.site-header { background: var(--header-bg); color: var(--header-text); padding: 8px 12px; margin-bottom: 10px; border: 1px solid var(--border); overflow: auto; }
.site-header h1 { margin: 0; font: bold 16px Georgia, "Times New Roman", serif; }
.site-header h1 a, .site-header h1 a:hover, .site-header h1 a:visited { color: var(--header-text); text-decoration: none; }
.user-info { float: right; font-size: 11px; padding-top: 4px; }
.user-info a, .user-info a:hover, .user-info a:visited { color: var(--header-text); }
.nav-links { margin: 0 0 8px; font-weight: bold; }
.panel { background: var(--panel); border: 1px solid var(--border); margin-bottom: 10px; }
.panel-header { background: var(--header-bg); color: var(--header-text); padding: 4px 8px; font-weight: bold; }
.thread-list { list-style: none; padding: 0; margin: 0; }
.thread-list li { padding: 5px 8px; border-bottom: 1px solid var(--border); }
.thread-list li:last-child { border-bottom: none; }
.thread-list li:nth-child(even) { background: var(--row-alt); }
.thread-list .empty { color: var(--muted); font-style: italic; }
.post { padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: pre-wrap; }
.post:nth-child(even) { background: var(--row-alt); }
.post:last-child { border-bottom: none; }
.post-meta { font-size: 10px; color: var(--muted); margin-bottom: 4px; padding-bottom: 4px; border-bottom: 1px dashed var(--border); }
.tag { display: inline-block; padding: 1px 5px; font-size: 10px; font-weight: bold; color: #fff; text-transform: uppercase; }
.tag-agent { background: var(--tag-agent); }
.tag-user { background: var(--tag-user); }
.pending { color: var(--muted); font-style: italic; }
.post-edit { float: right; font-size: 10px; }
.user-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: middle; margin: 0 3px 0 4px; }
textarea, input[type="text"], input[type="password"] { width: 100%; padding: 3px 4px; font: 11px Verdana, Arial, Helvetica, sans-serif; background: var(--panel); color: var(--text); border: 1px solid var(--border); box-sizing: border-box; }
.form-error { color: #c00; font-weight: bold; margin-bottom: 8px; }
textarea { min-height: 90px; resize: vertical; }
form.bodyform { padding: 10px; }
form label { display: block; font-weight: bold; margin: 0 0 3px; }
.form-row { margin-bottom: 8px; }
.form-buttons { margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap; }
button { font: bold 11px Verdana, Arial, Helvetica, sans-serif; background: var(--panel-alt); color: var(--text); border: 1px outset var(--border); padding: 3px 12px; cursor: pointer; }
button:hover { background: var(--row-alt); }
button:active { border-style: inset; }
.pagination { padding: 8px 0; text-align: center; }
"""

_PAGE_HEAD_TPL = ("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>""" + STYLES + """</style>
</head>
<body><div class="wrap">
<header class="site-header">
  <div class="user-info">
    {% if current_user %}Logged in as <b>{{ current_user }}</b> &middot; <a href="{{ url_for('forum.logout') }}">logout</a>{% else %}<a href="{{ url_for('forum.login') }}">login</a>{% endif %}
  </div>
  <h1><a href="{{ url_for('forum.index') }}">AI-Powered Forum</a></h1>
</header>
""")

_PAGE_FOOT = "</div></body></html>"

LOGIN_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "Log in") + """
<div class="panel">
  <div class="panel-header">Log in</div>
  <form class="bodyform" method="POST">
    {% if error %}<div class="form-error">{{ error }}</div>{% endif %}
    <div class="form-row">
      <label>Username</label>
      <input type="text" name="username" required autofocus>
    </div>
    <div class="form-row">
      <label>Password</label>
      <input type="password" name="password" required>
    </div>
    <input type="hidden" name="next" value="{{ next }}">
    <div class="form-buttons">
      <button type="submit">Log in</button>
    </div>
  </form>
</div>
""" + _PAGE_FOOT

INDEX_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "AI-Powered Forum") + """
<p class="nav-links"><a href="{{ url_for('forum.create') }}">New Thread</a></p>
<div class="panel">
  <div class="panel-header">Threads</div>
  <ul class="thread-list">
    {% for thread in threads %}
      <li><a href="{{ url_for('forum.thread', thread_id=thread.id) }}">{{ thread.title }}</a></li>
    {% else %}
      <li class="empty">No threads yet.</li>
    {% endfor %}
  </ul>
</div>
{% if has_prev or has_next %}
<div class="pagination">
  {% if has_prev %}<a href="{{ url_for('forum.index', page=page-1) }}">&laquo; Prev</a>{% endif %}
  {% if has_prev and has_next %} | {% endif %}
  {% if has_next %}<a href="{{ url_for('forum.index', page=page+1) }}">Next &raquo;</a>{% endif %}
</div>
{% endif %}
""" + _PAGE_FOOT

THREAD_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "{{ thread.title }}") + """
<p class="nav-links"><a href="{{ url_for('forum.index') }}">&laquo; Index</a></p>
<div class="panel">
  <div class="panel-header">{{ thread.title }}</div>
  <div class="post">
    <div class="post-meta">
      <span class="tag tag-user">OP</span>
      <span class="user-dot" style="background:#{{ thread.author | usercolor }}"></span><b>{{ thread.author or 'anon' }}</b>
      {% if is_admin or thread.author == current_user %}<a class="post-edit" href="{{ url_for('forum.edit_thread', thread_id=thread.id, page=page if page > 1 else None) }}">[edit]</a>{% endif %}
    </div>
    {{ thread.content | bbcode | safe }}
  </div>
  {% for post in posts %}
  <div class="post{% if post.pending %} pending{% endif %}" id="p-{{ post.id }}">
    <div class="post-meta">
      {% if post.is_agent %}<span class="tag tag-agent">Agent</span>{% else %}<span class="tag tag-user">User</span><span class="user-dot" style="background:#{{ post.author | usercolor }}"></span><b>{{ post.author or 'anon' }}</b>{% endif %}
      {% if not post.is_agent and not post.pending and (is_admin or post.author == current_user) %}<a class="post-edit" href="{{ url_for('forum.edit_post', thread_id=thread.id, post_id=post.id, page=page if page > 1 else None) }}">[edit]</a>{% endif %}
      <a href="#p-{{ post.id }}">#</a>
    </div>
    {{ post.content | bbcode | safe }}
  </div>
  {% endfor %}
</div>
{% if has_prev or has_next %}
<div class="pagination">
  {% if has_prev %}<a href="{{ url_for('forum.thread', thread_id=thread.id, page=page-1) }}">&laquo; Prev</a>{% endif %}
  {% if has_prev and has_next %} | {% endif %}
  {% if has_next %}<a href="{{ url_for('forum.thread', thread_id=thread.id, page=page+1) }}">Next &raquo;</a>{% endif %}
</div>
{% endif %}
<div class="panel">
  <div class="panel-header">Post a Reply</div>
  <form class="bodyform" method="POST">
    <div class="form-row">
      <textarea name="content" placeholder="Your message..." required></textarea>
    </div>
    <div class="form-buttons">
      <button type="submit">Post Reply</button>
      {% if is_admin %}<button type="submit" name="agent_reply" value="true">Get Agent Response</button>{% endif %}
    </div>
  </form>
</div>
""" + _PAGE_FOOT

EDIT_THREAD_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "Edit Thread") + """
<p class="nav-links"><a href="{{ url_for('forum.thread', thread_id=thread.id) }}">&laquo; Back to thread</a></p>
<div class="panel">
  <div class="panel-header">Edit Thread (changing the content discards all replies and triggers a new agent reply)</div>
  <form class="bodyform" method="POST">
    <div class="form-row">
      <label>Title</label>
      <input type="text" name="title" required value="{{ thread.title }}">
    </div>
    <div class="form-row">
      <label>Content</label>
      <textarea name="content" required>{{ thread.content }}</textarea>
    </div>
    <div class="form-buttons">
      <button type="submit">Save Edit</button>
    </div>
  </form>
</div>
""" + _PAGE_FOOT

EDIT_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "Edit Post") + """
<p class="nav-links"><a href="{{ url_for('forum.thread', thread_id=thread_id) }}">&laquo; Back to thread</a></p>
<div class="panel">
  <div class="panel-header">Edit Post (saving will discard all later posts and trigger a new agent reply)</div>
  <form class="bodyform" method="POST">
    <div class="form-row">
      <textarea name="content" required>{{ post.content }}</textarea>
    </div>
    <div class="form-buttons">
      <button type="submit">Save Edit</button>
    </div>
  </form>
</div>
""" + _PAGE_FOOT

CREATE_HTML = _PAGE_HEAD_TPL.replace("__TITLE__", "New Thread") + """
<p class="nav-links"><a href="{{ url_for('forum.index') }}">&laquo; Index</a></p>
<div class="panel">
  <div class="panel-header">New Thread</div>
  <form class="bodyform" method="POST">
    <div class="form-row">
      <label>Title</label>
      <input type="text" name="title" placeholder="Thread title..." required>
    </div>
    <div class="form-row">
      <label>Content</label>
      <textarea name="content" placeholder="What would you like to discuss?" required></textarea>
    </div>
    <div class="form-buttons">
      <button type="submit">Create Thread</button>
      {% if is_admin %}<button type="submit" name="agent_thread" value="true">Create Thread with Agent Response</button>{% endif %}
    </div>
  </form>
</div>
""" + _PAGE_FOOT


def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER,
                content TEXT NOT NULL,
                is_agent BOOLEAN DEFAULT 0,
                FOREIGN KEY(thread_id) REFERENCES threads(id)
            )
        ''')
        cols = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
        if 'is_agent' not in cols:
            conn.execute("ALTER TABLE posts ADD COLUMN is_agent BOOLEAN DEFAULT 0")
        if 'pending' not in cols:
            conn.execute("ALTER TABLE posts ADD COLUMN pending BOOLEAN DEFAULT 0")
        if 'author' not in cols:
            conn.execute("ALTER TABLE posts ADD COLUMN author TEXT")
        thread_cols = {row[1] for row in conn.execute("PRAGMA table_info(threads)")}
        if 'author' not in thread_cols:
            conn.execute("ALTER TABLE threads ADD COLUMN author TEXT")


class AgentBB:
    def __init__(
        self,
        db_path,
        *,
        max_workers=1,
        ollama_host="http://localhost:11434",
        main_model="gpt-oss:20b-cloud",
        formatter_model="gpt-oss:20b-cloud",
        system_prompt=AGENT_SYSTEM_PROMPT,
        formatter_prompt=FORUM_FORMATTER_PROMPT,
        tools=None,
    ):
        self.db_path = db_path
        self.ollama_host = ollama_host
        self.client = ollama.Client(host=ollama_host)
        self.main_model = main_model
        self.formatter_model = formatter_model
        self.system_prompt = system_prompt
        self.formatter_prompt = formatter_prompt
        self.tools = tools if tools is not None else [http_request, shell, searxng]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._recovery_done = False

    def thread_messages(self, thread_id, *, before_post_id=_MAX_ID):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            thread_row = conn.execute(
                'SELECT content, author FROM threads WHERE id = ?', (thread_id,)
            ).fetchone()
            post_rows = conn.execute(
                'SELECT content, is_agent, author FROM posts WHERE thread_id = ? AND id < ? ORDER BY id',
                (thread_id, before_post_id),
            ).fetchall()
        finally:
            conn.close()

        raw = []
        if thread_row:
            raw.append(("user", f"[{thread_row['author'] or 'anon'}] {thread_row['content']}"))
        for row in post_rows:
            if row["is_agent"]:
                raw.append(("assistant", row["content"]))
            else:
                raw.append(("user", f"[{row['author'] or 'anon'}] {row['content']}"))

        messages = []
        for role, text in raw:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"][0]["text"] += "\n\n" + text
            else:
                messages.append({"role": role, "content": [{"text": text}]})
        return messages

    def format_post(self, text):
        r = self.client.chat(
            model=self.formatter_model,
            think=False,
            messages=[
                {"role": "system", "content": self.formatter_prompt},
                {"role": "user", "content": text},
            ],
        )
        return r["message"]["content"]

    def process(self, prompt):
        try:
            model = OllamaModel(
                host=self.ollama_host,
                model_id=self.main_model,
                additional_args=dict(think="medium"),
            )
            agent = Agent(
                tools=self.tools,
                model=model,
                callback_handler=None,
                system_prompt=self.system_prompt,
            )
            response = agent(prompt)
            return self.format_post(str(response))
        except Exception as e:
            return f"Agent error: {e}"

    def dispatch(self, post_id, prompt):
        def task():
            response = self.process(prompt)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE posts SET content = ?, pending = 0 WHERE id = ?',
                    (response, post_id),
                )
                conn.commit()
        self.executor.submit(task)

    def recover_pending(self):
        if self._recovery_done:
            return
        self._recovery_done = True
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                'SELECT id, thread_id FROM posts WHERE pending = 1 ORDER BY id'
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            messages = self.thread_messages(row['thread_id'], before_post_id=row['id'])
            self.dispatch(row['id'], messages)


agentbb = AgentBB(DATABASE, max_workers=AGENT_MAX_WORKERS)


@forum.before_request
def _recover_pending_once():
    agentbb.recover_pending()

@forum.before_request
def authenticate():
    if request.endpoint in ("forum.login", "forum.logout", "forum.static"):
        return
    if not session.get("user"):
        return redirect(url_for("forum.login", next=request.path))


@app.context_processor
def _inject_auth():
    user = session.get("user")
    return dict(current_user=user, is_admin=user == "admin")



class LoginView(MethodView):
    def get(self):
        if request.endpoint == 'forum.logout':
            session.pop('user', None)
            return redirect(url_for('forum.index'))
        next_url = request.args.get('next', '/')
        if not (next_url.startswith('/') and not next_url.startswith('//')):
            next_url = '/'
        return render_template_string(LOGIN_HTML, next=next_url, error=None)

    def post(self):
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        next_url = request.form.get('next', '/')
        if not (next_url.startswith('/') and not next_url.startswith('//')):
            next_url = '/'
        if USERS.get(username) == password:
            session['user'] = username
            return redirect(next_url)
        return render_template_string(LOGIN_HTML, next=next_url, error='Invalid credentials')


class BoardView(MethodView):
    def get(self):
        if request.endpoint == 'forum.create':
            return render_template_string(CREATE_HTML)
        page = 1
        with suppress(TypeError, ValueError):
            page = int(request.args.get('page', 1))
        page = max(page, 1)
        offset = (page - 1) * PAGE_SIZE
        rows = get_db().execute(
            '''SELECT t.* FROM threads t
               LEFT JOIN posts p ON p.thread_id = t.id
               GROUP BY t.id
               ORDER BY MAX(p.id) DESC, t.id DESC
               LIMIT ? OFFSET ?''',
            (PAGE_SIZE + 1, offset),
        ).fetchall()
        return render_template_string(
            INDEX_HTML,
            threads=rows[:PAGE_SIZE],
            page=page,
            has_prev=page > 1,
            has_next=len(rows) > PAGE_SIZE,
        )

    def post(self):
        title = request.form['title']
        content = request.form['content']
        user = session['user']
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO threads (title, content, author) VALUES (?, ?, ?)', (title, content, user))
            thread_id = cursor.lastrowid
            pending_id = None
            if 'agent_thread' in request.form and user == 'admin':
                cursor.execute(
                    'INSERT INTO posts (thread_id, content, is_agent, pending, author) VALUES (?, ?, ?, ?, ?)',
                    (thread_id, "(Agent is thinking…)", 1, 1, None),
                )
                pending_id = cursor.lastrowid
            conn.commit()
        if pending_id is not None:
            agentbb.dispatch(pending_id, content)
        return redirect(url_for('forum.thread', thread_id=thread_id))


class ThreadView(MethodView):
    def get(self, thread_id):
        page = 1
        with suppress(TypeError, ValueError):
            page = int(request.args.get('page', 1))
        page = max(page, 1)
        offset = (page - 1) * PAGE_SIZE
        db = get_db()
        thread_data = db.execute('SELECT * FROM threads WHERE id = ?', (thread_id,)).fetchone()
        rows = db.execute(
            'SELECT * FROM posts WHERE thread_id = ? ORDER BY id LIMIT ? OFFSET ?',
            (thread_id, PAGE_SIZE + 1, offset),
        ).fetchall()
        return render_template_string(
            THREAD_HTML,
            thread=thread_data,
            posts=rows[:PAGE_SIZE],
            page=page,
            has_prev=page > 1,
            has_next=len(rows) > PAGE_SIZE,
        )

    def post(self, thread_id):
        content = request.form['content']
        user = session['user']
        pending_id = None
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                'INSERT INTO posts (thread_id, content, is_agent, author) VALUES (?, ?, ?, ?)',
                (thread_id, content, 0, user),
            )
            if 'agent_reply' in request.form and user == 'admin':
                cursor = conn.execute(
                    'INSERT INTO posts (thread_id, content, is_agent, pending, author) VALUES (?, ?, ?, ?, ?)',
                    (thread_id, "(Agent is thinking…)", 1, 1, None),
                )
                pending_id = cursor.lastrowid
            conn.commit()
        if pending_id is not None:
            messages = agentbb.thread_messages(thread_id, before_post_id=pending_id)
            agentbb.dispatch(pending_id, messages)
        return redirect(url_for('forum.thread', thread_id=thread_id, page=request.args.get('page') or None))


class EditView(MethodView):
    def get(self, thread_id, post_id=None):
        db = get_db()
        user = session['user']
        if post_id is None:
            thread = db.execute('SELECT * FROM threads WHERE id = ?', (thread_id,)).fetchone()
            if thread is None:
                return ('Thread not found', 404)
            if user != thread['author'] and user != 'admin':
                return ('Not allowed', 403)
            return render_template_string(EDIT_THREAD_HTML, thread=thread)
        post = db.execute(
            'SELECT * FROM posts WHERE id = ? AND thread_id = ?',
            (post_id, thread_id),
        ).fetchone()
        if post is None or post['is_agent'] or post['pending']:
            return ('Cannot edit this post', 404)
        if user != post['author'] and user != 'admin':
            return ('Not allowed', 403)
        return render_template_string(EDIT_HTML, thread_id=thread_id, post=post)

    def post(self, thread_id, post_id=None):
        db = get_db()
        user = session['user']
        if post_id is None:
            thread = db.execute('SELECT * FROM threads WHERE id = ?', (thread_id,)).fetchone()
            if thread is None:
                return ('Thread not found', 404)
            if user != thread['author'] and user != 'admin':
                return ('Not allowed', 403)
            new_title = request.form['title']
            new_content = request.form['content']
            content_changed = new_content != thread['content']
            pending_id = None
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    'UPDATE threads SET title = ?, content = ? WHERE id = ?',
                    (new_title, new_content, thread_id),
                )
                if content_changed:
                    conn.execute('DELETE FROM posts WHERE thread_id = ?', (thread_id,))
                    cursor = conn.execute(
                        'INSERT INTO posts (thread_id, content, is_agent, pending) VALUES (?, ?, ?, ?)',
                        (thread_id, "(Agent is thinking…)", 1, 1),
                    )
                    pending_id = cursor.lastrowid
                conn.commit()
            if pending_id is not None:
                messages = agentbb.thread_messages(thread_id, before_post_id=pending_id)
                agentbb.dispatch(pending_id, messages)
            return redirect(url_for('forum.thread', thread_id=thread_id, page=request.args.get('page') or None))

        post = db.execute(
            'SELECT * FROM posts WHERE id = ? AND thread_id = ?',
            (post_id, thread_id),
        ).fetchone()
        if post is None or post['is_agent'] or post['pending']:
            return ('Cannot edit this post', 404)
        if user != post['author'] and user != 'admin':
            return ('Not allowed', 403)
        new_content = request.form['content']
        with sqlite3.connect(DATABASE) as conn:
            conn.execute('UPDATE posts SET content = ? WHERE id = ?', (new_content, post_id))
            conn.execute('DELETE FROM posts WHERE thread_id = ? AND id > ?', (thread_id, post_id))
            cursor = conn.execute(
                'INSERT INTO posts (thread_id, content, is_agent, pending) VALUES (?, ?, ?, ?)',
                (thread_id, "(Agent is thinking…)", 1, 1),
            )
            pending_id = cursor.lastrowid
            conn.commit()
        messages = agentbb.thread_messages(thread_id, before_post_id=pending_id)
        agentbb.dispatch(pending_id, messages)
        return redirect(url_for('forum.thread', thread_id=thread_id, page=request.args.get('page') or None))


forum.add_url_rule("/login", view_func=LoginView.as_view("login"), methods=["GET", "POST"])
forum.add_url_rule("/logout", view_func=LoginView.as_view("logout"), methods=["GET"])
forum.add_url_rule("/", view_func=BoardView.as_view("index"), methods=["GET"])
forum.add_url_rule("/create", view_func=BoardView.as_view("create"), methods=["GET", "POST"])
forum.add_url_rule("/thread/<int:thread_id>", view_func=ThreadView.as_view("thread"), methods=["GET", "POST"])
forum.add_url_rule(
    "/thread/<int:thread_id>/edit",
    view_func=EditView.as_view("edit_thread"),
    defaults={"post_id": None},
    methods=["GET", "POST"],
)
forum.add_url_rule(
    "/thread/<int:thread_id>/post/<int:post_id>/edit",
    view_func=EditView.as_view("edit_post"),
    methods=["GET", "POST"],
)

app.register_blueprint(forum)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
