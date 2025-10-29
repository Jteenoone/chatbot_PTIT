from flask import Flask, render_template, request, redirect, url_for, session
import os, json, shutil, uuid, datetime
from dotenv import load_dotenv

# --- Module chính ---
from rag_chatbot import RAGChatbot
from faq_service import FAQService
from rag_system import initialize_vector_store, update_knowledge_base_auto

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "ptit_secret_123")

# --- Đường dẫn chính ---
CHAT_HISTORY_FILE = "chat_history.json"
OLD_DOCS_DIR = "./old_docs"
CHROMA_DB_PATH = "./knowledge_base_ptit"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Khởi tạo chatbot ---
rag_chatbot = RAGChatbot()
faq_service = FAQService()

# --- HÀM TIỆN ÍCH ---
def load_sessions():
    """Đọc file chat_history.json an toàn."""
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_sessions(sessions):
    """Lưu danh sách session, loại bỏ các session rỗng."""
    clean = [s for s in sessions if s.get("messages")]
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Lỗi khi lưu sessions:", e)

def find_session(sid):
    """Tìm session theo id."""
    sessions = load_sessions()
    if not sid:
        return None, sessions
    for s in sessions:
        if s.get("id") == sid:
            return s, sessions
    return None, sessions

def create_session(initial_name=None):
    """Tạo session mới."""
    s = {
        "id": str(uuid.uuid4()),
        "name": initial_name or "Cuộc trò chuyện mới",
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    sessions = load_sessions()
    sessions.append(s)
    save_sessions(sessions)
    return s, sessions

# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    sid = request.args.get("session_id")
    sessions = load_sessions()
    current = None
    if sid:
        current, _ = find_session(sid)
    return render_template("index.html", sessions=sessions, current=current)

@app.route("/send", methods=["POST"])
def send():
    text = (request.form.get("message") or "").strip()
    sid = request.form.get("session_id")
    if not text:
        # Không lưu tin nhắn rỗng
        return redirect(url_for("index", session_id=sid) if sid else url_for("index"))

    s, sessions = find_session(sid)
    if not s:
        s, sessions = create_session(text[:40])

    s.setdefault("messages", [])
    s["messages"].append({
        "role": "user",
        "text": text,
        "ts": datetime.datetime.now().isoformat()
    })

    # ✅ Ưu tiên FAQ trước
    try:
        faq_reply = faq_service.check(text)
        if faq_reply:
            # reply = f"📘 Trả lời từ câu hỏi thường gặp:\n{faq_reply}"
            reply = faq_reply
        else:
            reply = rag_chatbot.get_answer(text)
    except Exception as e:
        reply = f"Lỗi khi xử lý: {e}"

    s["messages"].append({
        "role": "bot",
        "text": reply,
        "ts": datetime.datetime.now().isoformat()
    })

    if s.get("name") == "Cuộc trò chuyện mới":
        s["name"] = (text[:40] + ("..." if len(text) > 40 else ""))

    save_sessions(sessions)
    return redirect(url_for("index", session_id=s["id"]))

@app.route("/new-session", methods=["POST"])
def new_session():
    return redirect(url_for("index"))

@app.route("/rename-session", methods=["POST"])
def rename_session():
    sid = request.form.get("session_id")
    new_name = (request.form.get("new_name") or "").strip()
    if not new_name:
        return redirect(url_for("index", session_id=sid))
    s, sessions = find_session(sid)
    if s:
        s["name"] = new_name
        save_sessions(sessions)
    return redirect(url_for("index", session_id=sid))

@app.route("/delete-session", methods=["POST"])
def delete_session():
    sid = request.form.get("session_id")
    sessions = load_sessions()
    sessions = [s for s in sessions if s.get("id") != sid]
    save_sessions(sessions)
    return redirect(url_for("index"))

# --- ADMIN ---
@app.route("/admin", methods=["GET"])
def admin_page():
    if not session.get("is_admin"):
        return render_template("admin.html", auth=False)
    knowledge_files = []
    if os.path.exists(OLD_DOCS_DIR):
        knowledge_files = [
            f for f in os.listdir(OLD_DOCS_DIR)
            if os.path.isfile(os.path.join(OLD_DOCS_DIR, f))
        ]
    return render_template("admin.html", auth=True, knowledge_files=knowledge_files)

@app.route("/admin-login", methods=["POST"])
def admin_login():
    pw = request.form.get("password")
    if pw and pw == os.getenv("ADMIN_PASSWORD"):
        session["is_admin"] = True
        session["admin_pw"] = pw
        return redirect(url_for("admin_page"))
    return render_template("admin.html", auth=False, error="Sai mật khẩu.")

@app.route("/admin-logout", methods=["POST"])
def admin_logout():
    session.pop("is_admin", None)
    session.pop("admin_pw", None)
    return redirect(url_for("admin_page"))

@app.route("/upload", methods=["POST"])
def upload_file():
    if not session.get("is_admin"):
        return redirect(url_for("admin_page"))
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("admin.html", auth=True, error="Chưa chọn file.")
    os.makedirs("new_docs", exist_ok=True)
    path = os.path.join("new_docs", file.filename)
    file.save(path)
    update_knowledge_base_auto()
    global rag_chatbot
    rag_chatbot = RAGChatbot()
    return redirect(url_for("admin_page"))

@app.route("/reset-knowledge", methods=["POST"])
def reset_knowledge():
    if not session.get("is_admin"):
        return redirect(url_for("admin_page"))
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    if os.path.exists(OLD_DOCS_DIR) and os.listdir(OLD_DOCS_DIR):
        initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
    else:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    global rag_chatbot
    rag_chatbot = RAGChatbot()
    return redirect(url_for("admin_page"))

@app.route("/rebuild-faq", methods=["POST"])
def rebuild_faq():
    if not session.get("is_admin"):
        return redirect(url_for("admin_page"))
    try:
        faq_service.rebuild()
        return render_template("admin.html", auth=True, message="Đã rebuild FAQ thành công!")
    except Exception as e:
        return render_template("admin.html", auth=True, error=f"Lỗi rebuild FAQ: {e}")

if __name__ == "__main__":
    app.run(debug=True)
