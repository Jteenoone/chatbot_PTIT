from flask import Flask, render_template, request, redirect, url_for, session
import os, json, shutil, uuid, datetime
from dotenv import load_dotenv

# --- Module chính ---
from rag_chatbot import RAGChatbot
from faq_service import FAQService
from rag_system import add_or_update_file, delete_knowledge, chatbot_reload_callback

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

# Cho phép reload chatbot sau khi cập nhật tri thức
def reload_chatbot():
    global rag_chatbot
    rag_chatbot = RAGChatbot()

chatbot_reload_callback = reload_chatbot


# ===========================================
# 🧾 LỊCH SỬ TRÒ CHUYỆN
# ===========================================

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
            return data if isinstance(data, list) else []
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
    sessions = load_sessions()
    if not sid:
        return None, sessions
    for s in sessions:
        if s.get("id") == sid:
            return s, sessions
    return None, sessions


def create_session(initial_name=None):
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


# ===========================================
# 💬 ROUTES CHATBOT
# ===========================================

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

    # Không lưu tin nhắn trống
    if not text:
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

    # Ưu tiên FAQ
    try:
        faq_reply = faq_service.check(text)
        if faq_reply:
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
        s["name"] = text[:40] + ("..." if len(text) > 40 else "")

    save_sessions(sessions)
    return redirect(url_for("index", session_id=s["id"]))


@app.route("/rename-session", methods=["POST"])
def rename_session():
    sid = request.form.get("session_id")
    new_name = (request.form.get("new_name") or "").strip()
    s, sessions = find_session(sid)
    if s and new_name:
        s["name"] = new_name
        save_sessions(sessions)
    return redirect(url_for("index", session_id=sid))


@app.route("/delete-session", methods=["POST"])
def delete_session():
    sid = request.form.get("session_id")
    sessions = [s for s in load_sessions() if s.get("id") != sid]
    save_sessions(sessions)
    return redirect(url_for("index"))


# ===========================================
# 🧠 ADMIN QUẢN LÝ TRI THỨC
# ===========================================

@app.route("/admin", methods=["GET"])
def admin_page():
    knowledge_files = []
    if os.path.exists(OLD_DOCS_DIR):
        knowledge_files = [
            f for f in os.listdir(OLD_DOCS_DIR)
            if os.path.isfile(os.path.join(OLD_DOCS_DIR, f))
        ]
    message = request.args.get("msg")
    error = request.args.get("err")
    return render_template("admin.html", auth=True, knowledge_files=knowledge_files, message=message, error=error)


# --- Upload file tri thức ---
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("admin_page", err="Chưa chọn file."))

    file_name = file.filename
    os.makedirs("./temp_uploads", exist_ok=True)
    temp_path = os.path.join("./temp_uploads", file_name)
    file.save(temp_path)

    # Gọi RAG system để kiểm tra trùng tên
    result = add_or_update_file(temp_path, force_replace=False)
    if result.get("exists"):
        return render_template("confirm_replace.html", file_name=file_name)

    msg = result.get("message", "Đã cập nhật tri thức.")
    return redirect(url_for("admin_page", msg=msg))


# --- Xác nhận ghi đè file ---
@app.route("/confirm-replace", methods=["POST"])
def confirm_replace():
    file_name = request.form.get("file_name")
    decision = request.form.get("decision")
    temp_path = os.path.join("./temp_uploads", file_name)

    if decision == "yes":
        result = add_or_update_file(temp_path, force_replace=True)
        msg = result.get("message", "Đã ghi đè tri thức.")
    else:
        msg = "❌ Đã hủy cập nhật."

    return redirect(url_for("admin_page", msg=msg))


# --- Xóa tri thức ---
@app.route("/delete-knowledge", methods=["POST"])
def delete_knowledge_file():
    file_name = request.form.get("file_name")
    success = delete_knowledge(file_name)
    msg = f"Đã xóa {file_name}" if success else f"Lỗi khi xóa {file_name}"
    return redirect(url_for("admin_page", msg=msg))


# --- Reset toàn bộ tri thức ---
@app.route("/reset-knowledge", methods=["POST"])
def reset_knowledge():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    reload_chatbot()
    return redirect(url_for("admin_page", msg="Đã reset lại cơ sở tri thức."))


# --- Rebuild FAQ ---
@app.route("/rebuild-faq", methods=["POST"])
def rebuild_faq():
    try:
        faq_service.rebuild()
        return redirect(url_for("admin_page", msg="Đã rebuild FAQ thành công."))
    except Exception as e:
        return redirect(url_for("admin_page", err=f"Lỗi rebuild FAQ: {e}"))


# ===========================================
# 🚀 CHẠY ỨNG DỤNG
# ===========================================
if __name__ == "__main__":
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    app.run(debug=True)
