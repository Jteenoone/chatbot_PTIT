from flask import Flask, render_template, request, jsonify
import json, os, shutil
from dotenv import load_dotenv

# Import RAGChatbot (đã tích hợp FAQ)
from rag_chatbot import RAGChatbot

# Import hàm build/update KB
from rag_system import initialize_vector_store, update_knowledge_base_auto

app = Flask(__name__)
load_dotenv()

# Khởi tạo chatbot (đã có FAQ)
rag_chatbot = RAGChatbot()

# ====== Lịch sử chat ======
CHAT_HISTORY_FILE = "chat_history.json"
OLD_DOCS_DIR = "./old_docs"
CHROMA_DB_PATH = "./knowledge_base_ptit"
EMBEDDING_MODEL = "text-embedding-3-small"

if not os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def load_history():
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_message(role, text):
    history = load_history()
    history.append({"role": role, "text": text})
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ====== ROUTES ======

@app.route("/")
def index():
    return render_template("index.html", history=load_history())


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Tin nhắn trống."}), 400

    save_message("user", user_message)

    try:
        # ✅ Chatbot có FAQ + RAG
        bot_reply = rag_chatbot.get_answer(user_message)
    except Exception as e:
        bot_reply = f"Lỗi khi xử lý: {str(e)}"

    save_message("bot", bot_reply)
    return jsonify({"reply": bot_reply})


@app.route("/admin")
def admin_page():
    knowledge_files = []
    if os.path.exists(OLD_DOCS_DIR):
        knowledge_files = [f for f in os.listdir(OLD_DOCS_DIR) if os.path.isfile(os.path.join(OLD_DOCS_DIR, f))]
    return render_template("admin.html", knowledge_files=knowledge_files)


@app.route("/upload", methods=["POST"])
def upload():
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.form.get("password")

    if not admin_password or submitted_password != admin_password:
        return jsonify({"error": "Mật khẩu không đúng hoặc chưa được thiết lập."}), 403

    if "file" not in request.files:
        return jsonify({"error": "Không có file được gửi."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ."}), 400

    try:
        os.makedirs("new_docs", exist_ok=True)
        path = os.path.join("new_docs", file.filename)
        file.save(path)

        result = update_knowledge_base_auto();
        if result.get("success"):
            global rag_chatbot
            rag_chatbot = RAGChatbot()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý file: {str(e)}"}), 500


@app.route("/check-admin-password", methods=["POST"])
def check_admin_password():
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.json.get("password")
    if admin_password and submitted_password == admin_password:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 401


@app.route("/reset-knowledge", methods=["POST"])
def reset_knowledge_base():
    admin_password = os.getenv("ADMIN_PASSWORD")
    submitted_password = request.form.get("password")

    if not admin_password or submitted_password != admin_password:
        return jsonify({"error": "Mật khẩu không đúng."}), 403

    global rag_chatbot

    try:
        rag_chatbot = None  # giải phóng để tránh file lock

        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        if os.path.exists(OLD_DOCS_DIR) and os.listdir(OLD_DOCS_DIR):
            initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
        else:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

        rag_chatbot = RAGChatbot()
        return jsonify({"success": True, "message": "Đã reset và tự động nạp lại tri thức thành công."})

    except Exception as e:
        if rag_chatbot is None:
            rag_chatbot = RAGChatbot()
        return jsonify({"error": f"Lỗi khi reset: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
