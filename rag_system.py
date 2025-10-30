import os
import json
import shutil
import threading
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ==============================
# ⚙️ Cấu hình
# ==============================
load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
OLD_DOCS_DIR = "./old_docs"
UPDATE_LOG_FILE = "./update_log.json"

update_lock = threading.Lock()
_vector_cache = None
chatbot_reload_callback = None  # Flask sẽ gán callback reload chatbot


# ==============================
# 🔹 Xử lý & nhúng tài liệu
# ==============================
def process_file(file_path):
    """Đọc & chia nhỏ nội dung file."""
    try:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        for c in chunks:
            c.metadata["file_name"] = os.path.basename(file_path)
        return chunks
    except Exception as e:
        print(f"[❌] Lỗi khi xử lý {file_path}: {e}")
        return []


# ==============================
# 🔹 Khởi tạo hoặc tải vector store
# ==============================
def get_vector_store():
    """Tạo hoặc load vector store cache."""
    global _vector_cache
    if _vector_cache is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vector_cache = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return _vector_cache


# ==============================
# 🔹 Ghi log cập nhật
# ==============================
def log_update(file_name, status, chunks, duration):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": file_name,
        "status": status,
        "chunks": chunks,
        "duration_sec": round(duration, 2)
    }
    with open(UPDATE_LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")


def get_update_logs(limit=10):
    if not os.path.exists(UPDATE_LOG_FILE):
        return []
    with open(UPDATE_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]


# ==============================
# 🔹 Xóa tri thức theo file
# ==============================
def delete_knowledge(file_name):
    """Xóa file & vector tương ứng"""
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        store.delete(where={"file_name": file_name})

        old_path = os.path.join(OLD_DOCS_DIR, file_name)
        if os.path.exists(old_path):
            os.remove(old_path)

        print(f"[🗑️] Đã xóa tri thức: {file_name}")
        return True
    except Exception as e:
        print(f"[❌] Lỗi khi xóa {file_name}: {e}")
        return False


# ==============================
# 🔹 Thêm hoặc cập nhật tri thức
# ==============================
def add_or_update_file(file_path, force_replace=False):
    """
    Nếu file đã tồn tại:
        - Nếu force_replace=True => xóa file + vector cũ, rồi thêm mới.
        - Nếu False => trả về cảnh báo.
    """
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    file_name = os.path.basename(file_path)
    old_path = os.path.join(OLD_DOCS_DIR, file_name)

    # Nếu file tồn tại mà chưa chọn ghi đè
    if os.path.exists(old_path) and not force_replace:
        return {"exists": True, "file": file_name}

    start = time.time()
    try:
        store = get_vector_store()
        chunks = process_file(file_path)

        if not chunks:
            return {"success": False, "message": f"Không thể xử lý {file_name}"}

        # Nếu ghi đè: xóa vector + file cũ
        if os.path.exists(old_path):
            delete_knowledge(file_name)

        # Thêm tài liệu mới
        store.add_documents(chunks)
        shutil.copy(file_path, old_path)

        duration = time.time() - start
        log_update(file_name, "success", len(chunks), duration)

        # Reload chatbot sau khi cập nhật
        if chatbot_reload_callback:
            chatbot_reload_callback()

        return {
            "success": True,
            "message": f"✅ Đã thêm/cập nhật {file_name} ({len(chunks)} đoạn)."
        }
    except Exception as e:
        log_update(file_name, "error", 0, 0)
        return {"success": False, "message": str(e)}


# ==============================
# 🔹 Tự động cập nhật nền (nếu cần)
# ==============================

def start_auto_update(interval=7200):
    """Cập nhật tự động mỗi interval giây (nếu muốn)."""
    def loop():
        while True:
            try:
                print("[⏳] Kiểm tra cập nhật tri thức định kỳ...")
                time.sleep(interval)
            except KeyboardInterrupt:
                break
    threading.Thread(target=loop, daemon=True).start()


# ==============================
# 🔹 Khi chạy trực tiếp
# ==============================
if __name__ == "__main__":
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    print("[💡] Hệ thống tri thức PTIT sẵn sàng.")
