import os
import glob
import json
import shutil
import threading
import time
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ===============================================
# Cấu hình
# ===============================================

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
OLD_DOCS_DIR = "./old_docs"
NEW_DOCS_DIR = "./new_docs"
UPDATE_LOG_FILE = "./update_log.json"

update_lock = threading.Lock()
_vector_cache = None
chatbot_reload_callback = None   # Flask sẽ gán callback reload vào đây


# ===============================================
# 1. Load & xử lý tài liệu
# ===============================================

def load_and_process_documents(docs_dir: str):
    try:
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*",
            loader_cls=UnstructuredFileLoader,
            show_progress=False,
            use_multithreading=True
        )
        documents = loader.load()
        if not documents:
            return [], []

        for doc in documents:
            if "source" in doc.metadata:
                doc.metadata["file_name"] = os.path.basename(doc.metadata["source"])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        source_files = list(set([doc.metadata.get("source") for doc in documents]))
        return chunks, source_files
    except Exception as e:
        return [], []


# ===============================================
# 2. Tạo hoặc load Vector Store
# ===============================================

def initialize_vector_store(db_path: str, embedding_model: str, docs_dir: str):
    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        if os.path.exists(db_path):
            return Chroma(persist_directory=db_path, embedding_function=embeddings)

        chunks, _ = load_and_process_documents(docs_dir)
        if not chunks:
            return Chroma(embedding_function=embeddings, persist_directory=db_path)

        return Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_path)
    except Exception as e:
        print(f"[Init Error] {e}")
        return None


# ===============================================
# 3️. Ghi log cập nhật
# ===============================================

def save_update_log(files_count, chunks_count, status="success", duration=0):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_files": files_count,
        "chunks_added": chunks_count,
        "duration_sec": round(duration, 2),
        "status": status
    }
    with open(UPDATE_LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(log_entry, f, ensure_ascii=False)
        f.write("\n")


def get_update_logs(limit=10):
    if not os.path.exists(UPDATE_LOG_FILE):
        return []
    with open(UPDATE_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
        return [json.loads(line.strip()) for line in lines if line.strip()]


# ===============================================
# 4️. Kiểm tra & cập nhật tri thức
# ===============================================

def check_and_update_database(vector_store: Chroma, new_docs_dir: str, old_docs_dir: str):
    os.makedirs(new_docs_dir, exist_ok=True)
    os.makedirs(old_docs_dir, exist_ok=True)

    if not os.listdir(new_docs_dir):
        return {"updated_files": 0, "chunks_added": 0}

    start = time.time()
    new_chunks, processed_files = load_and_process_documents(new_docs_dir)
    if not new_chunks:
        return {"updated_files": 0, "chunks_added": 0}

    # Ghi lại file đã thêm
    for file_path in processed_files:
        dest = os.path.join(old_docs_dir, os.path.basename(file_path))
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(file_path, dest)

    vector_store.add_documents(new_chunks)
    save_update_log(len(processed_files), len(new_chunks), "success", time.time() - start)
    return {"updated_files": len(processed_files), "chunks_added": len(new_chunks)}


# ===============================================
# 5️. Tự động cập nhật tri thức
# ===============================================
def update_knowledge_base_auto():
    if update_lock.locked():
        return {"success": False, "message": "Đang có quá trình cập nhật khác."}

    with update_lock:
        try:
            os.makedirs(OLD_DOCS_DIR, exist_ok=True)
            os.makedirs(NEW_DOCS_DIR, exist_ok=True)

            db = get_vector_store()
            result = check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)

            # Reload chatbot sau update
            if chatbot_reload_callback:
                chatbot_reload_callback()
            return {
                "success": True,
                "message": f"Đã cập nhật {result['updated_files']} file.",
                "stats": result
            }
        except Exception as e:
            save_update_log(0, 0, "error")
            return {"success": False, "message": str(e)}


# ===============================================
# 6️. Auto-update nền (background thread)
# ===============================================

def start_auto_update(interval=3600):
    def loop():
        while True:
            update_knowledge_base_auto()
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


# ===============================================
# 7️. Vector store caching
# ===============================================

def get_vector_store():
    global _vector_cache
    if _vector_cache is None:
        _vector_cache = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)
    return _vector_cache


# ===============================================
# Khi chạy trực tiếp
# ===============================================
if __name__ == "__main__":
    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    os.makedirs(NEW_DOCS_DIR, exist_ok=True)
    db = get_vector_store()
    info = check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)
    print("Update done:", info)
