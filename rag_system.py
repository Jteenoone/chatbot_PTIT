import glob
import os
import shutil
import threading

from certifi import where
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    print("Lỗi: OPENAI_API_KEY chưa được thiết lập. Vui lòng tạo file .env và thêm key vào.")
    exit()

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
OLD_DOCS_DIR = "./old_docs"
NEW_DOCS_DIR = "./new_docs"


def load_and_process_documents(docs_dir: str):
    """Load và xử lý tài liệu từ thư mục"""
    try:
        loader = DirectoryLoader(
            docs_dir,
            glob="**/*",
            loader_cls=UnstructuredFileLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        if not documents:
            return [], []

        for doc in documents:
            if "source" in doc.metadata:
                doc.metadata["file_name"] = os.path.basename(doc.metadata["source"])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        source_files = list(set([doc.metadata.get("source") for doc in documents]))
        return chunks, source_files
    except Exception as e:
        print(f"Lỗi khi load tài liệu: {e}")
        return [], []


def initialize_vector_store(db_path: str, embedding_model: str, docs_dir: str):
    """Khởi tạo hoặc load Vector Store"""
    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)

        if os.path.exists(db_path):
            # print(f"Đang load Knowledge Base từ '{db_path}'...")
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
        else:
            # print(f"Knowledge Base chưa tồn tại. Đang tạo mới từ '{docs_dir}'...")
            chunks, _ = load_and_process_documents(docs_dir)

            if not chunks:
                # print("Không có tài liệu ban đầu, tạo một Knowledge Base rỗng.")
                vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=db_path
                )
            else:
                # print(f"Đang embedding {len(chunks)} chunks...")
                vector_store = Chroma.from_documents(
                    chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )

            # vector_store.persist()
            # print(f"Đã tạo và lưu Knowledge Base vào '{db_path}'.")

        return vector_store
    except Exception as e:
        print(f"Lỗi khi khởi tạo Vector Store: {e}")
        exit(1)


def check_and_update_database(vector_store: Chroma, new_docs_dir: str, old_docs_dir: str):
    """Kiểm tra và cập nhật database với tài liệu mới"""
    try:
        if not os.path.exists(new_docs_dir) or not os.listdir(new_docs_dir):
            print("Thư mục 'new_docs' rỗng, không có gì để cập nhật.")
            return

        new_chunks, processed_files = load_and_process_documents(new_docs_dir)
        if not new_chunks:
            print("Không tìm thấy tài liệu mới hợp lệ.")
            return

        # Lấy tên file từ metadata
        new_file_names = set()
        for chunk in new_chunks:
            if "file_name" in chunk.metadata:
                new_file_names.add(chunk.metadata["file_name"])

        old_files = [
            os.path.basename(f)
            for f in glob.glob(os.path.join(old_docs_dir, "**", "*.*"), recursive=True)
        ]

        existing = [f for f in old_files if f in new_file_names]

        if existing:
            print("\nBạn muốn cập nhật những file sau:")
            for f in existing:
                print(f"  - {f}")
            confirm = input("\nNhập 0 để hủy bỏ, Nhập Enter để tiếp tục: ")
            if confirm.strip() == "0":
                print("Đã hủy bỏ cập nhật.")
                return

            # Xóa các file cũ từ vector store
            for chunk in new_chunks:
                if "file_name" in chunk.metadata and chunk.metadata["file_name"] in existing:
                    vector_store.delete(
                        where={"file_name": chunk.metadata["file_name"]}
                    )
            print(f"Đã xóa {len(existing)} file cũ khỏi Knowledge Base.")

        print(f"\nĐang thêm {len(new_chunks)} chunks mới vào Knowledge Base...")
        vector_store.add_documents(new_chunks)
        # vector_store.persist()

        # Di chuyển file từ new_docs sang old_docs
        for file_path in processed_files:
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(old_docs_dir, file_name)

            if os.path.exists(destination_path):
                os.remove(destination_path)  # Xóa bản cũ

            shutil.move(file_path, destination_path)

        print(f"Đã di chuyển {len(processed_files)} file sang 'old_docs'.")
        print("--- Hoàn tất quy trình cập nhật ---")

    except Exception as e:
        print(f"Lỗi khi cập nhật database: {e}")

update_lock = threading.Lock()

def update_knowledge_base_auto():
    if update_lock.locked():
        print("[Auto Update] 🚧 Đang cập nhật, vui lòng đợi!")
        return {"success": False, "message": "Đang có tác vụ cập nhật khác."}

    with update_lock:  # Chỉ 1 update được phép chạy
        try:
            print("\n[Auto Update] Bắt đầu cập nhật tri thức...")

            os.makedirs(OLD_DOCS_DIR, exist_ok=True)
            os.makedirs(NEW_DOCS_DIR, exist_ok=True)

            # Load DB (không tạo mới nếu có sẵn)
            db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)

            # Update từ thư mục new_docs
            check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)

            print("[Auto Update] Hoàn tất cập nhật tri thức.")
            return {"success": True, "message": "Cập nhật thành công"}

        except Exception as e:
            print(f"[Auto Update] Lỗi: {e}")
            return {"success": False, "message": str(e)}


if __name__ == "__main__":
    print("=== Bắt đầu quá trình xây dựng/cập nhật Knowledge Base ===\n")

    os.makedirs(OLD_DOCS_DIR, exist_ok=True)
    os.makedirs(NEW_DOCS_DIR, exist_ok=True)

    # 1. Khởi tạo hoặc tải Vector Store
    db = initialize_vector_store(CHROMA_DB_PATH, EMBEDDING_MODEL, OLD_DOCS_DIR)

    # 2. Kiểm tra và cập nhật từ thư mục new_docs
    check_and_update_database(db, NEW_DOCS_DIR, OLD_DOCS_DIR)

    print("\n=== Hoàn tất ===")