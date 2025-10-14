import os
import getpass
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Cấu hình và Tải API Key ---
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    print("Không tìm thấy API key. Vui lòng nhập:")
    os.environ["OPENAI_API_KEY"] = getpass.getpass()

LLM_MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
HISTORY_FILE = "./chat_history.json"


# --- Lớp Quản lý Lịch sử Chat ---
class ChatHistoryManager:
    """Quản lý và lưu trữ lịch sử cuộc trò chuyện vào file JSON."""
    def __init__(self, history_file=HISTORY_FILE):
        self.history_file = history_file
        self.history = self._load_history()

    def add_message(self, role, content, sources=None):
        self.history.append({
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        })
        self._save_history()

    def show(self, limit=10):
        if not self.history:
            return "Lịch sử trò chuyện trống."
        # Lấy các tin nhắn gần nhất
        recent = self.history[-limit:]
        result = []
        for i, msg in enumerate(recent, 1):
            timestamp = msg.get("timestamp", "")
            role = msg["role"].capitalize()
            content = msg["content"][:80] # Giới hạn độ dài nội dung
            result.append(f"{i}. [{timestamp[:16]}] {role}: {content}...")
        return "\n".join(result)

    def clear(self):
        self.history = []
        self._save_history()
        print("Đã xóa lịch sử trò chuyện.")

    def _save_history(self):
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Lỗi khi lưu lịch sử: {e}")

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Lỗi khi tải lịch sử: {e}")
        return []

# --- Các Hàm Tiện Ích ---
def format_docs(docs):
    """Nối nội dung các tài liệu thành một chuỗi duy nhất."""
    return "\n\n".join(doc.page_content for doc in docs)

def display_menu():
    """Hiển thị menu các lệnh có thể sử dụng."""
    print("\n" + "=" * 60)
    print("- MENU LỆNH -")
    print("=" * 60)
    print("  'thoát' / 'exit'     - Kết thúc phiên trò chuyện")
    print("  'history'            - Xem 10 tin nhắn gần nhất")
    print("  'clear'              - Xóa toàn bộ lịch sử chat")
    print("  'help'               - Hiển thị menu này")
    print("=" * 60 + "\n")

# --- Hàm tạo RAG Chain ---
def create_rag_chain():
    """Tạo chuỗi RAG để xử lý và trả lời câu hỏi."""
    try:
        # 1. Khởi tạo embedding model
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # 2. Tải vector store từ ổ đĩa
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # 3. Tạo retriever để tìm kiếm tài liệu
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 4. Định nghĩa prompt template cho AI
        template = """
        Bạn là một trợ lý ảo hỗ trợ sinh viên tại Học viện Công nghệ Bưu chính Viễn thông (PTIT).
        Trả lời câu hỏi của người dùng cách chuyên nghiệp, ngắn gọn và chỉ dựa vào BỐI CẢNH được cung cấp.
        Nếu thông tin không có trong bối cảnh, hãy nói: "Xin lỗi, mình chưa có thông tin về vấn đề này trong tài liệu tham khảo."
        Luôn trả lời bằng tiếng Việt và bắt đầu bằng lời chào thân thiện.

        BỐI CẢNH:
        {context}

        CÂU HỎI:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. Khởi tạo mô hình ngôn ngữ
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)

        # 6. Xây dựng RAG chain bằng LCEL
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        print("✅ Trợ lý ảo đã sẵn sàng!")
        return rag_chain

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo RAG chain: {e}")
        return None

# --- Hàm Chính ---
def main():
    """Hàm chính để chạy ứng dụng chatbot."""
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Lỗi: Không tìm thấy 'thư viện' kiến thức tại '{CHROMA_DB_PATH}'.")
        print("Vui lòng chạy file `build_database.py` để tạo 'thư viện' trước.")
        return

    rag_chain = create_rag_chain()
    if rag_chain is None:
        return

    history_manager = ChatHistoryManager()

    print("\n" + "=" * 60)
    print(" CHÀO MỪNG BẠN ĐẾN VỚI TRỢ LÝ ẢO PTIT ")
    print("=" * 60)
    print("Gõ 'help' để xem các lệnh.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Bạn hỏi: ").strip()

            if not user_input:
                continue

            # Xử lý các lệnh đặc biệt
            if user_input.lower() in ["thoat", "exit", "thoát"]:
                print("Trợ lý PTIT: Tạm biệt! Hẹn gặp lại bạn.")
                break
            if user_input.lower() == "help":
                display_menu()
                continue
            if user_input.lower() == "history":
                print("\n--- Lịch sử trò chuyện ---\n")
                print(history_manager.show(10))
                print("\n--------------------------\n")
                continue
            if user_input.lower() == "clear":
                confirm = input("Bạn có chắc chắn muốn xóa toàn bộ lịch sử? (yes/no): ")
                if confirm.lower() == "yes":
                    history_manager.clear()
                else:
                    print("Đã hủy thao tác.")
                continue

            # Xử lý câu hỏi thông thường
            print("Trợ lý PTIT: Đang tìm kiếm câu trả lời...")
            response = rag_chain.invoke(user_input)
            print(f"Trợ lý PTIT: {response}\n")

            # Lưu vào lịch sử
            history_manager.add_message("user", user_input)
            history_manager.add_message("bot", response)

        except KeyboardInterrupt:
            print("\n\nTrợ lý PTIT: Tạm biệt!")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}\n")

if __name__ == "__main__":
    main()