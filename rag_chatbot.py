import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from faq_service import FAQService

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DB_PATH = "./knowledge_base_ptit"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


class RAGChatbot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )

        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.3
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # Prompt cho RAG
        template = """
        Bạn là trợ lý ảo của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
        Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về trường.
        Nguyên tắc trả lời bắt buộc:
        -Trả lời chính xác, trực tiếp vào câu hỏi.
        -Không được thêm lời chào hỏi, cảm ơn hoặc câu xã giao không cần thiết.
        -Chỉ trả lời các câu hỏi liên quan đến quản lý và đào tạo của PTIT. Được cung cấp trong file tài liệu tin cậy đã được upload lên.
        -Nếu người dùng hỏi bình thường thì chỉ trích xuất và trả lời tổng quan ngắn gọn, đầy đủ thông tin mà người dùng cần, nếu hỏi chi tiết thì trả lời đầy đủ các thông tin có liên quan.
        -KHÔNG được nói "Xin lỗi", "Tôi không biết", "Không có thông tin" - PHẢI trả lời dựa trên context có sẵn
        -Nếu context không đủ, hãy suy luận LOGIC từ thông tin có sẵn mà KHÔNG bịa thêm.
        -Sử dụng ngôn ngữ tự nhiên, dễ hiểu, phù hợp với phong cách trả lời của con người.

        -Nếu người dùng hỏi về các trường khác, hoặc chủ đề không liên quan tới PTIT,
        hãy trả lời: "Tôi chỉ có thể cung cấp thông tin liên quan đến Học viện Công nghệ Bưu chính Viễn thông (PTIT)."

        Dưới đây là thông tin lấy được từ tài liệu nội bộ của PTIT (nếu có):
        ----------------
        {context}
        ----------------
        Dựa trên thông tin trên, hãy trả lời câu hỏi:
        {question}

        Nếu không có dữ liệu rõ ràng trong context, hãy nói:
        “Hiện tại tôi chưa có dữ liệu cụ thể về vấn đề này, nhưng bạn có thể tham khảo tại website chính thức của PTIT: https://ptit.edu.vn”
        -Phong cách phản hồi:
            -Viết bằng tiếng Việt.
            -Giọng điệu thân thiện, rõ ràng, lịch sự..
        """

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}

        )

        self.faq = FAQService(self.embeddings.model)

    def get_answer(self, question: str):
        """Trả lời câu hỏi dựa trên dữ liệu RAG"""
        try:
            faq_ans = None
            if hasattr(self, "faq"):
                faq_ans = self.faq.check(question)
            if faq_ans:
                return faq_ans
            response = self.qa_chain.invoke({"query": question})
            result = response.get("result", "").strip()

            if not result:
                return "Mình chưa có dữ liệu về vấn đề này, bạn có thể hỏi lại cách khác nhé."
            return result

        except Exception as e:
            return f"Lỗi khi truy vấn RAG: {str(e)}"

    def close(self):
        """Đóng kết nối tới Chroma để có thể xóa DB."""
        try:
            if hasattr(self, "db") and hasattr(self.db, "client"):
                self.db.client._client.teardown()  # đóng kết nối Chroma
            elif hasattr(self, "db") and hasattr(self.db, "_client"):
                self.db._client.teardown()
        except Exception as e:
            print("Không thể đóng DB:", e)
