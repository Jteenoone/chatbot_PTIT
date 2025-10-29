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
        Hãy chỉ trả lời các câu hỏi liên quan đến PTIT.
        Nếu câu hỏi nằm ngoài phạm vi PTIT, hãy trả lời:
        "Mình chỉ có thể cung cấp thông tin liên quan đến Học viện Công nghệ Bưu chính Viễn thông (PTIT)."

        --- Dữ liệu tri thức từ PTIT ---
        {context}
        --------------------------------
        Dựa trên thông tin trên, hãy trả lời ngắn gọn, rõ ràng:
        {question}
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
