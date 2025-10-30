import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
#from langchain.chains import RetrievalQA
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

        # ... giữ nguyên self.embeddings, self.vector_store, self.llm, self.retriever

        template = """
        Bạn là trợ lý ảo của Học viện Công nghệ Bưu chính Viễn thông (PTIT).
        Nhiệm vụ của bạn là chỉ trả lời các câu hỏi liên quan đến PTIT.
        Nếu người dùng hỏi về các trường khác, hoặc chủ đề không liên quan tới PTIT,
        hãy trả lời: "Tôi chỉ có thể cung cấp thông tin liên quan đến Học viện Công nghệ Bưu chính Viễn thông (PTIT)."

        Dưới đây là thông tin lấy được từ tài liệu nội bộ của PTIT (nếu có):
        ----------------
        {context}
        ----------------
        Dựa trên thông tin trên, hãy trả lời câu hỏi:
        {question}

        Nếu không tìm thấy câu trả lời trong tài liệu PTIT, hãy nói rõ rằng bạn chưa có dữ liệu cụ thể.
        """
        self.prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        self.faq = FAQService(self.embeddings.model)


    def get_answer(self, question: str):
    ###Trả lời câu hỏi dựa trên dữ liệu RAG (không dùng RetrievalQA cũ)###
        try:
            # 1) Check FAQ
            faq_ans = None
            if hasattr(self, "faq"):
                faq_ans = self.faq.check(question)
            if faq_ans:
                return faq_ans

            # 2) Lấy ngữ cảnh từ retriever
            docs = self.retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs) if docs else ""

            # 3) Lắp prompt và gọi LLM
            prompt_text = self.prompt.format(context=context, question=question)
            resp = self.llm.invoke(prompt_text)
            result = getattr(resp, "content", str(resp)).strip()

            # 4) Fallback nếu rỗng
            if not result:
                return "Mình chưa có dữ liệu về vấn đề này, bạn có thể hỏi lại cách khác nhé."
            return result

        except Exception as e:
            return f"Lỗi khi truy vấn RAG: {str(e)}"