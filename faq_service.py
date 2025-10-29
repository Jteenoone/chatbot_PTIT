import os
import json
import pickle
from typing import Optional, Tuple, List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

FAQ_FILE = "local_faq.json"
FAQ_VECTOR_FILE = "faq_vectors.pkl"
FAQ_THRESHOLD = 0.80


class FAQService:

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.faq_data = self._load_faq_data()
        self.faq_vectors = self._load_or_create_vectors()

    def _load_faq_data(self) -> List[dict]:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            print(f"Không tìm thấy {FAQ_FILE}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_or_create_vectors(self) -> List[np.ndarray]:
        vec_file = os.path.join(os.path.dirname(__file__), FAQ_VECTOR_FILE)

        # Load nếu tồn tại
        if os.path.exists(vec_file):
            with open(vec_file, "rb") as f:
                return pickle.load(f)

        # Chưa có vector, tạo mới
        questions = [faq["question"] for faq in self.faq_data]
        if not questions:
            return []

        vectors = self.embeddings.embed_documents(questions)

        with open(vec_file, "wb") as f:
            pickle.dump(vectors, f)

        print("Đã tạo embedding cho FAQ và lưu vào faq_vectors.pkl")
        return vectors

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def check(self, user_question: str) -> Optional[Tuple[str, float]]:
        if not self.faq_data or not self.faq_vectors:
            return None

        uq = user_question.lower().strip()
        if len(uq.split()) > 12:
            return None

        user_vec = self.embeddings.embed_query(uq)

        best_score = 0.0
        best_answer = None

        for faq, faq_vec in zip(self.faq_data, self.faq_vectors):
            score = self._cosine_similarity(user_vec, faq_vec)
            if score > best_score:
                best_score = score
                best_answer = faq["answer"]

        if best_score >= FAQ_THRESHOLD:
            return best_answer, best_score
        return None
