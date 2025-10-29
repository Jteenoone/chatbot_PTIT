import os, json, pickle, time
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

FAQ_FILE = "local_faq.json"
VEC_FILE = "faq_vectors.pkl"
THRESHOLD = 0.75

class FAQService:
    def __init__(self, model_name="text-embedding-3-small"):
        self.path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        self.vec_path = os.path.join(os.path.dirname(__file__), VEC_FILE)
        self.emb = OpenAIEmbeddings(model=model_name)
        self.faq = self._load_faq()
        self.vectors = self._load_or_build_vectors()

    def _load_faq(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _load_or_build_vectors(self):
        # load cached
        if os.path.exists(self.vec_path):
            try:
                with open(self.vec_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        # build
        questions = [q.get("question","") for q in self.faq]
        if not questions:
            return []
        vectors = self.emb.embed_documents(questions)
        with open(self.vec_path, "wb") as f:
            pickle.dump(vectors, f)
        return vectors

    @staticmethod
    def _cosine(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def check(self, text: str) -> Optional[str]:
        """Trả answer string nếu match, hoặc None"""
        if not self.faq or not self.vectors:
            return None
        # short queries hơn ưu tiên FAQ
        if len(text.split()) > 50:
            return None
        try:
            qvec = self.emb.embed_query(text)
        except Exception:
            return None
        best_idx, best_score = None, 0.0
        for i, v in enumerate(self.vectors):
            s = self._cosine(qvec, v)
            if s > best_score:
                best_score, best_idx = s, i
        if best_idx is not None and best_score >= THRESHOLD:
            return self.faq[best_idx].get("answer")
        return None

    def rebuild(self):
        """Force rebuild vectors (call after editing local_faq.json)"""
        self.faq = self._load_faq()
        if os.path.exists(self.vec_path):
            try:
                os.remove(self.vec_path)
            except Exception:
                pass
        self.vectors = self._load_or_build_vectors()
