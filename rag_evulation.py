from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

#embedding model used in your RAG system
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_cosine_similarity(text1: str, text2: str) -> float:
    embeddings = embedding_model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def relevance_score(query: str, retrieved_docs: List[str]) -> float:
    scores = [get_cosine_similarity(query, doc) for doc in retrieved_docs]
    return float(np.mean(scores))

def groundedness_score(answer: str, retrieved_docs: List[str]) -> float:
    answer_words = set(answer.lower().split())
    doc_words = set(" ".join(retrieved_docs).lower().split())
    if not answer_words:
        return 0.0
    overlap = answer_words.intersection(doc_words)
    return len(overlap) / len(answer_words)

# Example
if __name__ == "__main__":
    user_query = "বাংলা সাহিত্যের জনক কে?"
    retrieved_context = [
        "বাংলা সাহিত্যের জনক হিসেবে মুকুন্দরাম চক্রবর্তীকে অনেকে বিবেচনা করেন।",
        "মধ্যযুগে বাংলা সাহিত্যের সূচনা হয় চণ্ডী কাব্য দ্বারা।"
    ]
    generated_answer = "বাংলা সাহিত্যের জনক মুকুন্দরাম চক্রবর্তী।"

    relevance = relevance_score(user_query, retrieved_context)
    groundedness = groundedness_score(generated_answer, retrieved_context)

    print(f"Relevance Score: {relevance:.4f}")
    print(f"Groundedness Score: {groundedness:.4f}")
