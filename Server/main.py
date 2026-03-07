import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────
# 0.  INITIALIZE CLIENT & APP
# ─────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 1.  DOCUMENT LOADER + CHUNKER
# ─────────────────────────────────────────────
def load_document(file_path: str) -> str:
    """Load text from .txt or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            raise ImportError("Run: pip install pdfplumber")

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")


def chunk_text(text: str, chunk_size: int = 20, overlap: int = 10) -> list:
    """Split large text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# 2.  RETRIEVER
# ─────────────────────────────────────────────
class SimpleRetriever:
    def __init__(self, documents: list):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.1):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.documents[i], scores[i])
            for i in top_indices
            if scores[i] >= threshold
        ]


# ─────────────────────────────────────────────
# 3.  LOAD DOCUMENT & BUILD RETRIEVER AT STARTUP
# ─────────────────────────────────────────────
DOCUMENT_PATH = os.environ.get("DOCUMENT_PATH", "database.txt")  # ← set via env or change default

retriever = None

def init_retriever():
    global retriever
    if os.path.exists(DOCUMENT_PATH):
        raw_text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(raw_text, chunk_size=20, overlap=10)
        retriever = SimpleRetriever(chunks)
        print(f"✅ Loaded: {DOCUMENT_PATH} ({len(chunks)} chunks)")
    else:
        print(f"⚠️  Document not found at '{DOCUMENT_PATH}'. RAG disabled — fallback to general knowledge.")

init_retriever()


# ─────────────────────────────────────────────
# 4.  RAG PIPELINE
# ─────────────────────────────────────────────
def rag_query(query: str) -> str:
    relevant_docs = retriever.retrieve(query) if retriever else []

    if not relevant_docs:
        prompt = f"""You are a helpful assistant. Answer the following question to the best of your ability.

QUESTION: {query}

ANSWER:"""
    else:
        context_text = "\n".join([f"- {doc}" for doc, score in relevant_docs])
        prompt = f"""You are a helpful assistant. You have knowledge about certain people and topics.
Answer naturally as if this is your own knowledge — never say "context", "provided context", "based on context", or "according to the context".
Just answer like a human who already knows this information.

YOUR KNOWLEDGE:
{context_text}

RULES:
1. Answer naturally and confidently as if you already know this.
2. Never mention "context", "document", "provided information" or similar words.
3. If the question has wrong assumptions, correct them naturally.
4. For general questions, answer from your general knowledge.
5. Always format links as clickable markdown like [Project Name](url).
6. Always use the exact URLs from your knowledge, never modify or guess URLs.
7. Always copy URLs exactly as given, character by character. Never add, remove or change anything in a URL.
8. Never combine or merge URLs from different projects.

QUESTION: {query}

ANSWER:"""

    response = client.models.generate_content(
        model="gemma-3-1b-it",
        contents=prompt
    )
    return response.text


# ─────────────────────────────────────────────
# 5.  ROUTE  (same API as before)
# ─────────────────────────────────────────────
@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    query = data.get("prompt")

    if not query:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    answer = rag_query(query)
    return answer


if __name__ == "__main__":
    app.run(debug=True)
