# scripts/init_vector_db.py
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings

def initialize_kb():
    print("🚀 Initializing Vector Knowledge Base...")
    
    # Financial Knowledge Base - Curated snippets
    knowledge_data = [
        "RSI (Relative Strength Index) above 70 usually indicates an asset is overbought, while below 30 indicates it is oversold.",
        "A Bullish LSTM prediction with high confidence suggest price momentum is likely to continue in the near term.",
        "Company earnings reports and quarterly guidance are the primary drivers of stock volatility and investor sentiment.",
        "Moving averages (SMA/EMA) help identify the trend direction and potential support/resistance levels.",
        "High trade volume during a price surge confirms the strength of the trend, whereas low volume raises suspicion of a fake-out.",
        "P/E Ratio (Price-to-Earnings) measures a company's current share price relative to its per-share earnings, used to value companies.",
        "Market Capitalization (Market Cap) is the total dollar market value of a company's outstanding shares of stock.",
        "Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio."
    ]
    
    documents = [Document(page_content=text, metadata={"source": "expert_knowledge"}) for text in knowledge_data]
    
    # Initialize embeddings
    print(f"📦 Loading embedding model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
    # Create Vector Store
    print("🛠️ Creating FAISS index...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # Save to disk
    os.makedirs(settings.vector_db_path, exist_ok=True)
    vector_db.save_local(settings.vector_db_path)
    
    print(f"✅ Knowledge Base saved to: {settings.vector_db_path}")

if __name__ == "__main__":
    initialize_kb()
