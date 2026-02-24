# app/core/learning.py
"""
User Feedback Learning System
Stores thumbs up/down feedback per Q&A pair.
Uses highly-rated Q&A pairs as few-shot examples in future prompts.
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "learning_data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")
KNOWLEDGE_FILE = os.path.join(FEEDBACK_DIR, "learned_knowledge.json")

os.makedirs(FEEDBACK_DIR, exist_ok=True)


def _load_json(path: str) -> list:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_json(path: str, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def save_feedback(question: str, answer: str, rating: str, chat_id: str = None):
    """Save user feedback (thumbs_up / thumbs_down) for a Q&A pair"""
    feedbacks = _load_json(FEEDBACK_FILE)
    entry = {
        "question": question,
        "answer": answer,
        "rating": rating,  # "thumbs_up" or "thumbs_down"
        "chat_id": chat_id,
        "timestamp": datetime.now().isoformat(),
    }
    feedbacks.append(entry)
    _save_json(FEEDBACK_FILE, feedbacks)
    logger.info(f"📝 Feedback saved: {rating} for \"{question[:50]}...\"")

    # If thumbs_up, also add to learned knowledge for future use
    if rating == "thumbs_up":
        _add_to_knowledge(question, answer)


def _add_to_knowledge(question: str, answer: str):
    """Store good Q&A pairs as learned knowledge for few-shot prompting"""
    knowledge = _load_json(KNOWLEDGE_FILE)

    # Avoid duplicates
    for k in knowledge:
        if k["question"].lower().strip() == question.lower().strip():
            k["answer"] = answer  # Update with latest good answer
            k["count"] = k.get("count", 1) + 1
            k["updated_at"] = datetime.now().isoformat()
            _save_json(KNOWLEDGE_FILE, knowledge)
            return

    knowledge.append({
        "question": question,
        "answer": answer[:500],  # Cap at 500 chars to keep prompt manageable
        "count": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    })

    # Keep only top 50 most useful entries
    knowledge.sort(key=lambda x: x.get("count", 1), reverse=True)
    knowledge = knowledge[:50]

    _save_json(KNOWLEDGE_FILE, knowledge)
    logger.info(f"🧠 Knowledge base updated: {len(knowledge)} entries")


def get_relevant_knowledge(query: str, max_examples: int = 3) -> List[Dict]:
    """Retrieve the most relevant learned Q&A pairs for few-shot prompting"""
    knowledge = _load_json(KNOWLEDGE_FILE)
    if not knowledge:
        return []

    query_lower = query.lower()
    scored = []
    for k in knowledge:
        q_lower = k["question"].lower()
        # Simple keyword overlap scoring
        query_words = set(query_lower.split())
        knowledge_words = set(q_lower.split())
        overlap = len(query_words & knowledge_words)
        score = overlap + k.get("count", 1) * 0.5
        if overlap > 0:
            scored.append((score, k))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:max_examples]]


def get_feedback_stats() -> Dict:
    """Get statistics about collected feedback"""
    feedbacks = _load_json(FEEDBACK_FILE)
    knowledge = _load_json(KNOWLEDGE_FILE)
    
    thumbs_up = sum(1 for f in feedbacks if f["rating"] == "thumbs_up")
    thumbs_down = sum(1 for f in feedbacks if f["rating"] == "thumbs_down")
    
    return {
        "total_feedback": len(feedbacks),
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "approval_rate": round(thumbs_up / max(len(feedbacks), 1) * 100, 1),
        "learned_entries": len(knowledge),
    }
