"""
🎯 RAG Evaluation - Simple All-in-One Script
Đánh giá 3 metrics: Answer Relevancy, Faithfulness, Response Time
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.db.vector_db import VectorDatabase
from backend.db.mongo_storage import MongoStorage
from backend.retrieval.hybrid_retriever import EnhancedHybridRetriever
from backend.utils.llm_utils import call_llm
import asyncio

# ============================================
# 📊 TEST DATASET
# ============================================

TEST_CASES = [
    {
        "question": "What is artificial intelligence?",
        "expected_keywords": ["ai", "machine learning", "intelligence"]
    },
    {
        "question": "How does machine learning work?",
        "expected_keywords": ["algorithm", "data", "training", "model"]
    },
    {
        "question": "What are the applications of AI?",
        "expected_keywords": ["application", "use", "technology"]
    },
    # Thêm câu hỏi của bạn ở đây dựa trên documents đã upload
]

# ============================================
# 🎯 EVALUATOR CLASS
# ============================================

class RAGEvaluator:
    """Simple RAG Evaluator với 3 metrics"""
    
    def __init__(self, user_id: str = "admin_00000000"):
        """Initialize evaluator"""
        self.user_id = user_id
        
        # Initialize RAG components
        self.vector_db = VectorDatabase(user_id)
        self.mongo_storage = MongoStorage(user_id)
        self.retriever = EnhancedHybridRetriever(self.vector_db, self.mongo_storage)
        
        print(f"✅ Evaluator initialized for user: {user_id}")
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Metric 1: Answer Relevancy (1-5)
        Sử dụng LLM để đánh giá độ liên quan
        """
        prompt = f"""Rate how relevant this answer is to the question on a scale of 1-5.

Question: {question}

Answer: {answer}

Rating (1-5, where 5 is most relevant): """
        
        try:
            response = call_llm(
                prompt,
                system_prompt="You are an evaluator. Respond with only a number from 1 to 5.",
                temperature=0.0,
                max_tokens=10
            )
            
            # Extract number
            score = float(response.strip().split()[0])
            return min(max(score, 1.0), 5.0)
        
        except Exception as e:
            print(f"⚠️ Error evaluating relevancy: {e}")
            return 3.0  # Default middle score
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Metric 2: Faithfulness (0-1)
        Kiểm tra answer có dựa trên context không (hallucination check)
        """
        prompt = f"""Check if the answer is faithful to the context (no hallucination).

Context:
{context[:1000]}

Answer:
{answer}

Is the answer based on the context? Answer with:
- 1.0 if fully faithful (all claims in answer are in context)
- 0.5 if partially faithful (some claims not in context)
- 0.0 if not faithful (hallucination)

Score (0.0, 0.5, or 1.0): """
        
        try:
            response = call_llm(
                prompt,
                system_prompt="You are an evaluator. Respond with only 0.0, 0.5, or 1.0.",
                temperature=0.0,
                max_tokens=10
            )
            
            score = float(response.strip().split()[0])
            return min(max(score, 0.0), 1.0)
        
        except Exception as e:
            print(f"⚠️ Error evaluating faithfulness: {e}")
            return 0.5  # Default middle score
    
    def evaluate_single(self, question: str) -> Dict:
        """
        Đánh giá 1 câu hỏi với cả 3 metrics
        """
        print(f"\n{'='*80}")
        print(f"❓ Question: {question}")
        print(f"{'='*80}")
        
        # Metric 3: Response Time
        start_time = time.time()
        
        # Retrieve context
        context = self.retriever.retrieve(query=question, top_k=5)
        
        # Build prompt
        system_prompt = """You are a helpful AI assistant. Answer based on the provided context."""
        
        user_prompt = f"""{context.formatted_text}

Question: {question}

Answer:"""
        
        # Generate answer
        answer = call_llm(user_prompt, system_prompt=system_prompt, temperature=0.7)
        
        response_time = time.time() - start_time
        
        print(f"\n💬 Answer: {answer[:200]}...")
        print(f"\n⏱️ Response Time: {response_time:.2f}s")
        
        # Metric 1: Answer Relevancy
        relevancy_score = self.evaluate_answer_relevancy(question, answer)
        print(f"📊 Answer Relevancy: {relevancy_score}/5.0")
        
        # Metric 2: Faithfulness
        # Extract context text from formatted context
        context_text = context.formatted_text
        faithfulness_score = self.evaluate_faithfulness(answer, context_text)
        print(f"✅ Faithfulness: {faithfulness_score}/1.0")
        
        return {
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "relevancy_score": relevancy_score,
            "faithfulness_score": faithfulness_score,
            "num_chunks": len(context.global_chunks),
            "num_entities": len(context.local_entities)
        }
    
    def evaluate_dataset(self, test_cases: List[Dict]) -> Dict:
        """
        Đánh giá toàn bộ dataset
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING EVALUATION")
        print(f"{'='*80}")
        print(f"📝 Total test cases: {len(test_cases)}")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            result = self.evaluate_single(test_case["question"])
            results.append(result)
        
        # Calculate averages
        avg_relevancy = sum(r["relevancy_score"] for r in results) / len(results)
        avg_faithfulness = sum(r["faithfulness_score"] for r in results) / len(results)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        summary = {
            "total_cases": len(test_cases),
            "avg_relevancy": avg_relevancy,
            "avg_faithfulness": avg_faithfulness,
            "avg_response_time": avg_response_time,
            "detailed_results": results
        }
        
        return summary

# ============================================
# 📈 REPORT GENERATOR
# ============================================

def print_report(summary: Dict):
    """In báo cáo evaluation"""
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION REPORT")
    print(f"{'='*80}")
    print(f"\n🎯 Overall Metrics:")
    print(f"  • Answer Relevancy:  {summary['avg_relevancy']:.2f}/5.0")
    print(f"  • Faithfulness:      {summary['avg_faithfulness']:.2f}/1.0")
    print(f"  • Avg Response Time: {summary['avg_response_time']:.2f}s")
    
    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(summary['detailed_results'], 1):
        print(f"\n[{i}] {result['question']}")
        print(f"    Relevancy: {result['relevancy_score']}/5.0")
        print(f"    Faithfulness: {result['faithfulness_score']}/1.0")
        print(f"    Time: {result['response_time']:.2f}s")
        print(f"    Chunks: {result['num_chunks']}, Entities: {result['num_entities']}")
    
    print(f"\n{'='*80}")

def save_report(summary: Dict, output_file: str = "evaluation_results.json"):
    """Lưu báo cáo ra file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Report saved to: {output_file}")

# ============================================
# 🚀 MAIN
# ============================================

def main():
    """Run evaluation"""
    print("🎯 RAG Evaluation System")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(user_id="admin_00000000")
    
    # Run evaluation
    summary = evaluator.evaluate_dataset(TEST_CASES)
    
    # Print report
    print_report(summary)
    
    # Save report
    save_report(summary)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()
