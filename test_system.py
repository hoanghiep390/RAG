import os
import sys
import json
from pathlib import Path
import asyncio
import logging
from io import BytesIO
from typing import Any

# Thêm backend vào sys.path để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

# Import các modules từ code của bạn
from backend.core.pipeline import DocumentPipeline, DocChunkConfig
from backend.core.chunking import process_document_to_chunks
from backend.core.extraction import extract_entities_relations
from backend.core.graph_builder import build_knowledge_graph, merge_admin_graphs
from backend.core.embedding import VectorDatabase, search_similar
from backend.utils.file_utils import save_to_json, load_from_json, save_uploaded_file, delete_uploaded_file
from backend.utils.utils import logger
from backend.utils.llm_utils import call_llm_async  # Để mock nếu cần

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock UploadedFile cho Streamlit (vì script không chạy trong Streamlit)
class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content

    def getbuffer(self):
        return BytesIO(self.content)

# Mock LLM calls để test offline (trả về dữ liệu giả định)
def mock_call_llm_async(prompt, **kwargs):
    logger.info("Mocking LLM call...")
    return """
entity<|>Company A<|>ORGANIZATION<|>A tech company<|>
entity<|>Product X<|>PRODUCT<|>Main product<|>
relationship<|>Company A<|>Product X<|>produces<|>0.9<|COMPLETE|>
""".strip()

# Hàm test chính
async def test_workflow(user_id: str = "test_user"):
    logger.info("=== BẮT ĐẦU TEST WORKFLOW ===")
    
    # Step 0: Tạo file test tạm (nội dung đơn giản)
    test_content = b"""
    This is a test document.
    Company A produces Product X.
    It is located in Location Y.
    """
    test_filename = "test_document.txt"
    mock_file = MockUploadedFile(test_filename, test_content)
    
    # Step 1: Test file_utils - Save uploaded file
    try:
        saved_path = save_uploaded_file(mock_file, user_id=user_id)
        logger.info(f"Step 1: File saved at {saved_path}")
        assert Path(saved_path).exists(), "File không được lưu!"
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return False
    
    # Step 2: Khởi tạo pipeline
    pipeline = DocumentPipeline(user_id=user_id, enable_advanced=True)
    
    # Step 3: Test chunking.py - Chunk document
    try:
        config = DocChunkConfig(max_token_size=100, overlap_token_size=20)
        chunks = process_document_to_chunks(saved_path, config=config)
        logger.info(f"Step 3: Chunked into {len(chunks)} chunks")
        assert len(chunks) > 0, "Không có chunks!"
        save_to_json(chunks, "test_chunks.json")
    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        return False
    
    # Step 4: Test extraction.py - Extract entities/relations (mock LLM để nhanh)
    global call_llm_async
    original_llm = call_llm_async
    call_llm_async = mock_call_llm_async  # Mock để tránh gọi API thật
    
    try:
        global_config = {"entity_types": ["ORGANIZATION", "PRODUCT", "LOCATION"]}
        entities, relationships = extract_entities_relations(chunks, global_config)
        logger.info(f"Step 4: Extracted {len(entities)} entity chunks, {len(relationships)} relationship chunks")
        assert len(entities) > 0 or len(relationships) > 0, "Không extract được gì!"
        save_to_json({"entities": entities, "relationships": relationships}, "test_extraction.json")
    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        call_llm_async = original_llm  # Restore
        return False
    
    call_llm_async = original_llm  # Restore sau mock
    
    # Step 5: Test graph_builder.py - Build knowledge graph
    try:
        kg = build_knowledge_graph(entities, relationships)
        stats = kg.get_statistics()
        logger.info(f"Step 5: Built graph with {stats['num_entities']} entities, {stats['num_relationships']} relationships")
        assert stats['num_entities'] > 0, "Graph rỗng!"
        save_to_json(kg.to_dict(), "test_graph.json")
        
        # Test merge (giả lập nhiều graph)
        merge_admin_graphs(user_id)
        merged_path = Path(f"backend/data/{user_id}/graphs/COMBINED_graph.json")
        assert merged_path.exists(), "Merge graph thất bại!"
    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        return False
    
    # Step 6: Test embedding.py - Generate embeddings and search
    try:
        vector_db = VectorDatabase(db_path="test_faiss.index", metadata_path="test_faiss_meta.json")
        # Giả lập process_file trong embedding.py (dùng pipeline để tích hợp)
        result = pipeline.process_uploaded_file(
            mock_file,
            chunk_config=config,
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        logger.info(f"Step 6: Pipeline result: {result}")
        assert result['success'], "Pipeline thất bại!"
        
        # Test search
        query = "Company A"
        results = search_similar(query, vector_db, top_k=3)
        logger.info(f"Search results: {len(results)} items")
        assert len(results) > 0, "Search không tìm thấy gì!"
    except Exception as e:
        logger.error(f"Step 6 failed: {e}")
        return False
    
    # Step 7: Test pipeline.py full - Process uploaded file
    try:
        full_result = pipeline.process_uploaded_file(
            mock_file,
            chunk_config=config,
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        logger.info(f"Step 7: Full pipeline success: {full_result['success']}")
        assert full_result['success'], "Full pipeline thất bại!"
    except Exception as e:
        logger.error(f"Step 7 failed: {e}")
        return False
    
    # Step 8: Cleanup
    try:
        delete_uploaded_file(saved_path)
        for f in ["test_chunks.json", "test_extraction.json", "test_graph.json", "test_faiss.index", "test_faiss_meta.json"]:
            Path(f).unlink(missing_ok=True)
        # Xóa thư mục test_user
        import shutil
        shutil.rmtree(f"backend/data/{user_id}", ignore_errors=True)
        logger.info("Step 8: Cleanup done")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
    
    logger.info("=== TEST WORKFLOW HOÀN THÀNH - ALL PASS ===")
    return True

# Run test
if __name__ == "__main__":
    asyncio.run(test_workflow())