#!/usr/bin/env python3
"""
=======================================================
LightRAG Integration Test v2.0
=======================================================
Kiểm tra toàn bộ pipeline với mock LLM, async an toàn, cleanup đầy đủ
"""

import os
import sys
import json
import asyncio
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== TEST RESULTS ====================
class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.start_time = datetime.now()
    
    def add(self, name: str, passed: bool, msg: str = "", err: str = "", warn: bool = False):
        result = {
            'test': name, 'passed': passed, 'warning': warn,
            'message': msg, 'error': err, 'time': datetime.now().isoformat()
        }
        self.tests.append(result)
        if warn: self.warnings += 1
        elif passed: self.passed += 1
        else: self.failed += 1
        level = logging.WARNING if warn else logging.INFO if passed else logging.ERROR
        prefix = "WARN" if warn else "PASS" if passed else "FAIL"
        logger.log(level, f"{prefix}: {name} - {msg or err}")

    def summary(self) -> bool:
        duration = (datetime.now() - self.start_time).total_seconds()
        total = self.passed + self.failed
        print("\n" + "="*70)
        print("INTEGRATION TEST SUMMARY")
        print("="*70)
        print(f"Total: {total} | PASS: {self.passed} | FAIL: {self.failed} | WARN: {self.warnings}")
        print(f"Duration: {duration:.2f}s | Success: {(self.passed/total*100):.1f}%" if total else "N/A")
        print("="*70)
        if self.failed:
            print("\nFAILED:")
            for t in self.tests:
                if not t['passed'] and not t['warning']:
                    print(f"  • {t['test']}: {t['error']}")
        return self.failed == 0


# ==================== MOCK FILE ====================
class MockFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._content = content
    def getbuffer(self): return self._content
    def read(self): return self._content


def create_test_docs() -> List[MockFile]:
    return [
        MockFile("overview.md", b"""
# LightRAG System

## Features
- Chunking with hierarchy
- Entity extraction
- Knowledge graph
- Vector search

LightRAG combines **knowledge graphs** and **vector embeddings**.
        """.strip()),
        MockFile("tech.txt", b"""
LightRAG uses:
- tiktoken for token counting
- FAISS for vector search
- NetworkX for graphs
- SentenceTransformers for embeddings
        """.strip())
    ]


# ==================== MOCK LLM (AN TOÀN) ====================
_original_call = None
_original_retry = None

async def mock_llm(*args, **kwargs):
    await asyncio.sleep(0.01)
    return """entity<|>LightRAG<|>TECHNOLOGY<|>RAG system with graph and vector##
entity<|>Knowledge Graph<|>CONCEPT<|>Structured knowledge##
relationship<|>LightRAG<|>Knowledge Graph<|>uses<|>0.9<|COMPLETE|>"""

def setup_llm_mock():
    global _original_call, _original_retry
    try:
        from backend.utils import llm_utils
        _original_call = llm_utils.call_llm_async
        _original_retry = llm_utils.call_llm_with_retry
        llm_utils.call_llm_async = mock_llm
        llm_utils.call_llm_with_retry = mock_llm
    except: pass

def restore_llm():
    if _original_call:
        from backend.utils import llm_utils
        llm_utils.call_llm_async = _original_call
        llm_utils.call_llm_with_retry = _original_retry


# ==================== TEST FUNCTIONS ====================
def test_imports(results: TestResults):
    logger.info("\nTEST 1: Imports")
    imports = [
        ("backend.core.chunking", "process_document_to_chunks"),
        ("backend.core.extraction", "extract_entities_relations"),
        ("backend.core.graph_builder", "build_knowledge_graph"),
        ("backend.core.embedding", "VectorDatabase"),
        ("backend.core.pipeline", "DocumentPipeline"),
    ]
    for mod, func in imports:
        try:
            m = __import__(mod, fromlist=[func])
            getattr(m, func)
            results.add(f"import:{mod}.{func}", True, "OK")
        except Exception as e:
            results.add(f"import:{mod}.{func}", False, err=str(e))

def test_upload(results: TestResults):
    logger.info("\nTEST 2: Upload")
    from backend.utils.file_utils import save_uploaded_file
    user = "test_user_sync"
    paths = []
    for doc in create_test_docs():
        try:
            path = save_uploaded_file(doc, user_id=user)
            assert Path(path).exists()
            paths.append(path)
            results.add(f"upload:{doc.name}", True, f"Saved: {path}")
        except Exception as e:
            results.add(f"upload:{doc.name}", False, err=str(e))
    test_upload.paths = paths

async def test_chunking(results: TestResults):
    logger.info("\nTEST 3: Chunking")
    from backend.core.chunking import process_document_to_chunks, DocChunkConfig
    if not hasattr(test_upload, 'paths'): 
        results.add("chunking", False, err="No upload paths")
        return
    all_chunks = {}
    for path in test_upload.paths:
        try:
            chunks = process_document_to_chunks(path, DocChunkConfig(max_tokens=200, overlap_tokens=30))
            assert len(chunks) > 0
            assert all(isinstance(c['hierarchy'], list) for c in chunks), "hierarchy must be list"
            assert all('hierarchy_list' in c for c in chunks)
            name = Path(path).name
            all_chunks[name] = chunks
            results.add(f"chunk:{name}", True, f"{len(chunks)} chunks, hierarchy=list")
        except Exception as e:
            results.add(f"chunk:{Path(path).name}", False, err=str(e))
    test_chunking.chunks = all_chunks

async def test_extraction(results: TestResults):
    logger.info("\nTEST 4: Extraction")
    setup_llm_mock()
    from backend.core.extraction import extract_entities_relations
    if not hasattr(test_chunking, 'chunks'):
        results.add("extraction", False, err="No chunks")
        restore_llm()
        return
    total_ents = total_rels = 0
    for name, chunks in test_chunking.chunks.items():
        try:
            global_config = {
                "entity_types": ["TECHNOLOGY", "CONCEPT", "PRODUCT"]
            }
            ents, rels = extract_entities_relations(chunks)
            e_count = sum(len(v) for v in ents.values())
            r_count = sum(len(v) for v in rels.values())
            total_ents += e_count
            total_rels += r_count
            results.add(f"ext:{name}", True, f"{e_count}e {r_count}r")
        except Exception as e:
            results.add(f"ext:{name}", False, err=str(e))
    results.add("extraction:total", total_ents > 0 or total_rels > 0, f"{total_ents}e {total_rels}r")
    restore_llm()

def test_graph(results: TestResults):
    logger.info("\nTEST 5: Graph")
    if not hasattr(test_chunking, 'chunks') or not test_chunking.chunks:
        results.add("graph", False, err="No chunks")
        return
    from backend.core.graph_builder import build_knowledge_graph
    try:
        chunks = list(test_chunking.chunks.values())[0]
        from backend.core.extraction import extract_entities_relations
        ents, rels = extract_entities_relations(chunks, {"entity_types": ["TECHNOLOGY"]})
        kg = build_knowledge_graph(ents, rels)
        stats = kg.get_statistics()
        results.add("graph", stats['num_entities'] > 0, f"{stats['num_entities']} nodes")
    except Exception as e:
        results.add("graph", False, err=str(e))

def test_embedding(results: TestResults):
    logger.info("\nTEST 6: Embedding")
    if not hasattr(test_chunking, 'chunks') or not test_chunking.chunks:
        results.add("embedding", False, err="No chunks")
        return
    from backend.core.embedding import generate_embeddings, VectorDatabase
    try:
        chunks = list(test_chunking.chunks.values())[0]
        embeds = generate_embeddings(chunks)
        assert len(embeds) == len(chunks)
        assert all(len(e['embedding']) == 384 for e in embeds)
        db = VectorDatabase(db_path="test.index", metadata_path="test_meta.json", dim=384)
        db.add_embeddings(embeds)
        db.save()
        assert Path("test.index").exists()
        results.add("embedding", True, f"{len(embeds)} vectors saved")
    except Exception as e:
        results.add("embedding", False, err=str(e))
    finally:
        for f in ["test.index", "test_meta.json"]:
            if Path(f).exists(): Path(f).unlink()

def test_pipeline(results: TestResults):
    logger.info("\nTEST 7: Pipeline")
    setup_llm_mock()
    from backend.core.pipeline import DocumentPipeline, DocChunkConfig
    pipeline = DocumentPipeline(user_id="test_pipe", enable_advanced=True)
    doc = create_test_docs()[0]
    try:
        res = pipeline.process_uploaded_file(
            doc, DocChunkConfig(max_tokens=200, overlap_tokens=30),
            enable_extraction=True, enable_graph=True, enable_embedding=True
        )
        assert res['success']
        assert res['chunks_count'] > 0
        results.add("pipeline", True, f"{res['chunks_count']} chunks")
    except Exception as e:
        results.add("pipeline", False, err=str(e))
    finally:
        restore_llm()

def test_persistence(results: TestResults):
    logger.info("\nTEST 8: Persistence")
    from backend.core.pipeline import DocumentPipeline
    pipeline = DocumentPipeline(user_id="test_pipe")
    docs = pipeline.get_processed_docs()
    results.add("persistence", len(docs) > 0, f"{len(docs)} docs found")


# ==================== MAIN ====================
async def main():
    print("\nLIGHTTAG INTEGRATION TEST v2.0")
    print("="*70)
    results = TestResults()

    # Apply nest_asyncio if in interactive env
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except: pass

    test_imports(results)
    test_upload(results)
    await test_chunking(results)
    await test_extraction(results)
    test_graph(results)
    test_embedding(results)
    test_pipeline(results)
    test_persistence(results)

    success = results.summary()

    # Save results
    os.makedirs("test_output", exist_ok=True)
    with open("test_output/results.json", "w", encoding="utf-8") as f:
        json.dump(results.tests, f, indent=2, ensure_ascii=False)

    # Cleanup
    for d in ["backend/data/test_user_sync", "backend/data/test_pipe"]:
        if Path(d).exists():
            shutil.rmtree(d, ignore_errors=True)
    print(f"\nResults: test_output/results.json")
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)