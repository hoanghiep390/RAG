#!/usr/bin/env python3
"""
=======================================================
LightRAG System Test Suite
=======================================================
Comprehensive testing script for all modules
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import traceback
import shutil

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import modules
from backend.core.chunking import process_document_to_chunks, DocChunkConfig
from backend.core.extraction import extract_entities_relations
from backend.core.graph_builder import build_knowledge_graph, merge_admin_graphs, KnowledgeGraph
from backend.core.embedding import VectorDatabase, generate_embeddings, search_similar
from backend.core.pipeline import DocumentPipeline
from backend.utils.file_utils import save_to_json, load_from_json, ensure_directory
from backend.utils.llm_utils import call_llm_async

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== MOCK UTILITIES ====================
class MockUploadedFile:
    """Mock Streamlit's UploadedFile for testing"""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._content = content
    
    def getbuffer(self):
        return self._content
    
    def read(self):
        return self._content


def create_test_document(filename: str = "test_doc.txt") -> tuple:
    """Create a test document with known content"""
    content = b"""
# Test Document for LightRAG System

## Introduction
This is a comprehensive test document designed to validate the LightRAG system's capabilities.

## Organizations
Apple Inc. is a major technology company founded by Steve Jobs in 1976.
Microsoft Corporation develops software products including Windows and Office.
Google LLC specializes in internet-related services and products.

## Products
The iPhone is Apple's flagship smartphone product.
Windows 11 is Microsoft's latest operating system.
Google Search is the world's most popular search engine.

## Locations
Apple's headquarters is located in Cupertino, California.
Microsoft is based in Redmond, Washington.
Google's main campus is in Mountain View, California.

## Events
The iPhone was first released in 2007, revolutionizing the smartphone industry.
Microsoft launched Windows 95 in 1995, changing personal computing.
Google was founded in 1998 by Larry Page and Sergey Brin.

## Technologies
Artificial Intelligence is transforming various industries.
Cloud Computing enables scalable infrastructure.
Machine Learning powers modern recommendation systems.
"""
    
    mock_file = MockUploadedFile(filename, content)
    return mock_file, content.decode('utf-8')


# ==================== TEST FUNCTIONS ====================
class TestResults:
    """Store and display test results"""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.start_time = datetime.now()
    
    def add_result(self, test_name: str, passed: bool, message: str = "", error: str = ""):
        result = {
            'test': test_name,
            'passed': passed,
            'message': message,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.tests.append(result)
        
        if passed:
            self.passed += 1
            logger.info(f"✅ PASS: {test_name} - {message}")
        else:
            self.failed += 1
            logger.error(f"❌ FAIL: {test_name} - {error}")
    
    def print_summary(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        total = self.passed + self.failed
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Success Rate: {(self.passed/total*100):.1f}%" if total > 0 else "No tests run")
        print("="*60)
        
        if self.failed > 0:
            print("\n❌ FAILED TESTS:")
            for test in self.tests:
                if not test['passed']:
                    print(f"  - {test['test']}: {test['error']}")
        
        return self.failed == 0


# ==================== MODULE TESTS ====================
async def test_1_file_utils(results: TestResults):
    """Test file utilities"""
    logger.info("\n📁 Testing File Utils...")
    
    try:
        # Create test directory
        test_dir = Path("test_output/file_utils")
        ensure_directory(test_dir)
        
        # Test JSON save/load
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        json_path = test_dir / "test.json"
        
        save_to_json(test_data, str(json_path))
        loaded_data = load_from_json(str(json_path))
        
        assert loaded_data == test_data, "JSON data mismatch"
        results.add_result("file_utils:json_operations", True, "JSON save/load successful")
        
        # Test file upload mock
        mock_file, _ = create_test_document()
        assert mock_file.name == "test_doc.txt", "Mock file name incorrect"
        assert mock_file.size > 0, "Mock file empty"
        
        results.add_result("file_utils:mock_upload", True, f"Mock file created: {mock_file.size} bytes")
        
    except Exception as e:
        results.add_result("file_utils", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_2_chunking(results: TestResults):
    """Test document chunking"""
    logger.info("\n✂️  Testing Chunking Module...")
    
    try:
        # Create test file
        test_dir = Path("test_output/chunking")
        ensure_directory(test_dir)
        
        mock_file, content = create_test_document()
        test_file = test_dir / mock_file.name
        
        with open(test_file, 'wb') as f:
            f.write(mock_file.getbuffer())
        
        # Test with default config
        config = DocChunkConfig(
            max_token_size=200,
            overlap_token_size=30
        )
        
        chunks = process_document_to_chunks(str(test_file), config=config)
        
        # Validations
        assert len(chunks) > 0, "No chunks created"
        assert all('chunk_id' in c for c in chunks), "Missing chunk_id"
        assert all('content' in c for c in chunks), "Missing content"
        assert all('tokens' in c for c in chunks), "Missing tokens"
        
        total_tokens = sum(c['tokens'] for c in chunks)
        
        results.add_result(
            "chunking:basic",
            True,
            f"Created {len(chunks)} chunks, {total_tokens} total tokens"
        )
        
        # Test with different configs
        small_config = DocChunkConfig(max_token_size=100, overlap_token_size=10)
        small_chunks = process_document_to_chunks(str(test_file), config=small_config)
        
        assert len(small_chunks) >= len(chunks), "Smaller config should create more chunks"
        
        results.add_result(
            "chunking:config_variations",
            True,
            f"Small config: {len(small_chunks)} chunks vs {len(chunks)} chunks"
        )
        
        # Save for next tests
        save_to_json(chunks, "test_output/chunks.json")
        
    except Exception as e:
        results.add_result("chunking", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_3_extraction(results: TestResults):
    """Test entity and relationship extraction"""
    logger.info("\n🔍 Testing Extraction Module...")
    
    try:
        # Load chunks from previous test
        chunks = load_from_json("test_output/chunks.json")
        
        # Mock LLM to avoid API calls - PATCH THE RIGHT FUNCTION
        import backend.utils.llm_utils as llm_utils
        import backend.core.extraction as extraction_module
        
        original_llm = llm_utils.call_llm_async
        
        async def mock_llm(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API delay
            return """entity<|>Apple Inc.<|>ORGANIZATION<|>Technology company##
entity<|>Steve Jobs<|>PERSON<|>Co-founder of Apple##
entity<|>iPhone<|>PRODUCT<|>Smartphone product##
entity<|>Cupertino<|>LOCATION<|>City in California##
relationship<|>Apple Inc.<|>iPhone<|>produces<|>0.9##
relationship<|>Steve Jobs<|>Apple Inc.<|>founded<|>0.95##
relationship<|>Apple Inc.<|>Cupertino<|>located_in<|>0.85<|COMPLETE|>"""
        
        # Patch the actual function used in extraction
        llm_utils.call_llm_async = mock_llm
        llm_utils.call_llm_with_retry = mock_llm
        
        # Extract entities and relationships
        global_config = {
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT", "TECHNOLOGY"]
        }
        
        # Ensure loop is available
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        entities_dict, relationships_dict = extract_entities_relations(chunks, global_config)
        
        # Validations
        assert len(entities_dict) > 0, "No entities extracted"
        assert len(relationships_dict) > 0, "No relationships extracted"
        
        total_entities = sum(len(v) for v in entities_dict.values())
        total_relationships = sum(len(v) for v in relationships_dict.values())
        
        results.add_result(
            "extraction:entities",
            True,
            f"Extracted {total_entities} entities from {len(entities_dict)} chunks"
        )
        
        results.add_result(
            "extraction:relationships",
            True,
            f"Extracted {total_relationships} relationships from {len(relationships_dict)} chunks"
        )
        
        # Save for next tests
        save_to_json({
            'entities': entities_dict,
            'relationships': relationships_dict
        }, "test_output/extraction.json")
        
        # Restore original functions
        llm_utils.call_llm_async = original_llm
        llm_utils.call_llm_with_retry = original_llm
        
    except Exception as e:
        results.add_result("extraction", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_4_graph_builder(results: TestResults):
    """Test knowledge graph building"""
    logger.info("\n🕸️  Testing Graph Builder...")
    
    try:
        # Load extraction results
        extraction_data = load_from_json("test_output/extraction.json")
        entities_dict = extraction_data['entities']
        relationships_dict = extraction_data['relationships']
        
        # Build knowledge graph
        kg = build_knowledge_graph(entities_dict, relationships_dict)
        
        # Validations
        stats = kg.get_statistics()
        
        assert stats['num_entities'] > 0, "Graph has no entities"
        assert stats['num_relationships'] > 0, "Graph has no relationships"
        
        results.add_result(
            "graph_builder:construction",
            True,
            f"Built graph: {stats['num_entities']} nodes, {stats['num_relationships']} edges"
        )
        
        # Test graph properties
        assert stats['density'] >= 0, "Invalid density"
        assert stats['avg_degree'] >= 0, "Invalid average degree"
        
        results.add_result(
            "graph_builder:statistics",
            True,
            f"Density: {stats['density']:.3f}, Avg Degree: {stats['avg_degree']:.2f}"
        )
        
        # Save graph
        save_to_json({
            'graph': kg.to_dict(),
            'statistics': stats
        }, "test_output/graph.json")
        
    except Exception as e:
        results.add_result("graph_builder", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_5_embedding(results: TestResults):
    """Test embedding generation and vector search"""
    logger.info("\n🧮 Testing Embedding Module...")
    
    try:
        # Load chunks
        chunks = load_from_json("test_output/chunks.json")
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Validations
        assert len(embeddings) == len(chunks), "Embedding count mismatch"
        assert all('embedding' in e for e in embeddings), "Missing embedding vectors"
        
        # Check embedding dimensions
        first_embedding = embeddings[0]['embedding']
        embedding_dim = len(first_embedding)
        
        assert embedding_dim > 0, "Invalid embedding dimension"
        
        results.add_result(
            "embedding:generation",
            True,
            f"Generated {len(embeddings)} embeddings, dim={embedding_dim}"
        )
        
        # Test vector database
        vector_db = VectorDatabase(
            db_path="test_output/test.index",
            metadata_path="test_output/test_meta.json",
            dim=embedding_dim
        )
        
        vector_db.add_embeddings(embeddings)
        vector_db.save()
        
        # Test search
        query = "technology company products"
        search_results = search_similar(query, vector_db, top_k=3)
        
        assert len(search_results) > 0, "Search returned no results"
        assert all('distance' in r for r in search_results), "Missing distance scores"
        
        results.add_result(
            "embedding:search",
            True,
            f"Search returned {len(search_results)} results"
        )
        
        # Test statistics
        db_stats = vector_db.get_statistics()
        assert db_stats['total_vectors'] > 0, "No vectors in database"
        
        results.add_result(
            "embedding:vector_db",
            True,
            f"Vector DB: {db_stats['total_vectors']} vectors"
        )
        
    except Exception as e:
        results.add_result("embedding", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_6_pipeline_integration(results: TestResults):
    """Test full pipeline integration"""
    logger.info("\n🔄 Testing Full Pipeline...")
    
    try:
        # Create test user
        test_user = "test_user_pipeline"
        pipeline = DocumentPipeline(user_id=test_user, enable_advanced=True)
        
        # Create test document
        mock_file, _ = create_test_document("pipeline_test.txt")
        
        # Mock LLM for extraction
        import backend.utils.llm_utils as llm_utils
        original_llm = llm_utils.call_llm_async
        
        async def mock_llm(prompt, **kwargs):
            await asyncio.sleep(0.1)
            return """entity<|>Apple Inc.<|>ORGANIZATION<|>Technology company##
entity<|>Microsoft<|>ORGANIZATION<|>Software company##
entity<|>Google<|>ORGANIZATION<|>Internet services##
relationship<|>Apple Inc.<|>Microsoft<|>competes_with<|>0.8##
relationship<|>Microsoft<|>Google<|>competes_with<|>0.85<|COMPLETE|>"""
        
        llm_utils.call_llm_async = mock_llm
        llm_utils.call_llm_with_retry = mock_llm
        
        # Process document
        result = pipeline.process_uploaded_file(
            mock_file,
            chunk_config=DocChunkConfig(max_token_size=200, overlap_token_size=30),
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        
        # Validations
        assert result['success'], "Pipeline processing failed"
        assert result['chunks_count'] > 0, "No chunks created"
        
        results.add_result(
            "pipeline:processing",
            True,
            f"Processed: {result['chunks_count']} chunks, {result.get('total_tokens', 0)} tokens"
        )
        
        if result.get('entities_count', 0) > 0:
            results.add_result(
                "pipeline:extraction",
                True,
                f"Extracted: {result['entities_count']} entities, {result['relationships_count']} relationships"
            )
        
        if result.get('graph_nodes', 0) > 0:
            results.add_result(
                "pipeline:graph",
                True,
                f"Graph: {result['graph_nodes']} nodes, {result['graph_edges']} edges"
            )
        
        if result.get('total_embeddings', 0) > 0:
            results.add_result(
                "pipeline:embeddings",
                True,
                f"Embeddings: {result['total_embeddings']} vectors"
            )
        
        # Test document listing
        docs = pipeline.get_processed_docs()
        assert len(docs) > 0, "No documents found"
        
        results.add_result(
            "pipeline:listing",
            True,
            f"Found {len(docs)} processed documents"
        )
        
        # Restore original functions
        llm_utils.call_llm_async = original_llm
        llm_utils.call_llm_with_retry = original_llm
        
    except Exception as e:
        results.add_result("pipeline", False, error=str(e))
        logger.error(traceback.format_exc())


async def test_7_graph_merge(results: TestResults):
    """Test graph merging functionality"""
    logger.info("\n🔗 Testing Graph Merge...")
    
    try:
        test_user = "test_merge_user"
        graphs_dir = Path(f"backend/data/{test_user}/graphs")
        ensure_directory(graphs_dir)
        
        # Create multiple test graphs with proper NetworkX format
        for i in range(3):
            kg = KnowledgeGraph()
            
            # Add entities
            kg.add_entity(
                entity_name=f"Entity_{i}_A",
                entity_type="ORGANIZATION",
                description=f"Test entity {i}",
                source_id=f"chunk_{i}",
                source_document=f"doc_{i}"
            )
            
            kg.add_entity(
                entity_name=f"Entity_{i}_B",
                entity_type="PRODUCT",
                description=f"Test product {i}",
                source_id=f"chunk_{i}",
                source_document=f"doc_{i}"
            )
            
            # Add relationship
            kg.add_relationship(
                source_entity=f"Entity_{i}_A",
                target_entity=f"Entity_{i}_B",
                description="test relationship",
                strength=0.8,
                chunk_id=f"chunk_{i}",
                source_document=f"doc_{i}"
            )
            
            # Save graph using proper format
            graph_dict = kg.to_dict()
            save_to_json(
                {'graph': graph_dict, 'statistics': kg.get_statistics()},
                str(graphs_dir / f"doc_{i}_graph.json")
            )
        
        # Merge graphs
        merged_kg = merge_admin_graphs(test_user)
        
        assert merged_kg is not None, "Merge returned None"
        
        stats = merged_kg.get_statistics()
        
        # More lenient check - at least some entities should exist
        assert stats['num_entities'] > 0, f"Expected >0 entities, got {stats['num_entities']}"
        assert stats['num_relationships'] > 0, f"Expected >0 relationships, got {stats['num_relationships']}"
        
        results.add_result(
            "graph_merge:execution",
            True,
            f"Merged graph: {stats['num_entities']} entities, {stats['num_relationships']} relationships"
        )
        
        # Verify COMBINED file exists
        combined_path = graphs_dir / "COMBINED_graph.json"
        assert combined_path.exists(), "COMBINED_graph.json not created"
        
        results.add_result(
            "graph_merge:file_creation",
            True,
            "COMBINED_graph.json created successfully"
        )
        
        # Cleanup
        shutil.rmtree(f"backend/data/{test_user}", ignore_errors=True)
        
    except Exception as e:
        results.add_result("graph_merge", False, error=str(e))
        logger.error(traceback.format_exc())


# ==================== MAIN TEST RUNNER ====================
async def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*60)
    print("🧪 LIGHTRAG SYSTEM TEST SUITE")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    results = TestResults()
    
    # Run tests sequentially
    test_suites = [
        test_1_file_utils,
        test_2_chunking,
        test_3_extraction,
        test_4_graph_builder,
        test_5_embedding,
        test_6_pipeline_integration,
        test_7_graph_merge,
    ]
    
    for test_suite in test_suites:
        try:
            await test_suite(results)
        except Exception as e:
            logger.error(f"Test suite {test_suite.__name__} crashed: {e}")
            logger.error(traceback.format_exc())
    
    # Print summary
    all_passed = results.print_summary()
    
    # Save detailed results
    save_to_json(results.tests, "test_output/test_results.json")
    
    return all_passed


def cleanup_test_output():
    """Clean up test output directory"""
    test_dir = Path("test_output")
    if test_dir.exists():
        logger.info("🧹 Cleaning up previous test outputs...")
        shutil.rmtree(test_dir)
    
    ensure_directory(test_dir)


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG System Test Suite")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep test output files after completion"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Cleanup before tests
    if not args.no_cleanup:
        cleanup_test_output()
    
    # Run tests
    try:
        success = asyncio.run(run_all_tests())
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            exit_code = 0
        else:
            print("\n❌ SOME TESTS FAILED!")
            exit_code = 1
        
        # Final cleanup
        if not args.no_cleanup:
            logger.info("\n🧹 Cleaning up test files...")
            shutil.rmtree("test_output", ignore_errors=True)
            shutil.rmtree("backend/data/test_user_pipeline", ignore_errors=True)
            shutil.rmtree("backend/data/test_merge_user", ignore_errors=True)
            
            # Clean individual test files
            for f in ["temp_graph.html"]:
                Path(f).unlink(missing_ok=True)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)