mini-lightrag/
│
├── 📁 backend/                              # Backend processing logic
│   │
│   ├── 📁 core/                             # Core processing modules (Pure functions)
│   │   ├── 📄 chunking.py                   # ✂️ Text → Chunks (300 tokens default)
│   │   │   ├── extract_segments()           # PDF, DOCX, TXT, CSV, JSON, XML
│   │   │   ├── Chunker class               # Smart chunking with overlap
│   │   │   └── process_document_to_chunks() # Main entry point
│   │   │
│   │   ├── 📄 embedding.py                  # 🧮 Text → Vectors (384-dim)
│   │   │   ├── EmbeddingModel              # SentenceTransformer wrapper
│   │   │   ├── generate_embeddings()       # Chunk embeddings (batch 128)
│   │   │   ├── generate_entity_embeddings() # Entity embeddings
│   │   │   └── generate_relationship_embeddings()
│   │   │
│   │   ├── 📄 extraction.py                 # 🔍 Chunks → Entities/Relations
│   │   │   ├── extract_entities()          # 16 parallel LLM calls
│   │   │   ├── parse_extraction_result()   # LightRAG format parser
│   │   │   └── extract_entities_relations() # Sync wrapper
│   │   │
│   │   ├── 📄 graph_builder.py              # 🕸️ Entities → Knowledge Graph
│   │   │   ├── KnowledgeGraph class        # NetworkX DiGraph wrapper
│   │   │   ├── build_knowledge_graph()     # Async graph builder
│   │   │   ├── _merge_nodes_then_upsert()  # Smart node merging
│   │   │   └── _merge_edges_then_upsert()  # Smart edge merging
│   │   │
│   │   └── 📄 pipeline.py                   # 🔄 Main orchestrator
│   │       ├── DocumentPipeline class      # Unified processing
│   │       ├── process_file()              # Single file (progress tracking)
│   │       └── process_multiple_files_parallel() # Multi-file (3x parallel)
│   │
│   ├── 📁 db/                               # Storage layer
│   │   ├── 📄 mongo_storage.py              # 🗄️ MongoDB operations
│   │   │   ├── save_document()             # Document metadata
│   │   │   ├── save_chunks_bulk()          # Bulk chunk insert
│   │   │   ├── save_entities_bulk()        # Bulk entity insert
│   │   │   ├── save_relationships_bulk()   # Bulk relationship insert
│   │   │   ├── save_graph_bulk()           # Bulk graph upsert
│   │   │   ├── delete_document_cascade()   # Cascade delete
│   │   │   └── save_document_complete()    # All-in-one save
│   │   │
│   │   └── 📄 vector_db.py                  # 🚀 FAISS operations
│   │       ├── VectorDatabase class        # FAISS manager
│   │       ├── add_document_embeddings_batch() # Batch add
│   │       ├── search()                    # Vector search
│   │       ├── delete_document()           # Mark deleted
│   │       └── rebuild_index()             # Compact index
│   │
│   ├── 📁 utils/                            # Utility functions
│   │   ├── 📄 file_utils.py                 # 📁 File operations
│   │   │   ├── save_uploaded_file()        # Save to uploads/
│   │   │   ├── read_file_content()         # Read text files
│   │   │   ├── get_file_info()             # File metadata
│   │   │   └── delete_uploaded_file()      # Remove file
│   │   │
│   │   ├── 📄 llm_utils.py                  # 🤖 LLM API calls
│   │   │   ├── call_openai_async()         # OpenAI GPT
│   │   │   ├── call_groq_async()           # Groq Llama
│   │   │   ├── call_llm_async()            # Universal async
│   │   │   └── call_llm_batch()            # Batch processing
│   │   │
│   │   ├── 📄 utils.py                      # 📝 Logging setup
│   │   │   └── logger                      # Configured logger
│   │   │
│   │   └── 📄 cache_utils.py                # ⚠️ DEPRECATED (do not use)
│   │
│   ├── 📄 config.py                         # ⚙️ MongoDB configuration
│   │   ├── MongoDBConfig class             # Connection manager
│   │   ├── get_mongodb()                   # Get DB instance
│   │   └── close_mongodb()                 # Close connection
│   │
│   ├── 📄 main.py                           # (Empty placeholder)
│   │
│   └── 📁 data/                             # 💾 User data storage
│       └── {user_id}/                       # Per-user isolation
│           ├── uploads/                     # 📄 Original uploaded files
│           ├── vectors/                     # 🚀 FAISS indexes
│           │   ├── combined.index          # FAISS index file
│           │   ├── combined_metadata.json  # Chunk metadata
│           │   └── document_map.json       # Doc-to-index mapping
│           └── logs/                        # 📝 Processing logs
│
├── 📁 frontend/                             # Streamlit UI
│   ├── 📄 login.py                          # 🔐 Login/Register page
│   │   ├── User authentication             # SHA256 password hashing
│   │   ├── Session management              # st.session_state
│   │   └── Default admin account           # admin/admin123
│   │
│   └── 📁 pages/                            # Multi-page app
│       ├── 📄 upload.py                     # 📤 Document upload & processing
│       │   ├── File uploader               # Multiple files support
│       │   ├── Processing pipeline         # With progress bars
│       │   ├── MongoDB + FAISS save        # Bulk operations
│       │   ├── Document list               # View processed docs
│       │   ├── Unified delete              # MongoDB + FAISS + Files
│       │   └── FAISS rebuild UI            # Optimize index
│       │
│       └── 📄 graph.py                      # 🕸️ Knowledge graph viewer
│           ├── Load from MongoDB           # Get combined graph
│           ├── Interactive visualization   # PyVis network graph
│           ├── Statistics dashboard        # Nodes, edges, types
│           ├── Entity browser              # Search & filter
│           └── Relationship browser        # View connections
│
├── 📁 scripts/                              # Utility scripts (if any)
│
├── 📄 .env                                  # 🔑 Environment variables
│   ├── MONGODB_URI                         # MongoDB connection string
│   ├── MONGODB_DATABASE                    # Database name
│   ├── LLM_PROVIDER                        # openai / groq
│   ├── LLM_MODEL                           # Model name
│   ├── OPENAI_API_KEY                      # OpenAI API key
│   ├── GROQ_API_KEY                        # Groq API key
│   ├── MAX_CONCURRENT_LLM_CALLS            # 16 (default)
│   ├── EXTRACTION_BATCH_SIZE               # 20 (default)
│   └── EMBEDDING_BATCH_SIZE                # 128 (default)
│
├── 📄 .env.example                          # 📝 Example config
├── 📄 .gitignore                            # Git ignore rules
│
├── 📄 requirements.txt                      # 📦 Python dependencies
│
├── 📄 structure.md                          # 📁 This file
