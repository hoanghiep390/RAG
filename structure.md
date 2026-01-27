Đồ ÁN 2/
├── backend/                    # Backend xử lý logic và API
│   ├── config.py              # Cấu hình tập trung (MongoDB, LLM, Embeddings, Performance)
│   ├── main.py                # Entry point (hiện tại trống - dự phòng cho FastAPI)
│   │
│   ├── core/                  # Core Processing Pipeline
│   │   ├── chunking.py        # Chia văn bản thành chunks (Docling + Semantic Chunking)
│   │   ├── embedding.py       # Tạo embeddings (SentenceTransformer)
│   │   ├── extraction.py      # Trích xuất entities/relationships (LightRAG-style + Gleaning)
│   │   ├── graph_builder.py   # Xây dựng Knowledge Graph (NetworkX)
│   │   └── pipeline.py        # Orchestrate pipeline (Chunk → Extract → Graph → Embed)
│   │
│   ├── db/                    
│   │   ├── mongo_storage.py   # MongoDB operations (documents, chunks, entities, relationships, graph)
│   │   ├── vector_db.py       # FAISS vector database (similarity search)
│   │   ├── entity_linking.py  # Entity linking giữa documents (fuzzy matching)
│   │   ├── entity_validator.py # Validate entities (type, description quality)
│   │   ├── conversation_storage.py # Lưu conversation history
│   │   ├── feedback_storage.py # Lưu user feedback (ratings, comments)
│   │   └── user_manager.py    # Quản lý users (authentication, roles)
│   │
│   ├── retrieval/             
│   │   ├── query_analyzer.py  # Phân tích query (intent, entities, keywords)
│   │   ├── vector_retriever.py # Vector search trên chunks
│   │   ├── graph_retriever.py  # Graph traversal từ entities
│   │   ├── hybrid_retriever.py # Dual-level retrieval (Global + Local, LightRAG-inspired)
│   │   ├── retrieval_cache.py  # Cache retrieval results (TTL-based)
│   │   └── conversation_manager.py # Quản lý conversation context & query rewriting
│   │
│   ├── utils/                
│   │   ├── llm_utils.py       # LLM API calls (OpenAI, Groq) với streaming support
│   │   ├── file_utils.py      # File operations (save uploaded files)
│   │   └── utils.py           # Logging và utilities chung
│   │
│   └── data/                  
│       └── {user_id}/
│           ├── uploads/       # Tài liệu gốc đã upload
│           ├── vectors/       # FAISS index files
│           └── logs/          # Application logs
│
├── frontend/                 
│   ├── login.py              #  Login/Signup page (authentication)
│   │
│   └── pages/                #  Application Pages
│       ├── upload.py         # Upload documents (Admin only)
│       ├── chat.py           # Multi-conversation chat interface
│       ├── graph.py          # Knowledge graph visualization (Admin only)
│       └── analytics.py      #  Analytics dashboard (Admin only)
│
│
├── venv/                      #Python Virtual Environment
│
├── .env                      # Environment Variables (API keys, configs)
├── .env.example              # Example env file
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── structure.md              # This file