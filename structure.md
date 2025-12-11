
Đồ ÁN 2/
├── backend/                    # Backend API và xử lý logic
│   ├── config.py              # Cấu hình tập trung (MongoDB, LLM, embeddings, performance)
│   ├── main.py                # FastAPI entry point
│   │
│   ├── core/                  # Core processing pipeline
│   │   ├── chunking.py        # Chia văn bản thành chunks với semantic chunking
│   │   ├── embedding.py       # Tạo embeddings cho chunks và entities (SentenceTransformer)
│   │   ├── extraction.py      # 🆕 Extract entities/relationships (LightRAG-style với gleaning + LLM merge)
│   │   ├── graph_builder.py   # Xây dựng knowledge graph từ entities/relationships
│   │   └── pipeline.py        # Orchestrate toàn bộ pipeline (chunk → extract → build graph)
│   │
│   ├── db/                    # Database và storage
│   │   ├── mongo_storage.py   # MongoDB operations (graph, entities, relationships, chunks)
│   │   ├── vector_db.py       # FAISS vector database cho similarity search
│   │   ├── entity_linking.py  # Link entities giữa các chunks (fuzzy matching)
│   │   ├── entity_validator.py # Validate entities (type, description quality)
│   │   ├── conversation_storage.py # Lưu trữ conversation history
│   │   └── user_manager.py    # Quản lý users và permissions
│   │
│   ├── retrieval/             # Retrieval và query processing
│   │   ├── query_analyzer.py  # Phân tích query (intent, entities, keywords)
│   │   ├── vector_retriever.py # Vector search trên chunks
│   │   ├── graph_retriever.py  # Graph traversal từ entities
│   │   ├── hybrid_retriever.py # 🆕 Dual-level retrieval (global + local, LightRAG-inspired)
│   │   └── conversation_manager.py # Quản lý conversation context
│   │
│   ├── utils/                 # Utilities
│   │   ├── llm_utils.py       # LLM API calls (OpenAI, Groq)
│   │   ├── file_utils.py      # File processing (PDF, DOCX, TXT)
│   │   └── utils.py           # General utilities
│   │
│   └── data/                  # Data storage (user uploads, vectors)
│       └── {user_id}/
│           ├── uploads/       # Uploaded files
│           └── vectors/       # FAISS indices
│
├── frontend/                  # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   └── App.js            # Main app
│   └── package.json
│
├── lib/                       # Shared libraries
│
├── .env                       # Environment variables (API keys, configs)
├── .env.example              # Example env file với documentation
├── requirements.txt          # Python dependencies
└── structure.md              # This file