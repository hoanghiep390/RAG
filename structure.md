mini_lightrag_graph/
│
├── backend/                         # Backend xử lý logic LightRAG mini
│   │
│   ├── core/                             # Các module chính 
│   │   ├── chunking.py                   # → Tách văn bản thành các chunk nhỏ 
│   │   ├── embedding.py                  # → Sinh embedding vector từ mỗi chunk
│   │   ├── extraction.py                 # → Trích xuất entity và relationship từ văn bản 
│   │   ├── graph_builder.py              # → Xây dựng Knowledge Graph 
│   │   ├── retriever.py                  # → Tìm kiếm các chunk hoặc node liên quan đến truy vấn
│   │   ├── generator.py                  # → Gọi LLM để tạo câu trả lời từ ngữ cảnh đã truy xuất
│   │   └── pipeline.py                   # → Điều phối toàn bộ flow: upload → chunk → embed → graph → retrieve → answer
│   │
│   ├── db/                               #  Quản lý lưu trữ dữ liệu 
│   │   ├── vector_db.py                  # → Lưu trữ và truy vấn embedding 
│   │   ├── graph_storage.py              # → Lưu và tải Knowledge Graph 
│   │   ├── conversation_store.py         # → Lưu hội thoại tạm 
│   │   ├── conversation_mongo.py         # → Phiên bản lưu hội thoại bằng MongoDB 
│   │   └── user_store.py                 # → Quản lý người dùng: đăng ký, đăng nhập, hash mật khẩu
│   │
│   ├── utils/                            # Hàm tiện ích, dùng chung trong toàn hệ thống
│   │   ├── file_utils.py                 # → Xử lý file 
│   │   ├── text_utils.py                 # → Xử lý văn bản 
│   │   ├── llm_utils.py                  # → Gọi API LLM 
│   │   └── utils.py                  # → Ghi log hệ thống 
│   │
│   └── data/                             # Lưu trữ dữ liệu người dùng (dạng file, cho demo)
│       ├── uploads/                      # → Tài liệu gốc do người dùng upload
│       ├── chunks/                       # → Kết quả sau khi chia chunk
│       ├── graphs/                       # → File graph (JSON hoặc pickle)
│       ├── conversations/                # → Lịch sử hội thoại đã lưu
│       ├── logs/                         # → Log ghi lại hoạt động hệ thống
│       └── users.json                    # → File lưu thông tin người dùng (demo, không có DB)
│
├── frontend/                        # Frontend: giao diện người dùng bằng Streamlit
│   ├── login.py                      # → Trang đăng nhập / đăng ký người dùng
│   ├── upload.py                     # → Giao diện upload tài liệu và chạy pipeline xử lý
│   ├── chat.py                       # → Giao diện chat hỏi đáp theo tài liệu đã nạp
│   ├── graph.py                      # → Hiển thị đồ thị kiến thức (interactive graph viewer)
│   └── sidebar.py                        # → Thanh menu / hiển thị user info / chuyển trang
│
├── requirements.txt                      #  Danh sách thư viện cần cài đặt (streamlit, openai, faiss,…)
├── .env                                  #  Cấu hình môi trường (API key, DB URI,…)
├── .env.example                          #  Mẫu file .env để tham khảo
├── README.md                             #  Hướng dẫn cài đặt, chạy demo
│
└── docs/                                 # Tài liệu kiến trúc và hướng dẫn kỹ thuật
    ├── architecture.md                   # → Mô tả kiến trúc hệ thống và các module
    ├── data_flow.png                     # → Sơ đồ luồng dữ liệu qua các module
    └── api_reference.md                  # → Mô tả chi tiết API nội bộ (core/db/utils)
