import faiss
import numpy as np
import os
import json

class FaissDB:
    def __init__(self, db_path="faiss.index", metadata_path="faiss_meta.json", dim=384):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.dim = dim

        # Khởi tạo FAISS index nếu chưa có
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        # Khởi tạo metadata nếu chưa có
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf8") as f:
                self.metadatas = json.load(f)
        else:
            self.metadatas = []

    def add_embeddings(self, items):
        # items: list dict gồm 'id', 'text', 'embedding'
        vecs = np.array([item["embedding"] for item in items]).astype("float32")
        self.index.add(vecs)
        self.metadatas.extend(items)
        faiss.write_index(self.index, self.db_path)
        with open(self.metadata_path, "w", encoding="utf8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def search(self, text_embedding, topk=5):
        vec = np.array([text_embedding]).astype("float32")
        scores, idxs = self.index.search(vec, topk)
        return [self.metadatas[i] for i in idxs[0]]
