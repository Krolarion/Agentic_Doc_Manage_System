# 数据库引擎：SQLite(元数据中心) + ChromaDB(向量检索) 职责分离
import sqlite3
import json
import uuid
from typing import Optional, List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from src.config import SQLITE_DB_PATH, CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


class DatabaseManager:
    """双库引擎：SQLite管理结构化元数据，ChromaDB负责向量语义检索"""

    def __init__(self):
        print("正在连接底层数据库引擎...")

        # SQLite：结构化元数据中心
        self.sqlite_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        self.sqlite_conn.execute("PRAGMA foreign_keys = ON")
        self._init_sqlite_tables()

        # ChromaDB：纯向量检索引擎
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # Embedding模型
        print(f"正在加载本地嵌入模型: {EMBEDDING_MODEL_NAME}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        print("[OK] 数据库双引擎启动完毕！\n")

    # Schema

    def _init_sqlite_tables(self):
        cursor = self.sqlite_conn.cursor()

        # 1. 文档主表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT,
                file_size_bytes INTEGER,
                page_count INTEGER,
                parse_status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 2. 标签字典
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 3. 文档-标签关联
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_tags (
                doc_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (doc_id, tag_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(tag_id) ON DELETE CASCADE
            )
        ''')

        # 4. Chunk文本块
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER,
                content TEXT,
                char_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        ''')

        # 5. QA对（含完整过滤元数据）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                qa_id TEXT PRIMARY KEY,
                chunk_id TEXT,
                doc_id TEXT,
                question TEXT,
                answer TEXT,
                faith_score REAL,
                faith_reasoning TEXT,
                diversity_max_sim REAL,
                relevancy_score REAL,
                filter_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        ''')

        # 6. 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 7. 处理审计日志
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                doc_id TEXT,
                details TEXT,
                status TEXT DEFAULT 'success',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE SET NULL
            )
        ''')

        self.sqlite_conn.commit()

    # 文档管理

    def register_document(self, file_name: str, file_path: str = "",
                          file_size_bytes: int = 0, page_count: int = 0) -> str:
        """注册新文档，返回doc_id"""
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT INTO documents (doc_id, file_name, file_path, file_size_bytes, page_count, parse_status)
            VALUES (?, ?, ?, ?, ?, 'parsing')
        ''', (doc_id, file_name, file_path, file_size_bytes, page_count))
        self.sqlite_conn.commit()
        return doc_id

    def update_document_status(self, doc_id: str, status: str, error_message: str = ""):
        """更新文档处理状态"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            UPDATE documents SET parse_status=?, error_message=?, updated_at=CURRENT_TIMESTAMP
            WHERE doc_id=?
        ''', (status, error_message, doc_id))
        self.sqlite_conn.commit()

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档完整信息（含标签）"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('SELECT * FROM documents WHERE doc_id=?', (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        doc = dict(zip([c[0] for c in cursor.description], row))

        # 关联标签
        cursor.execute('''
            SELECT t.name, t.category FROM tags t
            JOIN document_tags dt ON t.tag_id = dt.tag_id
            WHERE dt.doc_id = ?
        ''', (doc_id,))
        doc["tags"] = [{"name": r[0], "category": r[1]} for r in cursor.fetchall()]
        return doc

    # 标签管理

    def add_tag(self, name: str, category: str = "") -> int:
        """添加标签（已存在则返回已有ID），返回tag_id"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO tags (name, category) VALUES (?, ?)",
            (name, category)
        )
        self.sqlite_conn.commit()
        cursor.execute("SELECT tag_id FROM tags WHERE name=?", (name,))
        return cursor.fetchone()[0]

    def tag_document(self, doc_id: str, tag_ids: List[int]):
        """为文档打标签"""
        cursor = self.sqlite_conn.cursor()
        for tid in tag_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?, ?)",
                (doc_id, tid)
            )
        self.sqlite_conn.commit()

    # Chunk管理 (SQLite + ChromaDB双写)

    def save_chunk(self, chunk_id: str, text: str, doc_id: str, chunk_index: int = 0):
        """Chunk双写：SQLite存元数据 + ChromaDB存向量"""
        # 1. SQLite写入
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO chunks (chunk_id, doc_id, chunk_index, content, char_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (chunk_id, doc_id, chunk_index, text, len(text)))
        self.sqlite_conn.commit()

        # 2. ChromaDB写入（纯向量索引，metadata含source_file供评估使用）
        cursor.execute("SELECT file_name FROM documents WHERE doc_id=?", (doc_id,))
        row = cursor.fetchone()
        source_file = row[0] if row else ""
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"doc_id": doc_id, "chunk_index": chunk_index, "source_file": source_file}]
        )

    # QA管理

    def save_qa(self, qa_id: str, chunk_id: str, doc_id: str,
                question: str, answer: str,
                faith_score: float = 1.0, faith_reasoning: str = "",
                diversity_max_sim: float = 0.0, relevancy_score: float = 1.0,
                filter_status: str = "passed"):
        """保存QA对，含完整三层过滤元数据"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO qa_pairs
            (qa_id, chunk_id, doc_id, question, answer,
             faith_score, faith_reasoning, diversity_max_sim, relevancy_score, filter_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (qa_id, chunk_id, doc_id, question, answer,
              faith_score, faith_reasoning, diversity_max_sim, relevancy_score, filter_status))
        self.sqlite_conn.commit()

    def get_qa_by_doc(self, doc_id: str) -> List[Dict]:
        """按文档查询所有QA对"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            SELECT qa_id, question, answer, filter_status, faith_score, relevancy_score
            FROM qa_pairs WHERE doc_id=? ORDER BY created_at
        ''', (doc_id,))
        return [dict(zip([c[0] for c in cursor.description], r)) for r in cursor.fetchall()]

    # 审计日志

    def log_event(self, run_id: str, event_type: str, doc_id: str = "",
                  details: Optional[Dict] = None, status: str = "success"):
        """写入审计日志。doc_id为空时存NULL（不触发外键约束）。"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT INTO audit_log (run_id, event_type, doc_id, details, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, event_type, doc_id or None,
              json.dumps(details or {}, ensure_ascii=False), status))
        self.sqlite_conn.commit()

    def get_audit_trail(self, run_id: str = "") -> List[Dict]:
        """查询审计日志（可按run_id过滤）"""
        cursor = self.sqlite_conn.cursor()
        if run_id:
            cursor.execute(
                'SELECT * FROM audit_log WHERE run_id=? ORDER BY created_at', (run_id,))
        else:
            cursor.execute('SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 200')
        return [dict(zip([c[0] for c in cursor.description], r)) for r in cursor.fetchall()]

    # 向量检索 (ChromaDB)

    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义检索最相关的chunk"""
        query_emb = self.embedder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)

        if not results:
            return []

        ids_list = results.get("ids")
        if not ids_list or not ids_list[0]:
            return []

        ids: list = ids_list[0]

        docs_list = results.get("documents")
        documents: list = docs_list[0] if docs_list else []

        meta_list = results.get("metadatas")
        metadatas: list = meta_list[0] if meta_list else []

        dist_list = results.get("distances")
        distances: list = dist_list[0] if dist_list else []

        hits = []
        for i in range(len(ids)):
            hits.append({
                "chunk_id": ids[i],
                "content": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None,
            })
        return hits

    # 统计查询

    def get_stats(self) -> Dict:
        """获取全局统计"""
        cursor = self.sqlite_conn.cursor()
        stats = {}
        for table in ["documents", "chunks", "qa_pairs", "audit_log"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        cursor.execute("SELECT parse_status, COUNT(*) FROM documents GROUP BY parse_status")
        stats["doc_by_status"] = dict(cursor.fetchall())

        cursor.execute("SELECT filter_status, COUNT(*) FROM qa_pairs GROUP BY filter_status")
        stats["qa_by_filter"] = dict(cursor.fetchall())

        return stats

    # 检索辅助查询

    def get_all_chunks(self) -> List[Dict]:
        """获取所有chunk（用于构建BM25索引）"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            SELECT c.chunk_id, c.doc_id, c.content, c.chunk_index,
                   d.file_name as source_file
            FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
            ORDER BY c.doc_id, c.chunk_index
        ''')
        return [dict(zip([col[0] for col in cursor.description], r)) for r in cursor.fetchall()]

    def get_qa_by_chunk(self, chunk_id: str) -> List[Dict]:
        """获取某个chunk对应的所有QA对（仅返回对生成答案有价值的核心字段）"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            SELECT question, answer
            FROM qa_pairs WHERE chunk_id=? AND filter_status='passed'
        ''', (chunk_id,))
        # 此时组装的字典里只有 'question' 和 'answer'，绝对纯净
        return [dict(zip([col[0] for col in cursor.description], r)) for r in cursor.fetchall()]

    def get_documents_by_ids(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """批量获取文档元数据"""
        if not doc_ids:
            return {}
        placeholders = ",".join("?" * len(doc_ids))
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            f'SELECT doc_id, file_name, file_size_bytes, page_count FROM documents WHERE doc_id IN ({placeholders})',
            doc_ids
        )
        return {r[0]: {"file_name": r[1], "file_size_bytes": r[2], "page_count": r[3]}
                for r in cursor.fetchall()}

    # 用户管理

    def get_user(self, username: str) -> Optional[Dict]:
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(zip([c[0] for c in cursor.description], row))

    def create_user(self, username: str, password_hash: str, role: str = "user") -> int:
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role))
        self.sqlite_conn.commit()
        return cursor.lastrowid

    def ensure_admin(self):
        """确保存在默认管理员账号"""
        import hashlib
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        if cursor.fetchone()[0] == 0:
            default_pw = hashlib.sha256("doc_sys_salt:admin123".encode()).hexdigest()
            self.create_user("admin", default_pw, "admin")
            print("  [OK] 默认管理员账号已创建: admin / admin123")

    def close(self):
        self.sqlite_conn.close()
