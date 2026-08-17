# 混合检索引擎：向量语义 + BM25关键词 + RRF融合 + Reranker精排 + 元数据增强
import jieba
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json

from src.storage.database_manager import DatabaseManager
from src.retrieval.reranker import Reranker


@dataclass
class SearchResult:
    """单条检索结果"""
    chunk_id: str
    content: str
    doc_id: str
    source_file: str
    chunk_index: int
    score: float                          # RRF融合分数
    vector_rank: Optional[int] = None     # 向量检索排名 (1-based)
    bm25_rank: Optional[int] = None       # BM25排名 (1-based)
    qa_pairs: List[Dict] = field(default_factory=list)  # 关联的QA对

    def to_context(self) -> str:
        """将检索结果组装为LLM可用的上下文文本"""
        parts = [f"[来源: {self.source_file}]\n{self.content}"]
        if self.qa_pairs:
            parts.append("\n关联问答:")
            for qa in self.qa_pairs[:3]:
                parts.append(f"  Q: {qa['question']}\n  A: {qa['answer']}")
        return "\n".join(parts)


class SimpleBM25:
    """轻量级BM25实现，零外部依赖"""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus          # tokenized documents
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.df: Dict[str, int] = {}  # document frequency
        self._compute_idf()

    def _compute_idf(self):
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    self.df[term] = self.df.get(term, 0) + 1
                    seen.add(term)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return float(np.log((self.N - n + 0.5) / (n + 0.5) + 1.0))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float64)
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = self._idf(term)
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                doc_len = len(doc)
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                scores[i] += score
        return scores

    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """返回 [(chunk_index, score), ...] 按分数降序"""
        scores = self.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridSearchEngine:
    """混合检索：向量 + BM25 → RRF融合 → Reranker精排 → 元数据增强"""

    def __init__(self, db: DatabaseManager, enable_rerank: bool = True):
        self.db = db
        self.embedder = db.embedder
        self.chunks_index: List[Dict] = []   # chunk_id → {doc_id, content, ...}
        self.bm25: Optional[SimpleBM25] = None
        self.reranker: Optional[Reranker] = None
        self.enable_rerank = enable_rerank
        self._build_bm25_index()
        if enable_rerank:
            self.reranker = Reranker()

    def _tokenize(self, text: str) -> List[str]:
        stop_words = {"的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "在", "对"}
        return [t for t in jieba.cut(text) if t.strip() and t.strip() not in stop_words]

    def _build_bm25_index(self):
        """从SQLite加载全部chunk，构建BM25索引"""
        print("  构建 BM25 关键词索引...")
        chunks = self.db.get_all_chunks()
        if not chunks:
            print("  [WARN] 数据库中没有 chunk，BM25 索引为空，请先运行 pipeline")
            self.chunks_index = []
            return

        self.chunks_index = chunks
        
        # 💡 核心修复1：将元数据（文件名）作为前缀拼接入BM25索引，打破“元数据致盲”
        tokenized = []
        for c in chunks:
            searchable_text = f"文档来源名称：{c.get('source_file', '')} \n {c.get('content', '')}"
            tokenized.append(self._tokenize(searchable_text))
            
        self.bm25 = SimpleBM25(tokenized)
        print(f"  [OK] BM25 索引就绪: {len(chunks)} 个 chunk, "
              f"词汇量 {len(self.bm25.df)}")

    def refresh_index(self):
        """重新构建索引（pipeline新增文档后调用）"""
        self._build_bm25_index()

    def search(self, query: str, top_k: int = 5,
               doc_ids: Optional[List[str]] = None,
               enable_rerank: Optional[bool] = None) -> List[SearchResult]:
        """
        混合检索主入口。
        """
        use_rerank = enable_rerank if enable_rerank is not None else self.enable_rerank

        # 💡 核心修复2：加深召回漏斗，确保底层引擎能把候选人带上来
        rerank_candidates = 60 if use_rerank else 0
        fetch_k = max(top_k * 15, 100, rerank_candidates)

        # 1. 向量语义检索
        vector_hits = self.db.search_chunks(query, fetch_k)

        # 2. BM25关键词检索
        bm25_hits: List[Tuple[int, float]] = []
        if self.bm25 and self.chunks_index:
            query_tokens = self._tokenize(query)
            bm25_hits = self.bm25.search(query_tokens, fetch_k)

        # 3. RRF融合
        fused = self._rrf_fusion(vector_hits, bm25_hits, k=60)

        # 4. 元数据过滤 + 构建候选集
        candidates: List[Dict] = []
        for chunk_id, rrf_score in fused:
            chunk_info = self._find_chunk(chunk_id)
            if chunk_info is None:
                continue
            if doc_ids and chunk_info.get("doc_id") not in doc_ids:
                continue

            vector_rank = self._get_rank(chunk_id, vector_hits)
            bm25_rank = self._get_bm25_rank(chunk_id, bm25_hits)
            qa_pairs = self.db.get_qa_by_chunk(chunk_id)

            candidates.append({
                "chunk_id": chunk_id,
                "chunk_info": chunk_info,
                "rrf_score": rrf_score,
                "vector_rank": vector_rank,
                "bm25_rank": bm25_rank,
                "qa_pairs": qa_pairs,
            })

            if len(candidates) >= rerank_candidates and not use_rerank:
                pass

        # 截断候选集
        if use_rerank and self.reranker:
            candidates = candidates[:rerank_candidates]

        # 5. Reranker精排
        if use_rerank and self.reranker and len(candidates) >= 2:
            # 💡 核心修复3：将文件名传给Reranker，防止它因为找不到代号而误杀正确切片
            contents = []
            for c in candidates:
                src_file = c["chunk_info"].get("source_file", "")
                txt = c["chunk_info"].get("content", "")
                contents.append(f"【文档来源：{src_file}】\n{txt}")
                
            ranked = self.reranker.rerank(query, contents, top_k=top_k)

            # 按精排分数重排
            reranked_candidates = []
            for idx, rerank_score in ranked:
                c = candidates[idx].copy()
                c["rerank_score"] = rerank_score
                reranked_candidates.append(c)
            candidates = reranked_candidates
        else:
            candidates = candidates[:top_k]

        # 6. 组装最终结果
        results = []
        def _clean_content(raw_text: str) -> str:
            text = raw_text.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    data = json.loads(text)
                    return data.get("text", text)
                except json.JSONDecodeError:
                    return text
            return text

        for c in candidates:
            ci = c["chunk_info"]
            score = c.get("rerank_score", c["rrf_score"])
            chunk_id = c["chunk_id"]
            
            raw_content = ci.get("content", "")
            
            if "_c" in chunk_id:
                parent_id = chunk_id.rsplit("_c", 1)[0]
                parent_info = self._find_chunk(parent_id)
                if parent_info:
                    raw_content = parent_info.get("content", raw_content)
            
            clean_content = _clean_content(raw_content)

            results.append(SearchResult(
                chunk_id=chunk_id,
                content=clean_content,
                doc_id=ci.get("doc_id", ""),
                source_file=ci.get("source_file", ""),
                chunk_index=ci.get("chunk_index", 0),
                score=score,
                vector_rank=c["vector_rank"],
                bm25_rank=c["bm25_rank"],
                qa_pairs=c["qa_pairs"],
            ))

        return results

    # 内部方法

    def _find_chunk(self, chunk_id: str) -> Optional[Dict]:
        for c in self.chunks_index:
            if c["chunk_id"] == chunk_id:
                return c
        return None

    def _get_rank(self, chunk_id: str, hits: List[Dict]) -> Optional[int]:
        for i, h in enumerate(hits):
            if h["chunk_id"] == chunk_id:
                return i + 1
        return None

    def _get_bm25_rank(self, chunk_id: str, bm25_hits: List[Tuple[int, float]]) -> Optional[int]:
        for rank, (idx, _) in enumerate(bm25_hits):
            if idx < len(self.chunks_index) and self.chunks_index[idx]["chunk_id"] == chunk_id:
                return rank + 1
        return None

    def _rrf_fusion(self, vector_hits: List[Dict],
                    bm25_hits: List[Tuple[int, float]], k: int = 60) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}

        for rank, hit in enumerate(vector_hits):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, (idx, bm25_score) in enumerate(bm25_hits):
            if bm25_score <= 0:
                continue
            chunk_id = self.chunks_index[idx]["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_search_engine(db: DatabaseManager) -> HybridSearchEngine:
    return HybridSearchEngine(db)