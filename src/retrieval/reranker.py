# 重排序模块：Cross-Encoder精排
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL_NAME


class Reranker:
    """
    Cross-Encoder重排序器。
    与Bi-Encoder（Embedding）不同，Cross-Encoder将 (query, document)
    成对输入，输出精确的相关性分数，大幅提升Top-K精度。
    代价是推理速度较慢，因此仅用于召回后的精排阶段。
    """

    def __init__(self, model_name: str = ""):
        model = model_name or RERANKER_MODEL_NAME
        print(f"  加载 Reranker 模型: {model}...")
        # 本地路径不连HF
        is_local = not model.startswith("BAAI/") and "/" not in model.split(":")[0]
        self.model = CrossEncoder(model, trust_remote_code=True, local_files_only=is_local)
        self.model_name = model

    def rerank(self, query: str, documents: List[str],
               top_k: int = 5) -> List[Tuple[int, float]]:
        """
        对候选文档列表精排。
        返回: [(原始索引, 相关性分数), ...] 按分数降序，最多top_k条
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        raw = self.model.predict(pairs, show_progress_bar=False)

        # predict返回numpy array / list / tensor，统一转float list
        try:
            arr = __import__('numpy').asarray(raw).flatten()
            flat_scores = [float(v) for v in arr]
        except Exception:
            flat_scores = [0.0] * len(documents)

        ranked = sorted(enumerate(flat_scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_results(self, query: str, contents: List[str],
                       top_k: int = 5) -> List[int]:
        """
        便捷方法：返回重排后的内容索引列表（按分数降序）。
        """
        ranked = self.rerank(query, contents, top_k=top_k)
        return [idx for idx, _ in ranked]
