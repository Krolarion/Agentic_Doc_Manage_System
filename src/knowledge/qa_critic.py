# QA质量蒸馏模块：三层过滤（每条QA附带质量分数元数据）
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME, CRITIC_TEMPERATURE
)


class FaithfulnessCritic:
    """第一层：LLM忠实度裁判 —— 答案是否100% 来源于原文"""

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = LLM_MODEL_NAME

    def evaluate(self, chunk: str, question: str, answer: str) -> Tuple[bool, str]:
        """
        Chain-of-Thought判别。
        返回: (is_faithful: bool, reasoning: str)
        """
        system_prompt = "你是一个严苛的数据质量评估专家。你的唯一任务是判断【答案】是否能完全由【参考文本】推导得出。"

        user_prompt = f"""请评估以下问答对的质量。

参考文本 (Chunk)：
{chunk}

待评估问题 (Question)：{question}
待评估答案 (Answer)：{answer}

评估规则：
1. 提取出【待评估答案】中的所有关键陈述和实体。
2. 逐一检查这些陈述是否在【参考文本】中有明确的文字支撑。
3. 如果答案中包含了任何【参考文本】未提及的信息（哪怕它是客观真理），也必须判定为"不忠实"(is_faithful: false)。
4. 如果答案是对参考文本的合理总结且没有引入新概念，判定为"忠实"(is_faithful: true)。

必须严格以 JSON 格式输出，不包含任何多余字符：
{{
    "reasoning": "你的逐步分析过程",
    "is_faithful": true或false
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=CRITIC_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            if raw_content is None:
                return False, "LLM返回空内容"

            result = json.loads(raw_content)
            return result.get("is_faithful", False), result.get("reasoning", "")

        except Exception as e:
            return False, f"评估异常: {e}"

    def filter(self, qa_pairs: List[Dict], chunks: List[str]) -> List[Dict]:
        """
        逐条评估，通过的QA附加faith_score + faith_reasoning。
        """
        passed = []
        for qa in qa_pairs:
            chunk_idx = qa.get("chunk_index", -1)
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                print(f"[WARN] chunk_index={chunk_idx} 越界，跳过此条。")
                continue

            chunk = chunks[chunk_idx]
            is_faithful, reasoning = self.evaluate(chunk, qa["question"], qa["answer"])

            if is_faithful:
                qa["faith_score"] = 1.0
                qa["faith_reasoning"] = reasoning
                passed.append(qa)
            else:
                print(f"[DROP-Faith] {qa['question'][:40]}... | {reasoning[:60]}")

        return passed


class DiversityFilter:
    """第二层：向量去重 —— 抑制语义重复的问题"""

    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.history_questions: List[str] = []
        self.history_embeddings: Optional[np.ndarray] = None

    def check_diversity(self, question: str, threshold: float = 0.85) -> Tuple[bool, float]:
        """
        检查问题多样性。
        返回: (is_diverse: bool, max_similarity: float)
        """
        q_emb = self.embedder.encode([question])

        if self.history_embeddings is None:
            self.history_embeddings = q_emb
            self.history_questions.append(question)
            return True, 0.0

        sims = cosine_similarity(q_emb, self.history_embeddings)[0]
        max_sim = float(np.max(sims))

        if max_sim < threshold:
            self.history_embeddings = np.vstack([self.history_embeddings, q_emb])
            self.history_questions.append(question)
            return True, max_sim
        else:
            return False, max_sim

    def filter(self, qa_pairs: List[Dict]) -> List[Dict]:
        """通过的QA附加diversity_max_sim"""
        passed = []
        for qa in qa_pairs:
            is_diverse, max_sim = self.check_diversity(qa["question"])
            if is_diverse:
                qa["diversity_max_sim"] = max_sim
                passed.append(qa)
            else:
                print(f"[DROP-Div] 重复度 {max_sim:.4f}: {qa['question'][:50]}...")
        return passed


class RelevancyFilter:
    """第三层：答案-原文相关性校验"""

    def __init__(self, embedder: Optional[SentenceTransformer] = None):
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL_NAME)

    def check_relevancy(self, chunk: str, answer: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        计算答案与源chunk的向量余弦相似度。
        返回: (is_relevant: bool, similarity_score: float)
        """
        embeddings = self.embedder.encode([chunk, answer])
        sim_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        return sim_score >= threshold, sim_score

    def filter(self, qa_pairs: List[Dict], chunks: List[str]) -> List[Dict]:
        """通过的QA附加relevancy_score"""
        passed = []
        for qa in qa_pairs:
            chunk_idx = qa.get("chunk_index", -1)
            if chunk_idx < 0 or chunk_idx >= len(chunks):
                continue

            chunk = chunks[chunk_idx]
            is_relevant, sim_score = self.check_relevancy(chunk, qa["answer"])

            if is_relevant:
                qa["relevancy_score"] = sim_score
                passed.append(qa)
            else:
                print(f"[DROP-Rel] 相似度 {sim_score:.4f}: {qa['question'][:50]}...")

        return passed
