# QA批量生成模块
import json
import re
from typing import List, Dict, Optional
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME, GENERATOR_TEMPERATURE


class QAGenerator:
    """批量QA生成器：将多个chunk合并后一次调用LLM生成问答对"""

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = LLM_MODEL_NAME
        self.temperature = GENERATOR_TEMPERATURE

    def generate_from_chunks(self, chunks: List[str]) -> List[Dict[str, str]]:
        """
        批量生成QA对。
        每个chunk被标记序号，LLM返回的JSON中需要包含chunk_index，
        以便回溯每条QA的来源chunk。

        Returns:
            [
                {"chunk_index": 0, "question": "...", "answer": "..."},
                ...
            ]
        """
        if not chunks:
            return []

        # 构建批量prompt
        chunks_text = self._build_batch_prompt(chunks)
        prompt = f"""你是一个专业的数据标注专家。请阅读以下 {len(chunks)} 个文本片段，为每个片段提取高质量的问答对。

要求：
1. 提问必须具有针对性，答案必须准确且【仅基于给定文本】，不要编造。
2. 尽可能覆盖每个片段中的核心概念、重要事实或逻辑推导。
3. 每个片段至少提取 1 对、最多提取 3 对问答。
4. 严格按照 JSON 数组格式返回，每条记录必须包含 chunk_index 标明来源片段。

返回格式（不要包含任何其他文字）：
[
    {{"chunk_index": 0, "question": "问题1", "answer": "答案1"}},
    {{"chunk_index": 0, "question": "问题2", "answer": "答案2"}},
    {{"chunk_index": 1, "question": "问题1", "answer": "答案1"}}
]

{chunks_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个严格遵循JSON格式输出的数据标注助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            raw_content = response.choices[0].message.content
            if raw_content is None:
                print("[WARN] LLM 返回内容为空，跳过此批次。")
                return []

            # 正则剥离markdown代码块标记（只剥首尾，不损坏内容）
            clean_content = self._strip_markdown_fence(raw_content)
            qa_pairs = json.loads(clean_content)
            return qa_pairs

        except json.JSONDecodeError:
            print("[ERROR] LLM 返回的不是标准 JSON 格式，跳过此批次。")
            return []
        except Exception as e:
            print(f"[ERROR] 调用 LLM 生成 QA 失败: {e}")
            return []

    def _build_batch_prompt(self, chunks: List[str]) -> str:
        """将多个chunk拼接为编号文本块"""
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[片段 {i}]\n{chunk}")
        return "\n\n".join(parts)

    def _strip_markdown_fence(self, text: str) -> str:
        """只剥离首尾的markdown代码块标记，不损坏JSON内容"""
        text = text.strip()
        # 去掉开头的 ```json或 ```
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # 去掉结尾的 ```
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()
