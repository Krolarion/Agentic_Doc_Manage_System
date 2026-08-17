# LLM-as-Judge评分器：三维度评估系统输出质量
import json
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME


@dataclass
class JudgeScore:
    """单条评估分数（每维度0-100%）"""
    faithfulness: int = 0       # 忠实度：答案是否完全来源于检索到的文档
    accuracy: int = 0           # 准确率：答案事实是否正确
    relevance: int = 0          # 相关性：答案是否切题
    reasoning: str = ""         # 裁判的详细推理过程
    overall: float = 0.0        # 综合分 (忠实度×0.4 + 准确率×0.3 + 相关性×0.3)

    def to_dict(self) -> Dict:
        return {
            "faithfulness": self.faithfulness,
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "overall": round(self.overall, 1),
            "reasoning": self.reasoning,
        }


class LLMJudge:
    """
    LLM-as-Judge裁判模型。
    对系统生成回答从三个维度独立评分：
    - 忠实度 (Faithfulness): 答案是否完全基于检索到的上下文，有无幻觉
    - 准确率 (Accuracy): 答案中的事实是否正确
    - 相关性 (Relevance): 答案是否直接回应了用户的问题
    """

    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = LLM_MODEL_NAME

    def evaluate(self, query: str, answer: str, contexts: str) -> JudgeScore:
        """
        对单条系统回答进行三维度评分。
        contexts: 检索到的参考文档内容
        """
        prompt = f"""你是一个严格的文档系统质量评估专家。请对以下系统回答进行三维度评分（百分比制）。

【用户问题】
{query}

【检索到的参考上下文】
{contexts[:3000]}

【系统回答】
{answer[:2000]}

【评分标准】
重要提示：回答简短不代表低分。只要关键信息正确且有原文支撑，简短回答可获高分。

1. **忠实度** (0-100%)：回答中的事实是否能在上下文中找到支撑？有无编造？
   - 90-100%：所有陈述都有明确原文支撑，无编造。即使回答简短，只要数字/事实精确匹配即可
   - 70-89%：绝大部分有支撑，极少量合理推断
   - 50-69%：过半有支撑，但有明显超出原文的扩展
   - 0-49%：多数内容缺乏依据或编造

2. **准确率** (0-100%)：回答中的事实是否正确？
   - 90-100%：所有事实精确无误。简短的数字提取也算完全准确
   - 70-89%：绝大部分正确，仅有细微偏差
   - 50-69%：过半正确，但有明显错误
   - 0-49%：多数事实有误

3. **相关性** (0-100%)：回答是否直接回应了用户问题？
   - 90-100%：精准命中问题核心，直接给出了答案。回答可以简短
   - 70-89%：基本回应核心问题，略有不足
   - 50-69%：部分相关，不够聚焦或有冗余
   - 0-49%：不够聚焦、答非所问，或只输出工具调用

【输出格式】
严格按照 JSON 格式输出，不包含 markdown 标记：
{{
    "faithfulness": <0-100的整数>,
    "accuracy": <0-100的整数>,
    "relevance": <0-100的整数>,
    "reasoning": "逐维度分析：忠实度方面... 准确率方面... 相关性方面..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个严格遵循JSON格式输出的文档质量评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            if not raw:
                return JudgeScore(reasoning="LLM返回空内容")

            result = json.loads(raw)
            score = JudgeScore(
                faithfulness=int(result.get("faithfulness", 0)),
                accuracy=int(result.get("accuracy", 0)),
                relevance=int(result.get("relevance", 0)),
                reasoning=result.get("reasoning", ""),
            )
            # 综合分 = 忠实度*0.4 + 准确率*0.3 + 相关性*0.3
            score.overall = score.faithfulness * 0.4 + score.accuracy * 0.3 + score.relevance * 0.3
            return score

        except json.JSONDecodeError:
            return JudgeScore(reasoning=f"JSON解析失败: {raw[:200] if raw else 'None'}")
        except Exception as e:
            return JudgeScore(reasoning=f"评估异常: {e}")
