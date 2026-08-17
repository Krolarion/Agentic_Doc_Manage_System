import uuid
import re
import json
from typing import List, Dict, Any
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL_NAME, LOCAL_LLM_URL, LOCAL_LLM_MODEL
# from src.agent.tools import TOOL_SCHEMA, TOOL_MAP, init_tools


# SYSTEM_PROMPT = """你是一个企业文档智能助手，可以帮助用户在文档知识库中查找信息、回答问题。

# # ## 核心原则
# # 1. **所有回答必须严格基于检索到的文档内容，不要使用你自己的知识。**
# # 2. **【极其重要】如果知识库中找不到相关信息，必须诚实地说"知识库中未找到相关内容"，绝对不要编造。**
# # 3. **当检索结果不充分时，优先承认信息不足，而不是给出不完整的推测。**
# # 4. **如果检索结果只覆盖了问题的部分内容，明确告知用户哪些部分有答案，哪些没有。**

# # ## 工作流程
# # - 当你需要调用工具时，请严格使用提供的Function Calling格式。
# # - 先调用rewrite_query将用户口语改写为精准关键词，再用改写结果调search_knowledge
# # - 如果你看到了工具返回的结果，必须自己阅读理解后，提取核心结论回答用户。绝对禁止以“工具返回：”等形式原样复读原始数据！
# # - 初次检索不理想时，换用不同关键词再次检索，而不是直接编造
# # - 检索后先检查结果是否与问题相关，再决定是否回答


# # ## 回答格式
# # - 每条事实陈述后标注 [来源: 文件名]
# # - 用清晰的语言和逻辑进行回答，避免不必要的背景介绍
# 你是一个极其严谨的企业文档智能助手。你的唯一任务是：基于提供的【检索到的文档内容】，回答用户的【问题】。

# ## 🚨 绝对红线（防幻觉规则）
# 1. **零推理原则**：你的回答必须完全来自于文档中的明确陈述，禁止动用预训练知识。
# 2. **拒绝脑补**：如果文档中没有明确包含具体的数字、金额、周期、层级等事实，绝对不允许你自己计算、推测或捏造。
# 3. **强制弃权**：当文档内容与问题无关，或者缺乏直接得出答案的关键信息时，你必须一字不差地回答：“知识库中未找到相关内容。”
# 4. **格式禁令**：禁止输出任何JSON、代码块或以“工具返回：”开头的原始结构。

# ## 💡 行为示范 (必须严格学习以下案例的处理方式)

# 【案例1 - 找到了明确答案】
# 文档内容：[来源: 预算书.pdf] 本项目的硬件预算为50万元，软件授权预算为20万元。
# 用户问题：硬件和软件的总费用是多少？
# 你的回答：硬件和软件的总费用为70万元。[来源: 预算书.pdf]

# 【案例2 - 缺乏关键数字（触发强制弃权）】
# 文档内容：[来源: 规划书.pdf] 本项目的硬件预算占总投资的一小部分，软件预算正在审批中。
# 用户问题：硬件和云资源预算为多少万元？
# 你的回答：知识库中未找到相关内容。

# 【案例3 - 内容完全无关（触发强制弃权）】
# 文档内容：[来源: 员工手册.pdf] 公司规定员工每年享有5天带薪年假。
# 用户问题：第二阶段“试点验证”的周期是多少？
# 你的回答：知识库中未找到相关内容。
# """
SYSTEM_PROMPT = """你是一个极其严谨的企业文档智能助手。你的唯一任务是：基于提供的【检索到的文档内容】，回答用户的【问题】。

## 🚨 绝对红线（防幻觉规则）
1. **零推测原则**：回答必须完全来自于文档中的明确陈述。如果原文没有，你必须回答“知识库中未找到相关内容。”，绝对禁止动用预训练知识进行任何编造！
2. **强制查验**：如果用户问题中指定了某个具体文档（例如“合同_18”），你必须在检索内容的 [来源: xxx] 中寻找对应标识。如果上下文完全不包含该文档，必须强制弃权。
3. **格式约束**：必须先在 <thinking> 标签中逐步分析，最后在 <answer> 标签中输出结论。

## 💡 行为示范 (必须严格遵守此 XML 格式)

【案例 1 - 找到了明确答案】
文档内容：[来源: 预算书_01.pdf] 本项目的硬件预算为 50 万元，软件授权预算为 20 万元。
用户问题：在预算书_01中，硬件和软件的总费用是多少？
你的回答：
<thinking>
1. 来源校验：问题指定“预算书_01”，上下文存在 [来源: 预算书_01.pdf]，匹配成功。
2. 证据定位：硬件 50 万元，软件 20 万元。
3. 逻辑推演：50 + 20 = 70 万元。
</thinking>
<answer>
硬件和软件的总费用为 70 万元。[来源: 预算书_01.pdf]
</answer>

【案例 2 - 来源不匹配（触发强制弃权）】
文档内容：[来源: 合同_08.pdf] 质保期后维护费用为每年 15%。
用户问题：在合同_18中，质保期后的维护费用是多少？
你的回答：
<thinking>
1. 来源校验：问题要求“合同_18”，但上下文只有“合同_08.pdf”。来源不匹配！
2. 逻辑推演：检索到的文档与问题要求的文档不一致，无法回答，必须弃权。
</thinking>
<answer>
知识库中未找到相关内容。
</answer>

【案例 3 - 缺乏关键事实（触发强制弃权）】
文档内容：[来源: 规划书_05.pdf] 本项目的硬件预算占总投资的一小部分。
用户问题：在规划书_05中，硬件预算为多少万元？
你的回答：
<thinking>
1. 来源校验：问题指定“规划书_05”，上下文匹配成功。
2. 证据定位：原文只提到“一小部分”，没有具体金额数字。
3. 逻辑推演：缺乏关键数字，不能脑补，必须弃权。
</thinking>
<answer>
知识库中未找到相关内容。
</answer>
"""


class DocumentAgent:
    """文档智能Agent：确定性RAG管道（改写 → 检索 → 生成）"""

    def __init__(self, backend: str = "deepseek"):
        """
        backend: "deepseek" (API) 或 "local" (本地Qwen)
        """
        if backend == "local":
            self.client = OpenAI(api_key="local", base_url=LOCAL_LLM_URL)
            self.model = LOCAL_LLM_MODEL
        else:
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            self.model = LLM_MODEL_NAME
        self.messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 注入LLM client到工具层（供rewrite_query等工具使用）
        import src.agent.tools as agent_tools
        agent_tools._client = self.client

    def reset(self):
        """重置对话历史（保留system prompt）"""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        确定性RAG管道：强制执行改写→检索→生成的固定序列。
        """
        self.messages.append({"role": "user", "content": user_input})
        
        if verbose:
            print(f"\n{'─' * 40}")
            print(f"[RAG Pipeline] 开始处理查询: {user_input}")

        # 步骤1: 强制执行Query Rewrite
        try:
            from src.agent.tools import rewrite_query
            optimized_query = rewrite_query(user_input)
            if verbose:
                print(f"  [1. Query Rewrite] 原始: {user_input}  ->  改写: {optimized_query}")
        except Exception as e:
            optimized_query = user_input
            if verbose:
                print(f"  [1. Query Rewrite 异常，使用原句]: {e}")

        # 步骤2: 强制执行知识库检索
        try:
            from src.agent.tools import search_knowledge
            search_context = search_knowledge(optimized_query, top_k=3)
            
            # 拦截空，防止上游返回None导致属性错误
            if not search_context or search_context.strip() == "":
                search_context = "未找到相关内容。"
                if verbose:
                    print("  [2. Search] 未检索到有效内容。")
            else:
                if verbose:
                    print(f"  [2. Search] 成功召回上下文 (长度: {len(search_context)} 字)")
        except Exception as e:
            search_context = "检索过程发生异常。"
            if verbose:
                print(f"  [2. Search 异常]: {e}")

        # 步骤3: 构造强约束的Prompt，强制LLM进行阅读理解
#         rag_prompt = f"""请严格基于以下【检索到的文档内容】来回答用户的【问题】。

# 【要求】
# 1. 如果文档内容能回答问题，请用自然语言总结并回答，并在句末标注 [来源: 文件名]。绝对禁止以“工具返回：”等形式原样复读原始数据！
# 2. 如果缺乏直接得出答案的关键信息（尤其是具体金额、周期等），必须直接回答“知识库中未找到相关内容”，绝对不要编造。
# 3. 绝对不要输出任何JSON、代码或工具调用指令。

# 【检索到的文档内容】
# {search_context}

# 【问题】
# {user_input}
# """
        rag_prompt = f"""请严格按照系统提示词的规范，先在 <thinking> 标签内思考（核对文档来源和关键事实），然后在 <answer> 标签内给出最终回答。
如果缺乏信息或文档来源不匹配，<answer> 内只能写：“知识库中未找到相关内容。”

【检索到的文档内容】
{search_context}

【问题】
{user_input}
"""
        
        temp_messages = [
            {"role": "system", "content": self.messages[0]["content"]},
            {"role": "user", "content": rag_prompt}
        ]

        if verbose:
            print("  [3. Generation] 大模型开始生成回答 (包含思维链)...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=temp_messages,
                temperature=0.1,  
            )
            
            if not response or not response.choices or not response.choices[0].message.content:
                answer = "大模型生成引擎返回了空值。"
            else:
                raw_output = response.choices[0].message.content.strip()
                
                # 透视镜：打印内部思考过程 (纯字符串提取，绝对不报错)
                if verbose and "<thinking>" in raw_output and "</thinking>" in raw_output:
                    thinking_content = raw_output.split("<thinking>")[1].split("</thinking>")[0].strip()
                    print(f"\n  [推理过程]:\n{thinking_content}\n")
                
                # 代码拦截：安全提取最终答案 (纯字符串提取)
                if "<answer>" in raw_output and "</answer>" in raw_output:
                    answer = raw_output.split("<answer>")[1].split("</answer>")[0].strip()
                elif "<answer>" in raw_output: 
                    # 应对模型抽风只有开头没结尾的情况
                    answer = raw_output.split("<answer>")[1].strip()
                else:
                    # 彻底失控，强制启动安全底线
                    answer = "知识库中未找到相关内容。"

        except Exception as e:
            answer = f"[ERROR] 生成回答失败: {e}"

        self.messages.append({"role": "assistant", "content": answer})
        
        if verbose:
            print(f"[RAG Pipeline] 最终回答完成 ({len(answer)}字)\n")
            
        return answer
        # """
        # Agentic RAG：ReAct推理循环。
        # """
        # self.messages.append({"role": "user", "content": user_input})

        # max_rounds = 8
        # for turn in range(max_rounds):
        #     if verbose:
        #         print(f"[Turn {turn + 1}] 思考中")

        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=self.messages,
        #         tools=TOOL_SCHEMA,
        #         tool_choice="auto",
        #         temperature=0.3,
        #     )

        #     msg = response.choices[0].message
        #     content = (msg.content or "").strip()

        #     # 防御性解析：无差别暴力提取JSON
        #     if not msg.tool_calls:
        #         tc_str = ""
                
        #         # 寻找内容中第一个 '{' 和最后一个 '}'
        #         start_idx = content.find('{')
        #         end_idx = content.rfind('}')
                
        #         if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        #             possible_json = content[start_idx:end_idx+1]
        #             # 确认它包含工具调用的特征词
        #             if '"name"' in possible_json and '"arguments"' in possible_json:
        #                 tc_str = possible_json

        #         # 将提取到的字符串安全转换为合法工具调用
        #         if tc_str:
        #             try:
        #                 tc_json = json.loads(tc_str)
        #                 if isinstance(tc_json, dict) and "name" in tc_json:
        #                     call_id = f"call_fix_{uuid.uuid4().hex[:8]}"
                            
        #                     class AttrDict:
        #                         def __init__(self, **kwargs):
        #                             self.__dict__.update(kwargs)
                                    
        #                     # 确保JSON序列化正确，防止双重序列化导致解析失败
        #                     args_val = tc_json.get("arguments", {})
        #                     args_str = json.dumps(args_val, ensure_ascii=False) if isinstance(args_val, dict) else str(args_val)
                                
        #                     fake_func = AttrDict(name=tc_json["name"], arguments=args_str)
        #                     fake_tc = AttrDict(id=call_id, type="function", function=fake_func)
                            
        #                     msg.tool_calls = [fake_tc]
        #                     msg.content = ""  # 清空原有的裸JSON内容，防止污染上下文
        #                     content = ""      # 同步清空局部变量
        #             except json.JSONDecodeError:
        #                 pass
                
        #     # 防御性拦截：执行严格的指令处理，禁止模型原样复读原始数据
        #     if not msg.tool_calls and content:
        #         if content.startswith("工具返回:"):
        #             if verbose:
        #                 print(f"  [拦截] 检测到模型复读工具结果，强制要求总结。")
        #             self.messages.append({
        #                 "role": "user", 
        #                 "content": "请根据刚才的检索结果，用自然语言为我进行总结和解答。禁止直接复制粘贴“工具返回”的原始内容。"
        #             })
        #             continue  # 强制拦截，进入下一轮思考
                    
        #         # 正常的自然语言回答
        #         self.messages.append({"role": "assistant", "content": content})
        #         if verbose:
        #             print(f"[Turn {turn + 1}] 最终回答 ({len(content)}字)")
        #         return content

        #     # 有工具调用 → 记录执行 → 继续观察
        #     self.messages.append({
        #         "role": "assistant",
        #         "content": msg.content or "",
        #         "tool_calls": [
        #             {
        #                 "id": tc.id,
        #                 "type": "function",
        #                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}
        #             }
        #             for tc in (msg.tool_calls or [])
        #         ]
        #     })

        #     for tc in (msg.tool_calls or []):
        #         tool_name = tc.function.name
        #         tool_args = json.loads(tc.function.arguments)
        #         tool_result = self._execute_tool(tool_name, tool_args)

        #         if verbose:
        #             args_summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in tool_args.items())
        #             result_len = len(tool_result)
        #             print(f"  [{tool_name}] {args_summary}")
        #             print(f"  [{tool_name}] → {result_len}字")
                    
        #             # 💡 【新增】如果调用的工具是search_knowledge，则打印出完整的检索上下文
        #             if tool_name == "search_knowledge":
        #                 print("\n 喂给大模型的完整Context")
        #                 print(tool_result)
        #                 print("\n")

        #         self.messages.append({
        #             "role": "tool",
        #             "tool_call_id": tc.id,
        #             "content": tool_result,
        #         })

        # # 达到最大轮次 → 强制生成回答
        # if verbose:
        #     print(f"\n[!] 达到最大轮次 {max_rounds}，强制总结...")
        # return self._force_answer()

    # def _execute_tool(self, name: str, args: Dict) -> str:
    #     """执行工具调用，返回结果字符串"""
    #     func = TOOL_MAP.get(name)
    #     if func is None:
    #         return f"[ERROR] 未知工具: {name}"

    #     try:
    #         result = func(**args)
    #         return str(result)
    #     except Exception as e:
    #         return f"[ERROR] 工具执行失败: {e}"

    # def _force_answer(self) -> str:
    #     """强制LLM基于历史生成最终回答"""
    #     self.messages.append({
    #         "role": "user",
    #         "content": "基于上述检索结果，请用中文给出最终答案，必须标注信息来源。如果知识库中没有相关信息，请如实告知。"
    #     })
    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=self.messages,
    #         temperature=0.3,
    #     )
    #     answer = response.choices[0].message.content or ""
    #     self.messages.append({"role": "assistant", "content": answer})
    #     return answer
