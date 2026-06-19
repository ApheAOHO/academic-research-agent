# agents.py
"""多 Agent 学术协作框架 - 不含自我评测 Agent"""
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from tools import execute_tool, TOOLS
import memory


class BaseAgent:
    """Agent 基类"""
    def __init__(self, name: str, llm_client: AsyncOpenAI, model: str = "deepseek-chat"):
        self.name = name
        self.client = llm_client
        self.model = model

    async def think(self, prompt: str, context: str = "") -> str:
        """调用 LLM 进行推理"""
        messages = [
            {"role": "system", "content": f"你是 {self.name}，一个专业的学术研究助手。{self._system_prompt()}"}
        ]
        if context:
            messages.append({"role": "user", "content": f"背景信息：\n{context}"})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _system_prompt(self) -> str:
        return ""  # 子类覆盖


class PlannerAgent(BaseAgent):
    """研究规划 Agent"""
    def _system_prompt(self) -> str:
        return "你的职责是将一个复杂的研究主题分解为 3~5 个可执行的子问题，并针对每个子问题给出建议的搜索关键词（添加学术限定词，如 review, paper, study）。输出 JSON 格式：{\"subtasks\": [{\"question\": \"...\", \"keywords\": [\"...\"]}]}"

    async def plan(self, topic: str, history_context: str = "") -> List[Dict]:
        prompt = f"研究主题：{topic}\n请分解为子任务。"
        if history_context:
            prompt += f"\n已有历史研究：{history_context}\n请避免重复已充分研究的子问题。"
        response = await self.think(prompt)
        # 提取 JSON
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response)
            return data.get("subtasks", [])
        except:
            # 降级：默认生成 3 个泛化子任务
            return [
                {"question": f"{topic} 的研究背景和现状", "keywords": [f"{topic} review", f"{topic} state of the art"]},
                {"question": f"{topic} 的主要方法论与技术", "keywords": [f"{topic} methodology", f"{topic} approach"]},
                {"question": f"{topic} 的最新进展与争议", "keywords": [f"{topic} recent advances", f"{topic} challenges"]}
            ]


class SearcherAgent(BaseAgent):
    """搜索与抓取 Agent"""
    def _system_prompt(self) -> str:
        return "你可以调用 google_ai_search 和 follow_link 工具。你的目标是针对给定的子问题，执行最多 3 次搜索和 2 次链接跟进，收集足够的信息。每次获得结果后调用 save_to_memory 存储。完成搜索后输出简短总结。不要输出无关内容。"
    
    async def research_subtask(self, subtask: Dict) -> str:
        question = subtask.get("question")
        keywords = subtask.get("keywords", [question])
        # 构建工具调用循环
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"子问题：{question}\n建议关键词：{', '.join(keywords)}\n请开始搜索研究。"}
        ]
        tool_call_count = 0
        max_tools = 5  # 最多调用工具次数
        
        while tool_call_count < max_tools:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,  # 复用原有工具定义
                temperature=0.2,
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_count += 1
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)
                    print(f"🔧 [{self.name}] 调用 {func_name}: {func_args}")
                    result = await execute_tool(func_name, func_args)
                    # 保存结果到记忆（自动）
                    memory.save_memory_entry(f"【{question}】\n{result}", "研究笔记")
                    # 追加到对话
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
                continue
            else:
                # 无工具调用，最终总结
                final_summary = msg.content or "研究完成，未生成总结。"
                memory.save_memory_entry(f"【子问题总结】{question}\n{final_summary}", "核心发现")
                return final_summary
        return "达到工具调用上限，停止搜索。"


class AnalyzerAgent(BaseAgent):
    """分析评估 Agent"""
    def _system_prompt(self) -> str:
        return "你是一个学术分析专家。阅读已有的研究记录（向量记忆中与主题相关的条目），提取关键观点，识别矛盾或缺失的信息，并给出是否需要进行补充搜索的建议。输出结构化的分析结果。"
    
    async def analyze(self, topic: str) -> Dict:
        # 检索与该主题相关的所有记忆（不限数量）
        memories = memory.search_similar_memories(topic, limit=10, rerank=True)
        if not memories:
            return {"insights": "无相关历史记忆", "gaps": ["需要先进行基础搜索"]}
        context = "\n".join([f"- {m['content'][:500]}" for m in memories])
        prompt = f"""研究主题：{topic}
以下是已有的研究记忆：
{context}

请分析：
1. 已经明确的核心发现（列出 3-5 点）
2. 当前存在的知识缺口或争议点（至少 2 点）
3. 是否需要额外搜索？如需，请给出建议的搜索查询（最多 3 条）

输出 JSON：{{"insights": "...", "gaps": ["..."], "need_more_search": true/false, "suggested_queries": ["..."]}}"""
        response = await self.think(prompt, context="")
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response)
        except:
            return {"insights": "分析失败", "gaps": [], "need_more_search": False, "suggested_queries": []}


class WriterAgent(BaseAgent):
    """报告撰写 Agent"""
    def _system_prompt(self) -> str:
        return "你是一个学术报告撰写专家。根据已有的研究记忆和分析结果，生成一份结构完整、引用规范的研究报告。报告应包含：标题、摘要、引言、分节讨论（每个子问题一节）、结论、参考文献。使用中文。"
    
    async def write_report(self, topic: str, analysis: Dict, planner_output: List[Dict]) -> str:
        # 收集所有相关记忆（不限量，但限制长度）
        memories = memory.search_similar_memories(topic, limit=15, rerank=True)
        memory_text = "\n".join([f"[相关性 {m.get('rerank_score', m.get('similarity', 0)):.2f}] {m['content'][:800]}" for m in memories])
        sub_tasks_text = "\n".join([f"- {item['question']}" for item in planner_output])
        prompt = f"""研究主题：{topic}
规划的子问题：
{sub_tasks_text}

分析结果：
{analysis.get('insights', '无')}
知识缺口：{', '.join(analysis.get('gaps', []))}

详细记忆条目：
{memory_text}

请撰写最终的学术研究报告。要求：
- 不少于 1000 字
- 引用具体内容时标注“来源：记忆条目”
- 明确回答每个子问题
- 最后给出未来研究方向
"""
        report = await self.think(prompt, context="")
        return report