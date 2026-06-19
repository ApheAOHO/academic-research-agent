# core.py
"""学术研究 Agent 核心引擎 - 支持历史记忆检索与链接深度抓取"""
import json
import asyncio
import random
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI

from tools import TOOLS, execute_tool
import memory

# 尝试导入多 Agent 组件（若 agents.py 不存在则降级）
try:
    from agents import PlannerAgent, SearcherAgent, AnalyzerAgent, WriterAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    print("⚠️ 未找到 agents.py，多 Agent 模式不可用。如需使用请创建 agents.py。")

# 导入评测模块（仅多 Agent 模式使用）
try:
    from evaluator import evaluate_full
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("⚠️ 未找到 evaluator.py，多 Agent 模式的自我评测将不可用。")

# ==================== 原有单 Agent 模式（不启用评测） ====================
class ResearchSession:
    """一次学术研究会话的核心引擎（单 Agent 模式，不进行自我评测）"""

    def __init__(
        self,
        topic: str,
        llm_client: AsyncOpenAI,
        model: str = "deepseek-chat",
        max_searches: int = 5,
        min_searches: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        enable_history: bool = True,
        history_limit: int = 3,
        enable_self_evaluation: bool = False,   # 单 Agent 默认不评测
    ):
        self.topic = topic
        self.client = llm_client
        self.model = model
        self.max_searches = max_searches
        self.min_searches = min_searches
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_history = enable_history
        self.history_limit = history_limit
        self.enable_self_evaluation = enable_self_evaluation   # 但本类中不会调用评测

        self.messages: List[Dict[str, Any]] = []
        self.tool_call_count = 0
        self.search_count = 0          # 统计 google_ai_search 次数
        self.follow_link_count = 0     # 统计 follow_link 次数
        self.final_report: Optional[str] = None

        memory.start_new_conversation(topic)

    def _build_system_prompt(self) -> str:
        base_prompt = (
            "你是一个**学术研究 Agent**。你的目标是基于可靠的学术资源为用户提供深入的研究报告。\n"
            "你可以使用以下工具：\n"
            "- google_ai_search：进行初始搜索，获取概览和关键链接。\n"
            "- follow_link：从搜索结果中获取一个学术链接（例如论文 PDF、期刊页面），打开并提取该链接内的详细内容。\n"
            "- save_to_memory：保存重要发现。\n"
            "推荐工作流程：\n"
            "1. 调用 google_ai_search 获取相关概览和一组来源链接。\n"
            "2. 分析结果，选择最有价值的 1-3 个链接，对每个链接调用 follow_link 以获取详细内容。\n"
            "3. 每次获得信息后调用 save_to_memory 保存。\n"
            f"总工具调用次数至少 {self.min_searches} 次，最多 {self.max_searches} 次。\n"
            "完成足够搜索后，输出**最终学术研究报告**，应包含：研究背景、主要发现、深度分析（引用链接中的具体内容）、结论。\n"
            "始终用中文回复。"
        )
        if self.enable_history:
            base_prompt += (
                "\n\n**历史记忆**：你会收到之前研究的相关内容，请避免重复搜索已知信息。"
            )
        return base_prompt

    async def _retrieve_relevant_history(self) -> str:
        if not self.enable_history:
            return ""
        print(f"🔍 正在检索与 '{self.topic}' 相关的历史学术记忆...")
        history_context = memory.get_relevant_history(self.topic, limit=self.history_limit)
        if history_context:
            print("📚 找到相关历史记忆，将用于指导本次研究")
        else:
            print("📭 未找到相关历史记忆，将进行全新研究")
        return history_context

    async def run(self) -> str:
        """执行整个学术研究会话，返回最终报告内容"""
        history_context = await self._retrieve_relevant_history()

        if history_context:
            initial_user_message = f"""研究主题：{self.topic}

{history_context}

【任务说明】
请基于以上历史研究成果（如有），继续深入研究当前主题。注意：
1. 使用 google_ai_search 进行初步搜索。
2. 从搜索结果中选择学术链接，调用 follow_link 获取详细内容。
3. 每次获得信息后调用 save_to_memory 保存。
4. 完成后输出综合学术研究报告。

【严格要求】请不要输出任何计划或介绍文字，直接开始调用工具。"""
        else:
            initial_user_message = f"{self.topic}\n\n【严格要求】请不要输出任何计划或介绍文字，直接开始调用工具。"

        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": initial_user_message}
        ]

        while True:
            try:
                use_tools = TOOLS if self.tool_call_count < self.max_searches else None

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=use_tools,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                msg = response.choices[0].message

                if use_tools is not None and msg.tool_calls:
                    self.tool_call_count += len(msg.tool_calls)
                    self.messages.append(msg)
                    results = await self._execute_sequential_tools(msg.tool_calls)
                    for res in results:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": res["tool_call_id"],
                            "content": res["content"],
                        })
                    continue

                if msg.content:
                    if self.tool_call_count < self.min_searches and use_tools is not None:
                        print(f"⚠️ 模型尝试提前输出，当前仅 {self.tool_call_count} 次搜索，强制继续")
                        self.messages.append(msg)
                        self.messages.append({
                            "role": "user",
                            "content": f"你只进行了 {self.tool_call_count} 次搜索，请继续调用工具至少达到 {self.min_searches} 次。"
                        })
                        continue
                    else:
                        self.final_report = msg.content
                        self.messages.append(msg)
                        memory.finalize_conversation(self.final_report)
                        memory.save_to_vector_memory(
                            content=f"【学术研究报告】\n{self.final_report[:2000]}",
                            category="研究报告",
                            metadata={"topic": self.topic, "is_final_report": True}
                        )
                        # 单 Agent 模式不进行 Precision/Recall 评测（按需求）
                        return self.final_report

            except Exception as e:
                error_msg = f"研究过程中发生错误：{e}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                memory.finalize_conversation(error_msg)
                raise

    async def _execute_sequential_tools(self, tool_calls):
        """顺序执行工具调用，避免并发反爬，并统计搜索和抓取次数"""
        results = []
        for idx, tc in enumerate(tool_calls):
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)
            print(f"🔧 调用工具 ({idx+1}/{len(tool_calls)}): {func_name}({func_args})")
            # 统计
            if func_name == "google_ai_search":
                self.search_count += 1
            elif func_name == "follow_link":
                self.follow_link_count += 1
            result = await execute_tool(func_name, func_args)
            display = result[:500] + "..." if len(result) > 500 else result
            print(f"📋 返回: {display}")
            results.append({"tool_call_id": tc.id, "content": result})
            if idx < len(tool_calls) - 1:
                delay = random.uniform(10, 20)
                print(f"⏳ 等待 {delay:.1f} 秒后执行下一个工具...")
                await asyncio.sleep(delay)
        return results


# ==================== 多 Agent 协作模式（默认启用评测） ====================
class MultiAgentSession:
    """多 Agent 协作的学术研究会话，默认启用 Precision/Recall 评测"""

    def __init__(
        self,
        topic: str,
        llm_client: AsyncOpenAI,
        model: str = "deepseek-chat",
        enable_history: bool = True,
        history_limit: int = 3,
        enable_self_evaluation: bool = True,   # 多 Agent 默认启用
    ):
        if not AGENTS_AVAILABLE:
            raise ImportError("多 Agent 模式需要 agents.py 文件，请创建后再试。")
        self.topic = topic
        self.client = llm_client
        self.model = model
        self.enable_history = enable_history
        self.history_limit = history_limit
        self.enable_self_evaluation = enable_self_evaluation   # 默认 True
        
        self.planner = PlannerAgent("Planner", llm_client, model)
        self.searcher = SearcherAgent("Searcher", llm_client, model)
        self.analyzer = AnalyzerAgent("Analyzer", llm_client, model)
        self.writer = WriterAgent("Writer", llm_client, model)
        
        memory.start_new_conversation(topic)
    
    async def run(self) -> str:
        """执行多 Agent 协作研究，返回最终报告，并启用评测"""
        # 1. 检索历史
        history_context = ""
        if self.enable_history:
            history_context = memory.get_relevant_history(self.topic, limit=self.history_limit)
            print("📚 历史记忆检索完成")
        
        # 2. Planner 制定计划
        print("🧠 Planner Agent 正在制定研究计划...")
        plan = await self.planner.plan(self.topic, history_context)
        if not plan:
            plan = [{"question": self.topic, "keywords": [self.topic]}]
        print(f"📋 研究计划：{len(plan)} 个子任务")
        for idx, item in enumerate(plan, 1):
            print(f"   {idx}. {item.get('question')}")
        
        # 3. Searcher 依次执行每个子任务
        print("\n🔍 Searcher Agent 开始执行研究...")
        for subtask in plan:
            print(f"\n--- 处理子任务: {subtask.get('question')} ---")
            await self.searcher.research_subtask(subtask)
            await asyncio.sleep(2)
        
        # 4. Analyzer 评估信息完整性
        print("\n📊 Analyzer Agent 正在评估信息完整性...")
        analysis = await self.analyzer.analyze(self.topic)
        if analysis.get("need_more_search") and analysis.get("suggested_queries"):
            print(f"⚠️ 信息不足，进行补充搜索: {analysis['suggested_queries']}")
            extra_subtask = {
                "question": "补充搜索：" + "；".join(analysis["suggested_queries"]),
                "keywords": analysis["suggested_queries"]
            }
            await self.searcher.research_subtask(extra_subtask)
        
        # 5. Writer 生成最终报告
        print("\n✍️ Writer Agent 正在撰写研究报告...")
        report = await self.writer.write_report(self.topic, analysis, plan)
        
        # 6. 保存最终报告
        memory.finalize_conversation(report)
        memory.save_to_vector_memory(
            content=f"【学术研究报告】\n{report[:2000]}",
            category="研究报告",
            metadata={"topic": self.topic, "is_final_report": True}
        )
        
        # 7. 启用 Precision/Recall 评测（如果 enable_self_evaluation 为 True）
        if self.enable_self_evaluation and EVALUATOR_AVAILABLE:
            try:
                # 获取当前会话的参考材料（所有研究笔记）
                reference = memory.get_current_session_reference()
                if reference:
                    print("📊 正在进行 Precision/Recall 评测...")
                    eval_result = evaluate_full(report, reference, threshold=0.7)
                    precision = eval_result["precision"]
                    recall = eval_result["recall"]
                    f1 = eval_result["f1"]
                    # 保存到向量记忆
                    memory.save_to_vector_memory(
                        content=f"【算法评测】Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}",
                        category="自我评测",
                        metadata={"topic": self.topic, "precision": precision, "recall": recall, "f1": f1}
                    )
                    # 追加到 TXT 文件
                    memory.append_to_conversation(
                        f"【Precision/Recall 评测结果】\n"
                        f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\n"
                        f"详情: 报告句子数={eval_result['details']['report_sentences']}, "
                        f"参考句子数={eval_result['details']['ref_sentences']}, "
                        f"正确报告句子={eval_result['details']['correct_report_sents']}, "
                        f"覆盖参考句子={eval_result['details']['covered_ref_sents']}",
                        "评测报告"
                    )
                    print(f"✅ 评测完成: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                else:
                    print("⚠️ 无参考内容，无法进行 Precision/Recall 评测")
            except Exception as e:
                print(f"⚠️ 评测失败: {e}")
        elif self.enable_self_evaluation and not EVALUATOR_AVAILABLE:
            print("⚠️ evaluator.py 未找到，无法进行 Precision/Recall 评测")
        
        return report