# tools.py
"""Agent 工具定义与执行器（学术研究异步版，支持链接深度抓取）"""
import json
import asyncio
import os
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI

from search import google_ai_search
from memory import save_memory_entry

load_dotenv()

summary_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


async def summarize_search_result(raw_text: str, query: str) -> str:
    """异步提取学术搜索结果中的核心内容和来源链接"""
    # 提取链接
    sources_section = ""
    if "--- Sources ---" in raw_text:
        parts = raw_text.split("--- Sources ---")
        if len(parts) > 1:
            sources_section = parts[1].split("---")[0].strip()
            if sources_section:
                sources_section = "**参考资料（学术来源优先）：**\n" + sources_section
            else:
                sources_section = ""

    # 调用模型总结
    try:
        response = await summary_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个学术信息提取助手。用户会提供一段 Google 搜索结果的原始文本（可能包含非学术内容）。"
                        "请提取与搜索查询最相关的**学术性核心内容**，例如：研究结论、数据、方法、作者观点等。"
                        "用简洁的中文归纳总结，至少3个段落。忽略广告、论坛、个人博客等非权威来源。"
                        "注意：原始文本末尾可能包含 '--- Sources ---' 部分，你**不需要**在总结中包含它，我会单独附加。"
                        "只输出总结段落。"
                    )
                },
                {
                    "role": "user",
                    "content": f"学术搜索查询：{query}\n\n原始搜索结果：\n{raw_text}"
                }
            ],
            temperature=0.1,
            max_tokens=800
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"⚠️ 总结失败，使用原始内容：\n{raw_text[:1500]}"

    final_result = summary
    if sources_section:
        final_result += f"\n\n{sources_section}"
    else:
        final_result += "\n\n（未找到来源链接，请考虑使用更学术的搜索词。）"
    return final_result


async def fetch_and_extract(url: str, reason: str = "") -> str:
    """异步抓取 URL 内容并用 LLM 提取学术信息"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            # 简单提取文本（去掉 script/style）
            soup = BeautifulSoup(resp.text, 'lxml')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')
            # 清理空白行
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            # 限制长度（前 5000 字符）
            if len(text) > 5000:
                text = text[:5000] + "..."
            
            # 使用 LLM 总结
            response = await summary_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个学术内容提取助手。用户会提供网页的正文文本。请提取与学术相关的核心信息：研究问题、方法、结果、结论。如果文本不全，提取可用信息。输出简洁的中文总结。"
                    },
                    {
                        "role": "user",
                        "content": f"URL: {url}\n原因: {reason if reason else '深度分析'}\n\n网页内容:\n{text}"
                    }
                ],
                temperature=0.2,
                max_tokens=1000
            )
            summary = response.choices[0].message.content
            return f"【链接内容摘要】\n来源: {url}\n{summary}\n\n原始内容长度: {len(text)} 字符"
    except Exception as e:
        return f"❌ 抓取链接失败 {url}: {str(e)}"


# 工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "google_ai_search",
            "description": (
                "使用 Google 搜索引擎搜索**学术论文、研究综述、权威报告**。"
                "**重要**：构造查询时，请添加学术限定词，例如：'review', 'paper', 'study', 'filetype:pdf', 'site:edu', 'site:researchgate.net' 等。"
                "返回的结果包含【AI 总结】和【参考资料】（含链接）。你必须保存完整的返回内容，不要删减链接。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "学术搜索查询词。示例：'machine learning review paper' 或 'climate change impact study filetype:pdf'。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "follow_link",
            "description": "从搜索结果中提取到的学术链接（如 PDF、论文页面、期刊文章），打开该链接并提取页面中的核心学术内容（摘要、方法、结论等）。用于深入挖掘某篇论文或某个具体资源。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要访问的完整 URL，必须来自搜索结果中的链接。"
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要跟进此链接，例如：'获取该论文的具体实验数据'。"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": "将重要的学术发现存入研究记忆。**重要：当你收到任何工具返回结果后，应调用此工具保存关键信息。**",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要存储的学术内容，必须包含完整的搜索结果或链接摘要。"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["核心发现", "来源链接", "研究方法", "深度研究"],
                        "description": "内容分类标签。"
                    }
                },
                "required": ["content", "category"]
            }
        }
    }
]


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """异步执行工具"""
    if tool_name == "google_ai_search":
        query = tool_args.get("query", "")
        print(f"🔍 正在执行学术搜索: {query}")
        raw_result = await google_ai_search(query)
        print("🧹 正在对搜索结果进行学术提炼...")
        summarized = await summarize_search_result(raw_result, query)
        return f"【学术搜索结果 - 请务必保存完整内容，包括链接】\n{summarized}"

    elif tool_name == "follow_link":
        url = tool_args.get("url", "")
        reason = tool_args.get("reason", "")
        print(f"🔗 正在跟进链接: {url} (原因: {reason if reason else '深度分析'})")
        result = await fetch_and_extract(url, reason)
        # 自动保存到记忆（避免模型忘记保存）
        await asyncio.to_thread(save_memory_entry, f"跟进链接 {url}:\n{result}", "深度研究")
        return result

    elif tool_name == "save_to_memory":
        content = tool_args.get("content", "")
        category = tool_args.get("category", "研究笔记")
        await asyncio.to_thread(save_memory_entry, content, category)
        return f"✅ 已成功存入学术记忆（分类：{category}）"

    else:
        return f"❌ 错误：未知工具 '{tool_name}'"