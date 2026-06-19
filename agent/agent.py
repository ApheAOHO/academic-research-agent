# agent.py
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 导入核心会话类（单Agent和多Agent）
from core import ResearchSession, MultiAgentSession
import memory

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY not found")
client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

async def configure_mode():
    """配置模式：查看统计、清空向量数据库等"""
    while True:
        print("\n🔧 配置模式")
        print("  1. 查看向量数据库统计 (/stats)")
        print("  2. 清空所有向量记忆 (/clear)")
        print("  3. 返回主菜单(/return)")
        choice = await asyncio.to_thread(input, "请选择 (1/2/3): ")
        
        if choice == "1":
            stats = memory.get_vector_stats()
            print(f"\n📊 向量数据库统计:")
            print(f"   Collection: {stats.get('collection_name', 'N/A')}")
            print(f"   总记忆数: {stats.get('total_memories', 0)}")
            if 'note' in stats:
                print(f"   备注: {stats['note']}")
        elif choice == "2":
            print("\n⚠️  警告：此操作将永久删除所有向量记忆，不可恢复！")
            confirm = await asyncio.to_thread(input, "请输入 'yes' 确认清空: ")
            if confirm.lower() == 'yes':
                memory.clear_vector_memories()
                print("✅ 向量记忆已清空")
            else:
                print("❌ 操作已取消")
        elif choice == "3":
            print("返回主菜单")
            break
        else:
            print("❌ 无效选择，请重新输入")

async def main():
    print("🧠 学术研究 Agent 启动 (支持长期记忆)")
    print("主菜单：")
    print("  直接输入研究主题 - 开始新的学术研究")
    print("  输入 'config' 或 '/config' - 进入配置模式（查看/管理向量数据库）")
    print("  输入 'DONE' - 退出程序")
    print("=" * 50)
    
    while True:
        user_input = await asyncio.to_thread(input, "\n👤 输入: ")
        cmd = user_input.strip()
        
        if cmd.upper() == "DONE":
            print("👋 退出")
            break
        
        if cmd.lower() in ["config", "/config"]:
            await configure_mode()
            continue
        
        # 询问是否启用历史记忆检索
        use_history = await asyncio.to_thread(
            input, "是否启用历史记忆检索？(y/n, 默认 y): "
        )
        enable_history = use_history.lower() != 'n'
        
        # 选择研究模式
        mode_choice = await asyncio.to_thread(
            input, "选择研究模式: 1-简易搜索 (默认) / 2-深度搜索: "
        )
        
        if mode_choice == "2":
            print(f"\n🚀 开始多 Agent 协作学术研究: {cmd}")
            if enable_history:
                print("📚 历史记忆检索已启用")
            print("-" * 50)
            session = MultiAgentSession(
                topic=cmd,
                llm_client=client,
                enable_history=enable_history,
                history_limit=3,
            )
        else:
            print(f"\n🚀 开始单 Agent 学术研究: {cmd}")
            if enable_history:
                print("📚 历史记忆检索已启用")
            print("-" * 50)
            session = ResearchSession(
                topic=cmd,
                llm_client=client,
                max_searches=5,
                min_searches=3,
                enable_history=enable_history,
                history_limit=3,
            )
        
        try:
            report = await session.run()
            print("\n" + "=" * 50)
            print("✅ 研究完成！")
            print(f"📄 报告已保存至 memories/ 目录")
            print(f"🤖 报告摘要: {report[:300]}...")
            print("=" * 50)
        except Exception as e:
            print(f"❌ 研究失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())