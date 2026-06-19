================================================================================
                    学术研究 Agent —— 基于大语言模型的多智能体协作系统
================================================================================

项目简介
--------
本项目是一个学术研究自动化助手，集成了单 Agent 快速研究和多 Agent 深度协作两种模式。
系统能够：

  * 通过 Google 搜索（真实浏览器模拟）获取学术资源
  * 深度抓取论文/网页内容并自动提取核心信息
  * 使用向量数据库（ChromaDB + Sentence-Transformers）实现长期记忆，跨会话复用历史知识
  * 多 Agent 分工协作：规划员、搜索员、分析员、写作者，共同完成高质量研究报告
  * 内置 Precision / Recall 自动评测，量化报告的信息覆盖质量
  * 每次研究自动生成带时间戳的 TXT 日志，所有记忆持久化保存


主要特性
--------
  双模式         : 单 Agent（简单快速） / 多 Agent（深度协作，默认启用评测）
  真实搜索       : 基于 Playwright 的无头/有头浏览器，模拟人类搜索行为，支持反检测
  链接深度抓取   : 从搜索结果中选择学术链接，自动提取页面正文并用 LLM 摘要
  向量记忆       : 所有研究笔记、发现、报告均存入 ChromaDB，支持语义检索
  历史记忆检索   : 每次研究开始时自动检索相关历史记忆，避免重复劳动
  多 Agent 协作  : Planner -> Searcher -> Analyzer -> Writer，可自动补充搜索
  自动评测       : 多 Agent 模式默认计算 Precision/Recall/F1（基于句子向量相似度）
  持久化         : 每个会话生成独立的 .txt 文件，向量库持久化，便于管理和回溯


系统架构
--------
用户输入主题
    |
    +-- 单 Agent 模式 (ResearchSession)
    |     +-- 检索历史记忆
    |     +-- 循环调用工具：搜索 -> 抓取 -> 保存记忆
    |     +-- 生成最终报告（不评测）
    |
    +-- 多 Agent 模式 (MultiAgentSession)
          +-- 检索历史记忆
          +-- Planner 分解子任务
          +-- Searcher 依次执行每个子任务（搜索 + 抓取）
          +-- Analyzer 评估信息完整性，必要时补充搜索
          +-- Writer 撰写完整报告
          +-- 自动执行 Precision/Recall 评测（可选）

核心模块说明：
  - agent.py      : 命令行交互入口，模式选择，配置管理
  - core.py       : 单/多 Agent 会话核心逻辑
  - agents.py     : 四个专业 Agent 的实现（规划、搜索、分析、写作）
  - tools.py      : 工具定义与执行（搜索、抓取、保存记忆）及学术摘要增强
  - search.py     : Playwright 驱动的 Google 搜索（带反检测）
  - memory.py     : 向量数据库管理、TXT 日志、历史检索
  - evaluator.py  : Precision/Recall 评测模块（基于 Sentence-BERT）


快速开始
--------
1. 环境准备
   - Python 3.9 或更高版本
   - DeepSeek API Key（或其他兼容 OpenAI 接口的 LLM）
   - 安装 Playwright 浏览器（用于搜索）：
       playwright install chromium

2. 克隆项目
       git clone https://github.com/your-username/academic-research-agent.git
       cd academic-research-agent

3. 安装依赖
       pip install -r requirements.txt
   注：requirements.txt 应包含 openai, chromadb, sentence-transformers, playwright,
       beautifulsoup4, lxml, httpx, python-dotenv, numpy 等。
       playwright-stealth 为可选。

4. 配置环境变量
   创建 .env 文件，内容如下：
       DEEPSEEK_API_KEY=your_deepseek_api_key_here
       SEARCH_LANG=en-US        # 可选
       SEARCH_REGION=US         # 可选

5. 运行
       python agent.py
   启动后，根据提示输入研究主题，选择模式（单/多 Agent），并决定是否启用历史记忆检索。


使用示例
--------
   学术研究 Agent 启动 (支持长期记忆)
   主菜单：
     直接输入研究主题 - 开始新的学术研究
     输入 'config' 或 '/config' - 进入配置模式
     输入 'DONE' - 退出程序
   ==================================================

   输入: Transformer 在 NLP 中的应用
   是否启用历史记忆检索？(y/n, 默认 y): y
   选择研究模式: 1-简易搜索 (默认) / 2-深度搜索: 2

   开始多 Agent 协作学术研究: Transformer 在 NLP 中的应用
   历史记忆检索已启用
   Planner Agent 正在制定研究计划...
   研究计划：3 个子任务
      1. Transformer 模型的基本架构与原理
      2. Transformer 在 NLP 任务中的主要应用
      3. Transformer 的最新进展与挑战
   ...
   Writer Agent 正在撰写研究报告...
   研究完成！
   报告已保存至 memories/ 目录
   评测完成: Precision=0.823, Recall=0.756, F1=0.788


配置选项
--------
  - 环境变量：DEEPSEEK_API_KEY 必填，其他可选。
  - 搜索参数：在 search.py 中可调整 GOOGLE_BASE_URL, SEARCH_LANGUAGE, USER_AGENTS 等。
  - 记忆设置：在 memory.py 中可修改 EMBEDDING_MODEL, VECTOR_COLLECTION_NAME。
  - 评测阈值：在 evaluator.py 中调整 _DEFAULT_THRESHOLD（默认 0.7）。
  - Agent 模型：默认使用 deepseek-chat，可在实例化时传入 model 参数。


文件结构
--------
   .
   ├── agent.py                 # 命令行入口
   ├── core.py                  # 会话核心（单/多 Agent）
   ├── agents.py                # 多 Agent 定义
   ├── tools.py                 # 工具函数 + LLM 摘要
   ├── search.py                # Google 搜索（Playwright）
   ├── memory.py                # 向量数据库 + TXT 日志
   ├── evaluator.py             # Precision/Recall 评测
   ├── memories/                # 自动生成的研究记录（TXT + ChromaDB）
   │   ├── 20250619_143022_Transformer在NLP中的应用.txt
   │   └── chroma_db/           # 向量数据库持久化目录
   ├── .env                     # 环境变量（需自行创建）
   └── requirements.txt


配置模式
--------
在主菜单输入 config 或 /config，可进入配置管理：
  - 查看向量数据库统计（总记忆数、集合名）
  - 清空所有向量记忆（谨慎操作）


评测机制
--------
多 Agent 模式下，系统会自动计算最终报告相对于本次研究收集的所有笔记的 Precision 和 Recall：

  - Precision ：报告中与参考材料语义相似的句子比例（反映报告的真实性/准确性）
  - Recall    ：参考材料中的信息被报告覆盖的比例（反映报告的完整性）
  - F1        ：两者的调和平均

评测结果会存入向量记忆并追加到会话 TXT 中，便于后续分析。
若想关闭评测，可在 MultiAgentSession 中设置 enable_self_evaluation=False。


依赖与注意事项
--------------
  - 网络要求：Google 搜索需能正常访问（若受限，可调整代理设置或使用其他搜索引擎）。
  - 反爬策略：search.py 内置随机 User-Agent、窗口大小、滚动延迟等，但仍有被 Google
    临时封禁的风险，建议合理控制搜索频率。
  - LLM 费用：每次研究会调用 DeepSeek API（或您配置的模型），注意额度消耗。
  - 首次运行：会下载 Sentence-Transformer 模型（约80MB）和 ChromaDB 所需文件，
    请确保网络畅通。


贡献
----
欢迎提交 Issue 或 Pull Request。如果您有改进建议（如增加更多搜索引擎、支持本地 LLM、
优化评测算法等），请随时联系。


许可证
------
本项目采用 MIT License（详见 LICENSE 文件）。


联系方式
--------
如有问题，请通过 GitHub Issues 联系。


Happy Researching!  🎓
================================================================================
