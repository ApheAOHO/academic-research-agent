# memory.py
"""研究记忆系统：每次对话生成独立的 TXT 文件 + 向量数据库长期记忆（无重排序）"""
import os
# 设置 Hugging Face 镜像（解决网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

# ========== 配置 ==========
CONVERSATIONS_DIR = "memories"
VECTOR_COLLECTION_NAME = "research_memories"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # 轻量级向量模型，~80MB

# 全局变量
_current_conversation_base = None  # 当前会话的 TXT 文件基础路径
_current_conversation_id = None    # 当前会话的 UUID/时间戳标识
_vector_client = None
_vector_collection = None
_embedding_model = None


# ========== 初始化向量数据库 ==========
def _init_vector_db():
    """初始化 Chroma 向量数据库（单例模式）"""
    global _vector_client, _vector_collection, _embedding_model
    
    if _vector_client is not None:
        return
    
    # 确保数据目录存在
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    
    # 初始化 Chroma（持久化到磁盘）
    _vector_client = chromadb.PersistentClient(
        path=os.path.join(CONVERSATIONS_DIR, "chroma_db")
    )
    
    # 获取或创建 collection
    try:
        _vector_collection = _vector_client.get_collection(VECTOR_COLLECTION_NAME)
    except:
        _vector_collection = _vector_client.create_collection(
            name=VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
    
    # 加载本地 embedding 模型（首次运行会自动下载）
    print(f"🧠 正在加载 Embedding 模型: {EMBEDDING_MODEL} (首次运行会下载，约80MB)...")
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding 模型加载完成")


def _get_embedding(text: str) -> List[float]:
    """将文本转换为向量"""
    if _embedding_model is None:
        _init_vector_db()
    return _embedding_model.encode(text).tolist()


def _generate_memory_id(content: str, category: str) -> str:
    """根据内容和分类生成唯一 ID（用于去重）"""
    hash_input = f"{category}:{content[:200]}"
    return hashlib.md5(hash_input.encode()).hexdigest()


# ========== 文件名处理 ==========
def sanitize_filename(name: str) -> str:
    """清理字符串，用于文件名"""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    if len(name) > 50:
        name = name[:50]
    return name.strip()


# ========== TXT 文件操作 ==========
def start_new_conversation(topic: str) -> str:
    """开始新对话，创建 TXT 文件，返回基础路径（不含扩展名）"""
    global _current_conversation_base, _current_conversation_id
    _init_vector_db()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _current_conversation_id = timestamp
    safe_topic = sanitize_filename(topic)
    filename = f"{timestamp}_{safe_topic}" if safe_topic else f"{timestamp}_research"
    
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    base_path = os.path.join(CONVERSATIONS_DIR, filename)
    _current_conversation_base = base_path
    
    # 初始化 TXT 文件
    txt_header = f"""研究对话: {topic}
开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
会话ID: {timestamp}

========== 研究记录 ==========

"""
    
    with open(f"{base_path}.txt", "w", encoding="utf-8") as f:
        f.write(txt_header)
    
    print(f"📄 研究记录将保存至 memories/{filename}.txt")
    print(f"🧠 向量记忆库已就绪 (collection: {VECTOR_COLLECTION_NAME})")
    return base_path


def _write_to_txt(content: str, mode: str = "a"):
    """向当前对话的 TXT 文件追加内容"""
    global _current_conversation_base
    if _current_conversation_base is None:
        print("警告: 未开始对话，内容未保存")
        return False
    try:
        with open(f"{_current_conversation_base}.txt", mode, encoding="utf-8") as f_txt:
            f_txt.write(content)
        return True
    except Exception as e:
        print(f"写入 TXT 文件失败: {e}")
        return False


def append_to_conversation(content: str, category: str = "笔记"):
    """向当前对话的 TXT 文件追加内容（纯文本格式）"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    txt_content = f"[{category}] ({timestamp})\n{content}\n\n"
    _write_to_txt(txt_content, "a")
    
    # 同时保存到向量数据库（长期记忆）
    save_to_vector_memory(content, category)


# ========== 向量数据库长期记忆 ==========
def save_to_vector_memory(content: str, category: str, metadata: Optional[Dict] = None):
    """
    将记忆存入向量数据库
    - content: 记忆内容
    - category: 分类（核心发现/来源链接/研究笔记/待深挖问题）
    - metadata: 额外元数据（如会话ID、时间戳等）
    """
    global _current_conversation_id
    _init_vector_db()
    
    if not content or len(content.strip()) < 10:
        return  # 内容太短，不保存
    
    memory_id = _generate_memory_id(content, category)
    
    # 准备元数据
    meta = {
        "category": category,
        "session_id": _current_conversation_id or "unknown",
        "timestamp": datetime.now().isoformat(),
        "content_preview": content[:200],
    }
    if metadata:
        meta.update(metadata)
    
    try:
        # 检查是否已存在（避免重复）
        existing = _vector_collection.get(ids=[memory_id])
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            # 已存在，可选择性更新（这里跳过）
            return
        
        # 生成向量并存储
        embedding = _get_embedding(content)
        _vector_collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[content]
        )
    except Exception as e:
        print(f"向量存储失败: {e}")


def search_similar_memories(
    query: str,
    limit: int = 5,
    category_filter: Optional[str] = None
) -> List[Dict]:
    """
    搜索相关的历史记忆（纯向量检索，无重排序）
    - query: 搜索查询
    - limit: 返回结果数量
    - category_filter: 可选，只返回特定分类（如"核心发现"）
    返回: 列表，每个元素包含 document, metadata, similarity
    """
    _init_vector_db()
    
    if _vector_collection is None or _vector_collection.count() == 0:
        return []
    
    try:
        query_embedding = _get_embedding(query)
        
        # 构建过滤条件
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        results = _vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化返回结果
        memories = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                memories.append({
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 0,
                })
        return memories
    except Exception as e:
        print(f"向量搜索失败: {e}")
        return []


def get_relevant_history(topic: str, limit: int = 3) -> str:
    """
    获取与主题相关的历史记忆，格式化为文本供 Agent 参考
    返回: 格式化的历史上下文文本
    """
    memories = search_similar_memories(topic, limit=limit)
    if not memories:
        return ""
    
    context_parts = ["\n【相关历史研究成果】"]
    for i, mem in enumerate(memories, 1):
        similarity = mem.get("similarity", 0)
        if similarity < 0.5:
            continue
        context_parts.append(f"\n{i}. [相似度 {similarity:.2f}] (分类: {mem['metadata'].get('category', '未知')})")
        context_parts.append(f"   {mem['content'][:500]}")
    
    if len(context_parts) == 1:
        return ""
    
    context_parts.append("\n注：以上是你之前研究过的内容，可以参考避免重复搜索\n")
    return "\n".join(context_parts)


# ========== 兼容旧接口 ==========
def save_memory_entry(content: str, category: str):
    """兼容原有接口：保存记忆条目（同时存 TXT 和向量库）"""
    append_to_conversation(content, category)


def get_all_memory() -> list:
    """保留兼容接口，返回空列表（实际内容已存入文件和向量库）"""
    return []


def finalize_conversation(final_summary: str = ""):
    """完成对话，追加最终总结并关闭当前会话"""
    global _current_conversation_base
    if final_summary:
        append_to_conversation(final_summary, "📄 最终报告")
    
    if _current_conversation_base:
        txt_path = f"{_current_conversation_base}.txt"
        try:
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"\n========== 对话结束 ==========\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except:
            pass
    
    _current_conversation_base = None


def ensure_conversation(topic="未命名研究"):
    """如果当前没有对话文件，自动创建一个（用于向后兼容）"""
    global _current_conversation_base
    if _current_conversation_base is None:
        start_new_conversation(topic)


# ========== 管理功能 ==========
def get_vector_stats() -> Dict:
    """获取向量数据库统计信息"""
    _init_vector_db()
    if _vector_collection:
        return {
            "total_memories": _vector_collection.count(),
            "collection_name": VECTOR_COLLECTION_NAME,
        }
    return {"total_memories": 0, "collection_name": VECTOR_COLLECTION_NAME}


def clear_vector_memories():
    """清空所有向量记忆（谨慎使用）"""
    global _vector_client, _vector_collection
    
    _init_vector_db()
    if _vector_collection:
        # 删除并重建 collection
        _vector_client.delete_collection(VECTOR_COLLECTION_NAME)
        _vector_collection = _vector_client.create_collection(
            name=VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("🧹 向量记忆已清空")


def get_current_session_reference() -> str:
    """获取当前会话中保存的所有研究笔记（排除最终报告），用于评测"""
    global _current_conversation_base
    if _current_conversation_base is None:
        return ""
    txt_path = f"{_current_conversation_base}.txt"
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        # 截取 "========== 研究记录 ==========" 之后的部分，直到 "📄 最终报告" 之前
        match = re.search(r"========== 研究记录 ==========\n(.*?)(?=\n📄 最终报告|\Z)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
    except Exception as e:
        print(f"读取参考材料失败: {e}")
        return ""