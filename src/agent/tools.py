# Agent工具集：函数实现+OpenAI Tool Schema
from typing import Optional, List, Dict
from openai import OpenAI
from src.storage.database_manager import DatabaseManager
from src.retrieval.search_engine import HybridSearchEngine
from src.config import LLM_MODEL_NAME

# 全局实例（agent初始化时注入）
_db: Optional[DatabaseManager] = None
_engine: Optional[HybridSearchEngine] = None
_client: Optional[OpenAI] = None


def init_tools(db: DatabaseManager, engine: HybridSearchEngine, client: OpenAI = None):
    """Agent启动时注入数据库、检索引擎和LLM客户端"""
    global _db, _engine, _client
    _db = db
    _engine = engine
    _client = client


# 工具实现

def search_knowledge(query: str, top_k: int = 5) -> str:
    """混合检索知识库（接收大模型提炼好的精准关键词进行检索）"""
    if _engine is None:
        return "[ERROR] 检索引擎未初始化"
    
    # 【核心修改】：此处不再二次调用rewrite_query。
    # 因为agent.chat() 的确定性管道在检索前已先调用rewrite_query，
    # 此时传入的query已经是改写后的精准检索关键词。
    results = _engine.search(query, top_k=top_k)
    
    if not results:
        return "未找到相关内容。"
        
    parts = []
    for i, r in enumerate(results):
        # 降噪策略1: 移除相关度得分和无用的格式前缀，只保留文件名作为来源标注
        # 降噪策略2: 确保文本前后没有多余的空格或换行
        clean_content = r.content.strip()
        parts.append(f"[来源: {r.source_file}]\n{clean_content}")
        
        # 降噪策略3: 如果当前是在评估RAG回答能力，而不是专门评估关联QA，
        # 强烈建议在此阶段暂时注释掉qa_pairs，因为它极容易误导大模型的理解逻辑。
        # if r.qa_pairs:
        #     for qa in r.qa_pairs[:1]: # 如果非要保留，最多只留1个
        #         parts.append(f"  参考问答: {qa['question']} -> {qa['answer']}")
                
    # 用分隔符区分不同的片段，清晰明了
    return "\n---\n".join(parts)


def get_chunk(chunk_id: str) -> str:
    """获取指定chunk的完整内容和关联QA"""
    if _db is None:
        return "[ERROR] 数据库未初始化"
    # 从SQLite查chunk
    cursor = _db.sqlite_conn.cursor()
    cursor.execute("SELECT content, source_file FROM chunks c JOIN documents d ON c.doc_id=d.doc_id WHERE chunk_id=?", (chunk_id,))
    row = cursor.fetchone()
    if not row:
        return f"未找到 chunk: {chunk_id}"
    content, source = row

    qa_pairs = _db.get_qa_by_chunk(chunk_id)
    parts = [f"[{source}]\n{content}"]
    if qa_pairs:
        parts.append("\n关联问答:")
        for qa in qa_pairs:
            parts.append(f"  Q: {qa['question']}\n  A: {qa['answer']}")
    return "\n".join(parts)


def get_document(doc_id: str) -> str:
    """获取文档元数据及其所有chunk"""
    if _db is None:
        return "[ERROR] 数据库未初始化"
    doc = _db.get_document(doc_id)
    if not doc:
        return f"未找到文档: {doc_id}"

    cursor = _db.sqlite_conn.cursor()
    cursor.execute("SELECT chunk_id, chunk_index, char_count FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,))
    chunks = cursor.fetchall()

    lines = [
        f"文档: {doc['file_name']}",
        f"ID: {doc['doc_id']}",
        f"大小: {doc['file_size_bytes']:,}字节 | 页数: {doc['page_count']}",
        f"状态: {doc['parse_status']} | 创建: {doc['created_at']}",
    ]
    if doc.get("tags"):
        lines.append(f"标签: {', '.join(t['name'] for t in doc['tags'])}")
    lines.append(f"\n包含 {len(chunks)} 个 Chunk:")
    for cid, idx, cc in chunks:
        lines.append(f"  [{idx}] {cid} ({cc}字)")

    return "\n".join(lines)


def list_documents(status: str = "completed") -> str:
    """列出知识库中的文档"""
    if _db is None:
        return "[ERROR] 数据库未初始化"
    cursor = _db.sqlite_conn.cursor()
    cursor.execute(
        "SELECT doc_id, file_name, page_count, file_size_bytes, created_at FROM documents WHERE parse_status=? ORDER BY created_at DESC LIMIT 20",
        (status,))
    docs = cursor.fetchall()
    if not docs:
        return f"没有状态为 '{status}' 的文档。"

    lines = [f"共 {len(docs)} 个文档 ({status}):"]
    for did, name, pages, size, created in docs:
        lines.append(f"  {did} | {name} | {pages}页 | {size:,}字节 | {created}")
    return "\n".join(lines)


def get_stats() -> str:
    """获取系统全局统计"""
    if _db is None:
        return "[ERROR] 数据库未初始化"
    stats = _db.get_stats()
    return (
        f"系统统计:\n"
        f"  文档: {stats['documents']} 个 ({stats.get('doc_by_status', {})})\n"
        f"  Chunk: {stats['chunks']} 个\n"
        f"  QA对: {stats['qa_pairs']} 条 ({stats.get('qa_by_filter', {})})\n"
        f"  审计日志: {stats['audit_log']} 条"
    )


# Query改写

def rewrite_query(original: str) -> str:
    """将用户口语改写为精确检索关键词"""
    if _client is None:
        return original  # 无LLM客户端时原样返回
    try:
        r = _client.chat.completions.create(
            model=LLM_MODEL_NAME, temperature=0, max_tokens=60,
            messages=[{"role": "system", "content": "将用户问题改写为5-10个搜索关键词，用空格分隔。只输出关键词。"},
                      {"role": "user", "content": original}],
        )
        return r.choices[0].message.content.strip()
    except:
        return original


# OpenAI Tool Schema

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rewrite_query",
            "description": "将用户口语化问题改写为精确搜索关键词，提升检索命中率。先改写再搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "original": {"type": "string", "description": "用户原始问题"}
                },
                "required": ["original"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在文档知识库中进行语义检索。当用户询问任何需要查文档的问题时，优先使用此工具。返回相关文本片段及来源标注。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询，应提炼用户问题的核心关键词"},
                    "top_k": {"type": "integer", "description": "返回结果数量，默认5，需要更多信息时可增大到8-10"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": "获取指定chunk的完整文本内容和关联QA对。当检索结果中的片段不够完整时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "chunk的唯一标识ID"}
                },
                "required": ["chunk_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "获取文档的元数据、标签和chunk列表。当用户询问某个文档的概况、或需要了解文档结构时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档ID，如 doc_xxx"}
                },
                "required": ["doc_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出知识库中的所有文档。当用户询问'有哪些文档'、'知识库里有什么'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["completed", "parsing", "failed"],
                        "description": "筛选文档状态，默认completed"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "获取知识库的全局统计信息，包括文档数、QA对数、处理日志数等。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]

# 工具名 → 函数映射
TOOL_MAP = {
    "rewrite_query": rewrite_query,
    "search_knowledge": search_knowledge,
    "get_chunk": get_chunk,
    "get_document": get_document,
    "list_documents": list_documents,
    "get_stats": get_stats,
}
