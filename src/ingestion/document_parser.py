# 文档处理模块：父子切片
import os, re, fitz
from typing import List, Dict, Tuple
from src.config import CHUNK_SIZE

PARENT_SIZE = 600     # 父块: 完整语义上下文
CHILD_SIZE = 200      # 子块: 精准检索单元
CHILD_OVERLAP = 60    # 子块间重叠


class DocumentParser:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        print(f"初始化文档解析器 (父子切片: Parent{PARENT_SIZE}/Child{CHILD_SIZE}字)")

    def load_pdf(self, file_path: str) -> Dict:
        try:
            file_size = os.path.getsize(file_path)
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text("text") + "\n" for page in doc)
                return {"text": text, "page_count": len(doc), "file_size_bytes": file_size}
        except Exception as e:
            print(f"读取PDF失败: {e}")
            return {}

    def split_text_semantically(self, text: str) -> List[str]:
        """兼容旧接口：返回child chunks"""
        return self.split_parent_child(text)["children"]

    def split_parent_child(self, text: str) -> Dict:
        """
        父子切片。
        先清理换行符与空格，再按标点符号切分自然句；
        将句子聚合成父块（PARENT_SIZE），再从父块切出子块（CHILD_SIZE）。
        返回: {"parents": [(chunk_id, content), ...], "children": [(chunk_id, parent_id, content), ...]}
        """
        if not text:
            return {"parents": [], "children": []}
        text = re.sub(r'\n+', ' ', text) #消除空格和换行
        sentences = re.split(r'(?<=[。！？.…;!?])', text) #按照中英文的句号、叹号、问号、省略号、分号等断句符号，将整篇文本切分成一个个独立的句子组成的列表
        sentences = [s.strip() for s in sentences if s.strip()] #清理空白与空句
        if not sentences:
            return {"parents": [], "children": []}

        parents, children = [], []
        parent_idx = 0
        current_parent = []
        current_len = 0

        for s in sentences:
            if current_len + len(s) <= PARENT_SIZE:
                current_parent.append(s)
                current_len += len(s)
            else:
                # 保存当前parent
                parent_text = " ".join(current_parent)
                parent_id = f"p{parent_idx:03d}"
                parents.append((parent_id, parent_text))

                # 从parent切出child chunks
                child_chunks = self._split_into_children(current_parent)
                for ci, child_text in enumerate(child_chunks):
                    children.append((f"{parent_id}_c{ci:02d}", parent_id, child_text))

                # 开始下一个parent（带overlap：最后两句作为上下文衔接）
                overlap_sents = current_parent[-2:] if len(current_parent) >= 2 else []
                current_parent = overlap_sents + [s]
                current_len = sum(len(x) for x in current_parent)
                parent_idx += 1

        # 最后一个parent
        if current_parent:
            parent_text = " ".join(current_parent)
            parent_id = f"p{parent_idx:03d}"
            parents.append((parent_id, parent_text))
            for ci, child_text in enumerate(self._split_into_children(current_parent)):
                children.append((f"{parent_id}_c{ci:02d}", parent_id, child_text))

        return {"parents": parents, "children": children}

    def _split_into_children(self, sentences: List[str]) -> List[str]:
        """将一组句子切成child chunks"""
        children = []
        current = []
        current_len = 0
        for s in sentences:
            if current_len + len(s) <= CHILD_SIZE:
                current.append(s)
                current_len += len(s)
            else:
                if current:
                    children.append(" ".join(current))
                # 新child从上一child末尾取overlap
                overlap_chars = 0
                start_idx = len(current) - 1
                while start_idx >= 0 and overlap_chars < CHILD_OVERLAP:
                    overlap_chars += len(current[start_idx])
                    start_idx -= 1
                current = current[max(0, start_idx + 1):] + [s]
                current_len = sum(len(x) for x in current)
        if current:
            children.append(" ".join(current))
        return children
