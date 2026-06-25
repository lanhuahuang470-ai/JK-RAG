"""
层次切片文档处理模块（Hierarchical Chunking）

与 processor.py（固定长度递归切片）相比，本模块按文档的层次结构切分：
    章（第X章） → 节（第X节） → 条（第X条） → 款/项
适合规章制度、法律法规、标准规范等结构清晰的文档。

核心优势：
1. 每个切片尽量对应一个完整的"条款"，语义边界天然完整
2. 每个切片携带层次元数据（属于哪章/哪节/哪条），检索来源更精确
3. 超长条款自动回退到字符级递归切分，避免单片过大
4. 复用 DocumentProcessor 的 PDF/DOCX 提取与清洗逻辑，接口完全一致

用法（与 DocumentProcessor 完全相同）：
    from src.document_processor.hierarchical_processor import HierarchicalProcessor
    processor = HierarchicalProcessor()
    docs = processor.process_file("某规章.pdf")
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from src.document_processor.processor import DocumentProcessor
from src.utils import get_logger

logger = get_logger(__name__)


class HierarchicalProcessor(DocumentProcessor):
    """
    层次切片处理器

    继承 DocumentProcessor，复用其 PDF/DOCX 提取和文本清洗能力，
    仅重写切分逻辑（_split）为层次切分。
    """

    # ── 层次标题的正则模式（按从大到小的层级顺序）────────────────────────────
    # 每个层级：(层级名, 匹配该层级标题行的正则)
    # 参考「4-层次切片.py」的标记思路，合并中文规章 + 中文数字 + markdown 三套
    HEADING_PATTERNS = [
        # 中文规章层级：第X章 > 第X节 > 第X条
        ("章", re.compile(r"^\s*第\s*[一二三四五六七八九十百零0-9]+\s*章[\s　]*.*$")),
        ("节", re.compile(r"^\s*第\s*[一二三四五六七八九十百零0-9]+\s*节[\s　]*.*$")),
        ("条", re.compile(r"^\s*第\s*[一二三四五六七八九十百零0-9]+\s*条.*$")),
        # markdown 风格标题（# / ## / ###）
        ("h1", re.compile(r"^\s*#\s+.*$")),
        ("h2", re.compile(r"^\s*##\s+.*$")),
        ("h3", re.compile(r"^\s*###\s+.*$")),
        # 中文数字编号：一、二、三、（视为一级），（一）（二）（视为二级）
        ("cn1", re.compile(r"^\s*[一二三四五六七八九十]+\s*、.*$")),
        ("cn2", re.compile(r"^\s*[（(][一二三四五六七八九十]+[）)].*$")),
        # 阿拉伯数字编号：1. 2. 3.（视为三级）
        ("num", re.compile(r"^\s*[0-9]+\s*[.、]\s*\S.*$")),
    ]

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        max_chunk_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            chunk_size: 超长条款回退切分时的目标长度
            chunk_overlap: 回退切分时的重叠长度
            max_chunk_size: 单个切片的硬上限，超过则触发回退切分。
                            默认 = chunk_size * 1.5（给完整条款留余地）
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.max_chunk_size = max_chunk_size or int(chunk_size * 1.5)

        # 用于超长条款的回退切分器
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        logger.info(
            "HierarchicalProcessor 初始化完成 | "
            f"max_chunk_size={self.max_chunk_size} | "
            f"回退 chunk_size={chunk_size} | overlap={chunk_overlap}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 重写切分逻辑（这是与父类唯一的区别）
    # ──────────────────────────────────────────────────────────────────────────

    def _split(self, text: str, source: str) -> List[Document]:
        """
        层次切分：按章/节/条边界切分，并附加层次元数据。

        流程：
        1. 逐行扫描，识别层次标题，把文本切成"层次块"
        2. 每个层次块作为一个候选切片，记录它所属的章/节/条
        3. 超长的块回退到字符级递归切分
        """
        lines = text.split("\n")

        # 当前所处的层次上下文（随扫描更新）
        current_context: Dict[str, str] = {}
        # 累积的层次块：每块是 (上下文快照, 文本行列表)
        blocks: List[Dict] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            """把当前缓冲区的内容作为一个块存起来"""
            if buffer and any(line.strip() for line in buffer):
                blocks.append(
                    {
                        "context": dict(current_context),
                        "text": "\n".join(buffer).strip(),
                    }
                )
            buffer.clear()

        for line in lines:
            level = self._match_heading(line)
            if level is not None:
                # 遇到新标题：先把之前累积的内容存为一个块
                flush_buffer()
                # 更新层次上下文：进入新层级时，清空比它更低的层级
                self._update_context(current_context, level, line.strip())
                # 标题行本身也加入新块的开头
                buffer.append(line.strip())
            else:
                buffer.append(line)

        flush_buffer()  # 收尾：最后一块

        # 如果整篇没有任何层次标题，退化为父类的普通递归切分
        if not blocks or all(not b["context"] for b in blocks):
            logger.info(f"{source} 未检测到层次结构，回退为递归切分")
            return super()._split(text, source)

        # 把每个层次块转成 Document（超长则再切）
        documents = self._blocks_to_documents(blocks, source)
        logger.info(
            f"{source} 层次切分完成 | 识别 {len(blocks)} 个层次块 "
            f"→ {len(documents)} 个切片"
        )
        return documents

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _match_heading(self, line: str) -> Optional[str]:
        """判断一行是否是层次标题，返回层级名（章/节/条/h1...），否则 None"""
        stripped = line.strip()
        if not stripped:
            return None
        for level_name, pattern in self.HEADING_PATTERNS:
            if pattern.match(stripped):
                return level_name
        return None

    @staticmethod
    def _update_context(context: Dict[str, str], level: str, title: str) -> None:
        """
        进入某层级时更新上下文，并清空所有更低的层级。

        三套层级体系（各自独立排序）：
          中文规章： 章 > 节 > 条
          markdown： h1 > h2 > h3
          中文编号： cn1（一、）> cn2（（一））> num（1.）
        例：进入新的"第二章"，要清掉上一章遗留的"节""条"。
        """
        hierarchies = [
            ["章", "节", "条"],
            ["h1", "h2", "h3"],
            ["cn1", "cn2", "num"],
        ]
        for order in hierarchies:
            if level in order:
                idx = order.index(level)
                context[level] = title
                # 清空同体系内比当前层级更低的
                for lower in order[idx + 1 :]:
                    context.pop(lower, None)
                return

    def _blocks_to_documents(
        self, blocks: List[Dict], source: str
    ) -> List[Document]:
        """把层次块转为 Document 列表，超长块回退切分"""
        documents: List[Document] = []
        chunk_index = 0

        for block in blocks:
            context = block["context"]
            block_text = block["text"]
            if not block_text.strip():
                continue

            # 构造层次路径字符串，如 "第一章 总则 > 第三条"
            hierarchy_path = " > ".join(
                v for v in context.values() if v
            )

            if len(block_text) <= self.max_chunk_size:
                # 块不超长：整块作为一个切片，保持条款完整
                documents.append(
                    self._make_doc(
                        block_text, source, chunk_index, context, hierarchy_path
                    )
                )
                chunk_index += 1
            else:
                # 块超长：回退到字符级递归切分，每个子片继承同样的层次元数据
                sub_chunks = self._fallback_splitter.split_text(block_text)
                for sub in sub_chunks:
                    if not sub.strip():
                        continue
                    documents.append(
                        self._make_doc(
                            sub, source, chunk_index, context, hierarchy_path
                        )
                    )
                    chunk_index += 1

        return documents

    @staticmethod
    def _make_doc(
        text: str,
        source: str,
        chunk_index: int,
        context: Dict[str, str],
        hierarchy_path: str,
    ) -> Document:
        """构造带层次元数据的 Document"""
        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "chunk_size": len(text),
            "hierarchy_path": hierarchy_path,  # 如 "第一章 总则 > 第三条"
        }
        # 把章/节/条分别存为独立字段，便于按层级过滤检索
        for level, title in context.items():
            metadata[f"level_{level}"] = title

        return Document(page_content=text, metadata=metadata)
