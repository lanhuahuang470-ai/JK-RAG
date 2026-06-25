"""
语义切片文档处理模块（Semantic Chunking）

参考自独立脚本「2-语义切片.py」，改造为符合本工程规范的版本：
按句子边界（。！？.!?；等）切分，累积句子直到接近目标长度，
保证每个切片都由完整句子组成，不会从句子中间切断。

与参考脚本的区别（工程化改造）：
1. 返回带元数据的 List[Document]，而非纯字符串列表
2. 复用 DocumentProcessor 的 PDF/DOCX 提取与清洗逻辑
3. 增加 chunk_overlap 支持（参考脚本没有重叠，本版补上）
4. 保留句子结束标点（参考脚本会丢标点，本版保留语义完整）

用法（与 DocumentProcessor 完全相同）：
    from src.document_processor.semantic_processor import SemanticProcessor
    processor = SemanticProcessor()
    docs = processor.process_file("某文档.pdf")
"""
from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from config.settings import settings
from src.document_processor.processor import DocumentProcessor
from src.utils import get_logger

logger = get_logger(__name__)


class SemanticProcessor(DocumentProcessor):
    """
    语义切片处理器

    继承 DocumentProcessor，复用提取与清洗能力，仅重写 _split。
    """

    # 句子结束标点（中英文），用于按句子切分
    # 用捕获组保留标点，避免切完丢失句号
    SENTENCE_PATTERN = re.compile(r"([。！？!?\n]+|；;)")

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(
            "SemanticProcessor 初始化完成 | "
            f"chunk_size={chunk_size} | chunk_overlap={chunk_overlap}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 重写切分逻辑
    # ──────────────────────────────────────────────────────────────────────────

    def _split(self, text: str, source: str) -> List[Document]:
        """
        语义切分：
        1. 把文本拆成一个个完整句子（保留句末标点）
        2. 依次累积句子，当再加一句会超过 chunk_size 时，结束当前切片
        3. 下一切片回溯 chunk_overlap 个字符的句子作为重叠，保持上下文连续
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []        # 当前切片累积的句子
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            # 加上这句会超长，且当前切片非空 → 先收束当前切片
            if current_len + sent_len > self.chunk_size and current:
                chunks.append("".join(current))
                # 回溯：从当前切片末尾取若干句子作为下一片的重叠开头
                current, current_len = self._build_overlap(current)

            current.append(sent)
            current_len += sent_len

        # 收尾：最后一片
        if current:
            chunks.append("".join(current))

        # 转成带元数据的 Document
        documents = [
            Document(
                page_content=chunk.strip(),
                metadata={
                    "source": source,
                    "chunk_index": idx,
                    "chunk_size": len(chunk.strip()),
                    "strategy": "semantic",
                },
            )
            for idx, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        logger.info(
            f"{source} 语义切分完成 | "
            f"{len(sentences)} 个句子 → {len(documents)} 个切片"
        )
        return documents

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        把文本切成句子列表，保留句末标点。
        例："今天晴。明天雨！" → ["今天晴。", "明天雨！"]
        """
        # 用捕获组分割，标点会被保留在结果列表里，需要和前一段重新拼接
        parts = self.SENTENCE_PATTERN.split(text)
        sentences: List[str] = []
        buf = ""
        for part in parts:
            if part is None:
                continue
            if self.SENTENCE_PATTERN.fullmatch(part):
                # 这是标点，拼到当前句子末尾，构成完整句子
                buf += part
                if buf.strip():
                    sentences.append(buf)
                buf = ""
            else:
                buf += part
        if buf.strip():
            sentences.append(buf)
        return sentences

    def _build_overlap(self, sentences: List[str]) -> tuple[List[str], int]:
        """
        从已完成切片的尾部，取出累计不超过 chunk_overlap 字符的句子，
        作为下一切片的重叠开头，保持上下文连续。
        """
        if self.chunk_overlap <= 0:
            return [], 0

        overlap_sents: List[str] = []
        overlap_len = 0
        # 从后往前取句子，直到达到 overlap 长度
        for sent in reversed(sentences):
            if overlap_len + len(sent) > self.chunk_overlap and overlap_sents:
                break
            overlap_sents.insert(0, sent)
            overlap_len += len(sent)
        return overlap_sents, overlap_len
