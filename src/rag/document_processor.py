"""
层次切片文档处理器（Hierarchical Chunking）

支持 PDF / DOCX 文档的文本提取与层次化切分：
    章（第X章） → 节（第X节） → 条（第X条） → 款/项
适合规章制度、法律法规、标准规范等结构清晰的文档。

核心优势：
1. 每个切片尽量对应一个完整的"条款"，语义边界天然完整
2. 每个切片携带层次元数据（属于哪章/哪节/哪条），检索来源更精确
3. 超长条款自动回退到字符级递归切分，避免单片过大
4. 无层次结构的文档自动回退到递归字符切分
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import PyPDF2
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    层次切片文档处理器

    职责：
    1. 从 PDF / DOCX 提取纯文本
    2. 清洗文本（去除噪声字符）
    3. 按文档层次结构切分 → List[Document]
    4. 无层次结构时自动回退到递归字符切分
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

    # ── 层次标题的正则模式（按从大到小的层级顺序）────────────────────────────
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size or int(chunk_size * 1.5)

        # 用于超长条款的回退切分器
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

        logger.info(
            "DocumentProcessor (层次切分) 初始化完成 | "
            f"max_chunk_size={self.max_chunk_size} | "
            f"回退 chunk_size={chunk_size} | overlap={chunk_overlap}"
        )

    # ── 公共接口 ───────────────────────────────────────────────────────────

    def process_file(self, file_path: str | Path) -> List[Document]:
        """
        处理单个文件，返回切分后的 LangChain Document 列表

        Args:
            file_path: 文件路径（PDF / DOCX）

        Returns:
            List[Document]：每个 Document 包含 page_content 和 metadata
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式：{suffix}。"
                f"当前支持：{self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"开始处理文件：{path.name}")

        if suffix == ".pdf":
            raw_text = self._extract_pdf(path)
        else:  # .docx / .doc
            raw_text = self._extract_docx(path)

        cleaned = self._clean_text(raw_text)
        chunks = self._split(cleaned, source=path.name)

        logger.info(f"文件 {path.name} 处理完成 | 共切分 {len(chunks)} 个 chunk")
        return chunks

    def process_files(self, file_paths: List[str | Path]) -> List[Document]:
        """批量处理多个文件"""
        all_docs: List[Document] = []
        for fp in file_paths:
            try:
                all_docs.extend(self.process_file(fp))
            except Exception as e:
                logger.error(f"处理文件 {fp} 失败：{e}")
        logger.info(f"批量处理完成 | 共 {len(file_paths)} 个文件 | {len(all_docs)} 个 chunk")
        return all_docs

    # ── PDF 提取 ──────────────────────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> str:
        """使用 PyPDF2 逐页提取 PDF 文本"""
        texts: List[str] = []
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                logger.debug(f"PDF 共 {total_pages} 页")

                for page_num, page in enumerate(reader.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            texts.append(page_text)
                    except Exception as e:
                        logger.warning(f"第 {page_num} 页提取失败：{e}")

        except PyPDF2.errors.PdfReadError as e:
            raise RuntimeError(f"PDF 读取错误（可能是加密文件）：{e}") from e

        full_text = "\n".join(texts)
        logger.debug(f"PDF 提取完成 | 原始字符数：{len(full_text)}")
        return full_text

    # ── DOCX 提取 ─────────────────────────────────────────────────────────

    def _extract_docx(self, path: Path) -> str:
        """使用 python-docx 提取 Word 文档文本（含表格）"""
        try:
            doc = DocxDocument(str(path))
        except Exception as e:
            raise RuntimeError(f"DOCX 读取错误：{e}") from e

        parts: List[str] = []

        # 提取正文段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    parts.append(row_text)

        full_text = "\n".join(parts)
        logger.debug(f"DOCX 提取完成 | 原始字符数：{len(full_text)}")
        return full_text

    # ── 文本清洗 ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        文本清洗：
        - 去除控制字符（保留换行/制表符）
        - 合并多余空白行
        - 去除首尾空白
        """
        text = re.sub(r"\x0c", "\n", text)          # 换页符 → 换行
        text = re.sub(r"[^\S\n]+", " ", text)        # 多个空格合并
        text = re.sub(r"\n{3,}", "\n\n", text)        # 多个空行合并为双换行
        text = text.strip()
        return text

    # ── 层次切分（核心逻辑）───────────────────────────────────────────────

    def _split(self, text: str, source: str) -> List[Document]:
        """
        层次切分：按章/节/条边界切分，并附加层次元数据。

        流程：
        1. 逐行扫描，识别层次标题，把文本切成"层次块"
        2. 每个层次块作为一个候选切片，记录它所属的章/节/条
        3. 超长的块回退到字符级递归切分
        4. 无层次结构时回退到普通递归切分
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

        # 如果整篇没有任何层次标题，退化为普通递归切分
        if not blocks or all(not b["context"] for b in blocks):
            logger.info(f"{source} 未检测到层次结构，回退为递归切分")
            return self._fallback_split(text, source)

        # 把每个层次块转成 Document（超长则再切）
        documents = self._blocks_to_documents(blocks, source)
        logger.info(
            f"{source} 层次切分完成 | 识别 {len(blocks)} 个层次块 "
            f"→ {len(documents)} 个切片"
        )
        return documents

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

    def _fallback_split(self, text: str, source: str) -> List[Document]:
        """回退切分：无层次结构时使用普通递归字符切分"""
        raw_chunks = self._fallback_splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                    "hierarchy_path": "",
                },
            )
            for idx, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]
        return documents

    @staticmethod
    def _extract_rule_id(context: Dict[str, str]) -> str:
        """
        从层次上下文提取规则 ID：'第一章' + '第三条' → '1-3'
        '第二章' + '第十四条' → '2-14'；无章或条信息时返回空
        """
        digit = {"一":1,"二":2,"三":3,"四":4,"五":5,
                 "六":6,"七":7,"八":8,"九":9}

        def _cn_to_int(s: str) -> int:
            if s == "十": return 10
            if s.startswith("十"): return 10 + digit.get(s[1], 0)
            if s.endswith("十"): return digit.get(s[0], 0) * 10
            if "十" in s:
                parts = s.split("十")
                return digit.get(parts[0], 0) * 10 + digit.get(parts[1], 0)
            return digit.get(s, 0)

        import re
        ch_num = art_num = ""
        chapter = context.get("章", "")
        if chapter:
            m = re.search(r"第([一二三四五六七八九十]+)章", chapter)
            if m: ch_num = str(_cn_to_int(m.group(1)))
        article = context.get("条", "")
        if article:
            m = re.search(r"第([一二三四五六七八九十]+)条", article)
            if m: art_num = str(_cn_to_int(m.group(1)))
        return f"{ch_num}-{art_num}" if ch_num and art_num else ""

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
            "rule_id": DocumentProcessor._extract_rule_id(context),  # 如 "1-3"
        }
        # 把章/节/条分别存为独立字段，便于按层级过滤检索
        for level, title in context.items():
            metadata[f"level_{level}"] = title

        return Document(page_content=text, metadata=metadata)
