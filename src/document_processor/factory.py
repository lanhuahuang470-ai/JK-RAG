"""
切分策略工厂

根据配置（CHUNK_STRATEGY）返回对应的文档处理器：
    recursive    → DocumentProcessor      固定长度递归切片（默认，通用）
    semantic     → SemanticProcessor      语义切片（按句子边界）
    hierarchical → HierarchicalProcessor  层次切片（按章/条/标题）

用法：
    from src.document_processor import get_processor
    processor = get_processor()          # 自动读取 .env 里的 CHUNK_STRATEGY
    docs = processor.process_file("x.pdf")

    # 或显式指定策略
    processor = get_processor("hierarchical")
"""
from __future__ import annotations

from typing import Optional

from config.settings import settings
from src.document_processor.processor import DocumentProcessor
from src.document_processor.semantic_processor import SemanticProcessor
from src.document_processor.hierarchical_processor import HierarchicalProcessor
from src.utils import get_logger

logger = get_logger(__name__)

# 策略名 → 处理器类
_STRATEGY_MAP = {
    "recursive": DocumentProcessor,
    "semantic": SemanticProcessor,
    "hierarchical": HierarchicalProcessor,
}


def get_processor(strategy: Optional[str] = None) -> DocumentProcessor:
    """
    根据策略名返回对应的文档处理器实例。

    Args:
        strategy: 策略名（recursive/semantic/hierarchical）。
                  为 None 时读取配置 settings.chunk_strategy。

    Returns:
        对应的处理器实例（都继承自 DocumentProcessor，接口一致）
    """
    name = (strategy or settings.chunk_strategy or "recursive").lower().strip()

    processor_cls = _STRATEGY_MAP.get(name)
    if processor_cls is None:
        logger.warning(
            f"未知的切分策略 '{name}'，可选：{list(_STRATEGY_MAP)}。"
            f"已回退为 recursive。"
        )
        processor_cls = DocumentProcessor
        name = "recursive"

    logger.info(f"使用切分策略：{name}（{processor_cls.__name__}）")
    return processor_cls()
