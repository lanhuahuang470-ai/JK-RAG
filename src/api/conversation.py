"""
会话管理模块
支持多轮对话的会话存储（内存实现，重启丢失）

功能：
- 创建新会话
- 获取会话列表
- 获取会话详情（历史消息）
- 追加消息到会话
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional


class ChatMessage:
    """单条聊天消息"""

    def __init__(
        self,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        source_type: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:
        self.role = role  # user / assistant
        self.content = content
        self.sources = sources or []
        self.source_type = source_type
        self.route = route
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "source_type": self.source_type,
            "route": self.route,
            "timestamp": self.timestamp,
        }


class Conversation:
    """单次会话（多轮对话）"""

    def __init__(self, title: str = "新对话") -> None:
        self.id = str(uuid.uuid4())[:8]  # 短 ID，方便展示
        self.title = title
        self.messages: List[ChatMessage] = []
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self.updated_at = time.time()
        # 第一条用户消息作为会话标题
        if len(self.messages) == 1 and message.role == "user":
            self.title = message.content[:20] + ("..." if len(message.content) > 20 else "")

    def to_dict(self, with_messages: bool = True) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }
        if with_messages:
            data["messages"] = [m.to_dict() for m in self.messages]
        return data


class ConversationManager:
    """会话管理器（内存存储）"""

    def __init__(self, max_conversations: int = 50) -> None:
        self._conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations

    def create(self, title: str = "新对话") -> Conversation:
        """创建新会话"""
        conv = Conversation(title=title)
        self._conversations[conv.id] = conv
        self._trim()
        return conv

    def get(self, conv_id: str) -> Optional[Conversation]:
        """获取会话"""
        return self._conversations.get(conv_id)

    def list(self) -> List[Dict[str, Any]]:
        """获取会话列表（按更新时间倒序）"""
        convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True,
        )
        return [c.to_dict(with_messages=False) for c in convs]

    def delete(self, conv_id: str) -> bool:
        """删除会话"""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            return True
        return False

    def _trim(self) -> None:
        """超过最大数量时删除最旧的会话"""
        if len(self._conversations) <= self.max_conversations:
            return
        sorted_convs = sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
        )
        to_delete = len(self._conversations) - self.max_conversations
        for conv in sorted_convs[:to_delete]:
            del self._conversations[conv.id]


# 全局单例
conversation_manager = ConversationManager()
