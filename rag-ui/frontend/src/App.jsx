import { useState, useRef, useEffect } from "react";

const API_BASE = "/api";

function formatSource(src) {
  if (!src) return null;
  // 后端返回格式："文件名.pdf · 第一章 > 第三条"
  // 去掉路径前缀，保留条款路径
  const name = src.split("/").pop() || src;
  return name;
}

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  if (isToday) {
    return d.toTimeString().slice(0, 5);
  }
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function SourceTag({ name }) {
  return (
    <span className="source-tag">
      <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
        <rect x="1" y="1" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5"/>
        <line x1="3.5" y1="4" x2="8.5" y2="4" stroke="currentColor" strokeWidth="1.2"/>
        <line x1="3.5" y1="6" x2="8.5" y2="6" stroke="currentColor" strokeWidth="1.2"/>
        <line x1="3.5" y1="8" x2="6.5" y2="8" stroke="currentColor" strokeWidth="1.2"/>
      </svg>
      {name}
    </span>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  const isError = msg.role === "error";

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      {!isUser && (
        <div className="avatar assistant-avatar">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.8"/>
            <path d="M8 12h2l2-4 2 8 2-4h2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      )}
      <div className={`bubble ${isUser ? "user-bubble" : isError ? "error-bubble" : "assistant-bubble"}`}>
        <div className="bubble-text">{msg.content}</div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources">
            <span className="sources-label">来源</span>
            {[...new Set(msg.sources.map(formatSource))].filter(Boolean).map((s, i) => (
              <SourceTag key={i} name={s} />
            ))}
            {msg.confidence != null && (
              <span className="source-tag confidence-tag">
                置信度 {(msg.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
        {msg.route && (
          <div className="sources">
            <span className="sources-label">{msg.sourceType || "Agent"}</span>
            <span className="source-tag">{msg.route}</span>
          </div>
        )}
      </div>
      {isUser && (
        <div className="avatar user-avatar">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="8" r="4" stroke="currentColor" strokeWidth="1.8"/>
            <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
          </svg>
        </div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message-row assistant">
      <div className="avatar assistant-avatar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.8"/>
          <path d="M8 12h2l2-4 2 8 2-4h2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className="bubble assistant-bubble typing-bubble">
        <span /><span /><span />
      </div>
    </div>
  );
}

function ConversationItem({ conv, active, onClick, onDelete }) {
  return (
    <div
      className={`conv-item ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="conv-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2v10z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className="conv-info">
        <div className="conv-title">{conv.title}</div>
        <div className="conv-meta">
          {conv.message_count} 条消息 · {formatTime(conv.updated_at)}
        </div>
      </div>
      <button
        className="conv-delete"
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        title="删除对话"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
          <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
    </div>
  );
}

function Sidebar({ conversations, activeId, onSelect, onCreate, onDelete, indexStatus }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-small">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#6366f1" strokeWidth="1.8" strokeLinejoin="round"/>
            <path d="M2 17l10 5 10-5" stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 12l10 5 10-5" stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>知识库 Agent</span>
        </div>
      </div>

      <button className="new-chat-btn" onClick={onCreate}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        新建对话
      </button>

      <div className="conv-list">
        <div className="conv-list-label">历史对话</div>
        {conversations.length === 0 && (
          <div className="conv-empty">暂无对话</div>
        )}
        {conversations.map((c) => (
          <ConversationItem
            key={c.id}
            conv={c}
            active={c.id === activeId}
            onClick={() => onSelect(c.id)}
            onDelete={() => onDelete(c.id)}
          />
        ))}
      </div>

      <div className="sidebar-footer">
        {indexStatus && (
          <div className="index-status-sidebar">
            <span className={`dot ${indexStatus.total_chunks > 0 ? "active" : "empty"}`} />
            知识库 · {indexStatus.total_chunks ?? 0} 片段
          </div>
        )}
      </div>
    </aside>
  );
}

function IndexStatus({ status }) {
  if (!status) return null;
  return (
    <div className="index-status">
      <span className={`dot ${status.total_chunks > 0 ? "active" : "empty"}`} />
      知识库 · {status.total_chunks ?? 0} 个片段
    </div>
  );
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "你好！我是你的知识库 Agent。知识库已经配置完成，请直接向我提问。" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);
  const [topK, setTopK] = useState(5);
  const bottomRef = useRef();
  const inputRef = useRef();

  // 拉取会话列表
  const fetchConversations = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
      }
    } catch {}
  };

  // 拉取会话详情
  const fetchConversation = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages.map(m => ({
          role: m.role,
          content: m.content,
          sources: m.sources,
          sourceType: m.source_type,
          route: m.route,
        })));
      }
    } catch {}
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/index/status`);
      if (res.ok) setIndexStatus(await res.json());
    } catch {}
  };

  useEffect(() => {
    fetchStatus();
    fetchConversations();
    const t = setInterval(fetchStatus, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // 新建对话
  const createConversation = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setActiveConvId(data.id);
        setMessages([
          { role: "assistant", content: "你好！我是你的知识库 Agent。请直接向我提问。" }
        ]);
        await fetchConversations();
        inputRef.current?.focus();
      }
    } catch {}
  };

  // 切换对话
  const selectConversation = async (id) => {
    if (id === activeConvId) return;
    setActiveConvId(id);
    await fetchConversation(id);
  };

  // 删除对话
  const deleteConversation = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}`, { method: "DELETE" });
      if (res.ok) {
        if (id === activeConvId) {
          setActiveConvId(null);
          setMessages([
            { role: "assistant", content: "你好！我是你的知识库 Agent。请点击左侧「新建对话」开始提问。" }
          ]);
        }
        await fetchConversations();
      }
    } catch {}
  };

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");

    // 如果没有活跃会话，先创建一个
    let convId = activeConvId;
    if (!convId) {
      try {
        const res = await fetch(`${API_BASE}/conversations`, { method: "POST" });
        if (res.ok) {
          const data = await res.json();
          convId = data.id;
          setActiveConvId(data.id);
          // 清空欢迎语
          setMessages([]);
        }
      } catch {
        setMessages(m => [...m, { role: "error", content: "创建会话失败，请检查服务" }]);
        return;
      }
    }

    setMessages(m => [...m, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/conversations/${convId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: topK })
      });
      const data = await res.json();
      if (res.ok) {
        setMessages(m => [...m, {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
          sourceType: data.source_type,
          route: data.route,
          confidence: data.retrieval_confidence
        }]);
        await fetchConversations(); // 刷新列表（更新时间/标题）
      } else {
        setMessages(m => [...m, { role: "error", content: data.detail || "查询失败，请重试" }]);
      }
    } catch {
      setMessages(m => [...m, { role: "error", content: "网络错误，请检查服务是否正常运行" }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const currentConv = conversations.find(c => c.id === activeConvId);

  return (
    <div className="app-layout">
      <Sidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={selectConversation}
        onCreate={createConversation}
        onDelete={deleteConversation}
        indexStatus={indexStatus}
      />

      <div className="main-area">
        <header className="topbar">
          <div className="chat-title">
            {currentConv ? currentConv.title : "欢迎使用"}
          </div>
          <div className="topbar-right">
            <IndexStatus status={indexStatus} />
          </div>
        </header>

        <main className="chat-area">
          <div className="messages">
            {messages.map((m, i) => <Message key={i} msg={m} />)}
            {loading && <TypingIndicator />}
            <div ref={bottomRef} />
          </div>
        </main>

        <footer className="input-area">
          <div className="input-row">
            <div className="input-wrap">
              <textarea
                ref={inputRef}
                className="chat-input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="输入问题，按 Enter 发送…"
                rows={1}
                disabled={loading}
              />
            </div>
            <button className="send-btn" onClick={send} disabled={!input.trim() || loading}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                <path d="M22 2L15 22l-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
          <div className="input-meta">
            <label className="topk-label">
              检索片段数
              <select value={topK} onChange={e => setTopK(Number(e.target.value))}>
                {[3,5,8,10].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <span className="hint">Shift+Enter 换行</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
