import { useState, useRef, useEffect } from "react";

const API_BASE = "/api";

function formatSource(src) {
  if (!src) return null;
  const name = src.split("/").pop() || src;
  return name;
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

function UploadPanel({ onClose }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setStatus(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setStatus({ ok: true, msg: `「${file.name}」已成功加入知识库` });
        setFile(null);
      } else {
        setStatus({ ok: false, msg: data.detail || "上传失败，请重试" });
      }
    } catch {
      setStatus({ ok: false, msg: "网络错误，请检查服务是否运行" });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-panel">
      <div className="upload-header">
        <span>上传文档</span>
        <button className="icon-btn" onClick={onClose}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>
      </div>
      <div
        className={`drop-zone ${file ? "has-file" : ""}`}
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current.click()}
      >
        <input ref={inputRef} type="file" accept=".pdf,.docx,.doc" hidden onChange={e => setFile(e.target.files[0])} />
        {file ? (
          <div className="file-selected">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" stroke="#6366f1" strokeWidth="1.8"/>
              <path d="M14 2v6h6" stroke="#6366f1" strokeWidth="1.8"/>
            </svg>
            <span>{file.name}</span>
            <small>{(file.size / 1024).toFixed(0)} KB</small>
          </div>
        ) : (
          <div className="drop-hint">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
              <path d="M12 15V3m0 0L8 7m4-4l4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
              <path d="M3 15v4a2 2 0 002 2h14a2 2 0 002-2v-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            </svg>
            <p>拖拽文件至此，或点击选择</p>
            <small>支持 PDF、Word 文档</small>
          </div>
        )}
        {msg.route && (
          <div className="sources">
            <span className="sources-label">{msg.sourceType || "Agent"}</span>
            <span className="source-tag">{msg.route}</span>
          </div>
        )}
      </div>
      {status && (
        <div className={`upload-status ${status.ok ? "ok" : "fail"}`}>{status.msg}</div>
      )}
      <button className="upload-btn" onClick={handleUpload} disabled={!file || uploading}>
        {uploading ? "处理中…" : "加入知识库"}
      </button>
    </div>
  );
}

function IndexStatus({ status }) {
  if (!status) return null;
  return (
    <div className="index-status">
      <span className={`dot ${status.total_chunks > 0 ? "active" : "empty"}`} />
      知识库 · {status.total_chunks ?? 0} 个片段 · {status.document_count ?? 0} 份文档
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "你好！我是你的知识库 Agent。知识库已经配置完成，请直接向我提问。" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [indexStatus, setIndexStatus] = useState(null);
  const [topK, setTopK] = useState(5);
  const bottomRef = useRef();
  const inputRef = useRef();

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/index/status`);
      if (res.ok) setIndexStatus(await res.json());
    } catch {}
  };

  useEffect(() => {
    fetchStatus();
    const t = setInterval(fetchStatus, 30000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages(m => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/query`, {
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
          route: data.route
        }]);
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

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#6366f1" strokeWidth="1.8" strokeLinejoin="round"/>
            <path d="M2 17l10 5 10-5" stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2 12l10 5 10-5" stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>企业知识库 Agent</span>
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
  );
}
