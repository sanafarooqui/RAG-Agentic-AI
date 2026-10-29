import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  busy: boolean;
  status: string;
  error: string;
  hasDocuments: boolean;
  onSend: (text: string) => void;
}

const ROUTE_LABEL: Record<string, { text: string; cls: string }> = {
  knowledge_base: { text: "From your documents", cls: "local" },
  web_search: { text: "From the web", cls: "web" },
  general: { text: "Model knowledge — no sources", cls: "general" },
  fallback: { text: "Insufficient evidence", cls: "fallback" },
};

function MessageBubble({ message }: { message: Message }) {
  const route = message.route ? ROUTE_LABEL[message.route] : undefined;
  return (
    <div className={`message ${message.role}`}>
      {message.role === "assistant" && route && <span className={`route ${route.cls}`}>{route.text}</span>}
      <div className="content">
        <ReactMarkdown>{message.content || "…"}</ReactMarkdown>
      </div>
      {message.sources.length > 0 && (
        <details className="sources">
          <summary>{message.sources.length} sources</summary>
          <ol>
            {message.sources.map((s) => (
              <li key={s.id} id={`src-${s.id}`}>
                <span className={`badge ${s.kind}`}>{s.kind === "web" ? "web" : "doc"}</span>{" "}
                {s.kind === "web" && s.location ? (
                  <a href={s.location} target="_blank" rel="noreferrer">
                    {s.title}
                  </a>
                ) : (
                  <strong>{s.title}</strong>
                )}
                {s.kind === "local" && s.location && <span className="muted"> · {s.location}</span>}
                <p className="snippet">{s.snippet}</p>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}

export default function ChatPanel({ messages, busy, status, error, hasDocuments, onSend }: Props) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <main className="chat">
      <div className="messages">
        {!messages.length && (
          <div className="empty">
            <h2>Ask about your research papers</h2>
            <p>
              {hasDocuments
                ? "Questions are answered from your documents, with the web as a fallback."
                : "Upload a paper or add one from ArXiv, or just ask a question to search the web."}
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        {busy && <div className="status">{status}…</div>}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </div>
      <form className="composer" onSubmit={submit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask a question…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
        />
        <button className="primary" disabled={busy || !text.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
