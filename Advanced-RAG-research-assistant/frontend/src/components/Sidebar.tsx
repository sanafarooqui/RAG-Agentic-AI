import { useRef, useState } from "react";
import { api } from "../api";
import type { ArxivResult, DocumentRecord, Session } from "../types";

interface Props {
  sessions: Session[];
  activeId: string | null;
  documents: DocumentRecord[];
  webSearchEnabled: boolean;
  busy: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onUpload: (file: File) => Promise<void>;
  onArxivImport: (arxivId: string) => Promise<void>;
  onRemoveDocument: (docId: string) => void;
  onWebSearch: (query: string) => void;
}

export default function Sidebar(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [arxivQuery, setArxivQuery] = useState("");
  const [arxivResults, setArxivResults] = useState<ArxivResult[]>([]);
  const [arxivBusy, setArxivBusy] = useState<string | null>(null);
  const [arxivNote, setArxivNote] = useState("");
  const [webQuery, setWebQuery] = useState("");

  const handleFile = async (file?: File) => {
    if (!file || !p.activeId) return;
    setUploading(true);
    try {
      await p.onUpload(file);
    } catch {
      /* surfaced by App */
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const searchArxiv = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!arxivQuery.trim()) return;
    setArxivBusy("search");
    setArxivNote("");
    try {
      const results = await api.arxivSearch(arxivQuery.trim());
      setArxivResults(results);
      if (!results.length) setArxivNote("No papers found");
    } catch (err) {
      setArxivNote((err as Error).message);
    } finally {
      setArxivBusy(null);
    }
  };

  const importArxiv = async (id: string) => {
    setArxivBusy(id);
    try {
      await p.onArxivImport(id);
      setArxivResults([]);
      setArxivQuery("");
    } catch {
      /* surfaced by App */
    } finally {
      setArxivBusy(null);
    }
  };

  const submitWeb = (e: React.FormEvent) => {
    e.preventDefault();
    if (!webQuery.trim()) return;
    p.onWebSearch(webQuery.trim());
    setWebQuery("");
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <h1>Research Assistant</h1>
        <button className="primary" onClick={p.onNew} disabled={p.busy}>
          + New
        </button>
      </div>

      <section>
        <h2>Documents</h2>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.md,.markdown,.txt"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <button className="block" onClick={() => fileRef.current?.click()} disabled={uploading || !p.activeId}>
          {uploading ? "Processing…" : "Upload PDF / MD / TXT"}
        </button>
        <ul className="docs">
          {p.documents.map((d) => (
            <li key={d.doc_id} title={d.title}>
              <span className="badge">{d.origin === "arxiv" ? "arXiv" : "file"}</span>
              <span className="truncate">{d.title}</span>
              <button className="icon" aria-label={`Remove ${d.title}`} onClick={() => p.onRemoveDocument(d.doc_id)}>
                ×
              </button>
            </li>
          ))}
          {!p.documents.length && <li className="muted">No documents in this chat yet</li>}
        </ul>
      </section>

      <section>
        <h2>ArXiv</h2>
        <form onSubmit={searchArxiv} className="row">
          <input
            value={arxivQuery}
            onChange={(e) => setArxivQuery(e.target.value)}
            placeholder="Title or ID (1706.03762)"
          />
          <button disabled={arxivBusy !== null}>{arxivBusy === "search" ? "…" : "Find"}</button>
        </form>
        {arxivNote && <p className="muted">{arxivNote}</p>}
        <ul className="arxiv-results">
          {arxivResults.map((r) => (
            <li key={r.arxiv_id}>
              <div className="truncate-2" title={r.title}>
                {r.title}
              </div>
              <div className="muted small">
                {r.authors.join(", ")} · {r.published}
              </div>
              <button disabled={arxivBusy !== null || !p.activeId} onClick={() => importArxiv(r.arxiv_id)}>
                {arxivBusy === r.arxiv_id ? "Importing…" : "Add to chat"}
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2>Web search</h2>
        <form onSubmit={submitWeb} className="row">
          <input
            value={webQuery}
            onChange={(e) => setWebQuery(e.target.value)}
            placeholder={p.webSearchEnabled ? "Search the web…" : "Set TAVILY_API_KEY to enable"}
            disabled={!p.webSearchEnabled || p.busy}
          />
          <button disabled={!p.webSearchEnabled || p.busy}>Go</button>
        </form>
      </section>

      <section className="history">
        <h2>Chat history</h2>
        <ul>
          {p.sessions.map((s) => (
            <li key={s.id} className={s.id === p.activeId ? "active" : ""}>
              <button className="session" onClick={() => p.onSelect(s.id)} disabled={p.busy}>
                <span className="truncate">{s.title}</span>
              </button>
              <button className="icon" aria-label="Delete chat" onClick={() => p.onDelete(s.id)} disabled={p.busy}>
                ×
              </button>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
