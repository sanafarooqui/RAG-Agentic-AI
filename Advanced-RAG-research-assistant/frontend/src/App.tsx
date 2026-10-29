import { useCallback, useEffect, useState } from "react";
import { api, streamChat } from "./api";
import ChatPanel from "./components/ChatPanel";
import Sidebar from "./components/Sidebar";
import type { DocumentRecord, Message, Route, Session } from "./types";

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);

  const refreshSessions = useCallback(async () => setSessions(await api.listSessions()), []);

  const openSession = useCallback(async (id: string) => {
    const detail = await api.getSession(id);
    setActiveId(id);
    setMessages(detail.messages);
    setDocuments(detail.documents);
    setError("");
  }, []);

  const newSession = useCallback(async () => {
    const session = await api.createSession();
    await refreshSessions();
    await openSession(session.id);
  }, [openSession, refreshSessions]);

  useEffect(() => {
    (async () => {
      try {
        const health = await api.health();
        setWebSearchEnabled(health.web_search);
        const existing = await api.listSessions();
        setSessions(existing);
        if (existing.length) await openSession(existing[0].id);
        else await newSession();
      } catch (e) {
        setError(`Cannot reach the backend: ${(e as Error).message}`);
      }
    })();
  }, [openSession, newSession]);

  const deleteSession = async (id: string) => {
    await api.deleteSession(id);
    const remaining = await api.listSessions();
    setSessions(remaining);
    if (id !== activeId) return;
    if (remaining.length) await openSession(remaining[0].id);
    else await newSession();
  };

  const send = async (text: string, forceRoute?: Route) => {
    if (!activeId || busy || !text.trim()) return;
    setBusy(true);
    setError("");
    setStatus("Thinking");
    setMessages((m) => [
      ...m,
      { role: "user", content: text, sources: [] },
      { role: "assistant", content: "", sources: [] },
    ]);

    const patchLast = (fn: (m: Message) => Message) =>
      setMessages((all) => [...all.slice(0, -1), fn(all[all.length - 1])]);

    await streamChat(activeId, text, forceRoute, {
      onStatus: setStatus,
      onToken: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
      onDone: (message) => patchLast(() => message),
      onError: (detail) => {
        setError(detail);
        setMessages((all) => (all[all.length - 1]?.content ? all : all.slice(0, -1)));
      },
    }).catch((e) => setError((e as Error).message));

    setBusy(false);
    setStatus("");
    refreshSessions();
  };

  const attach = async (action: Promise<DocumentRecord>) => {
    setError("");
    try {
      await action;
      if (activeId) await openSession(activeId);
      refreshSessions();
    } catch (e) {
      setError((e as Error).message);
      throw e;
    }
  };

  const removeDocument = async (docId: string) => {
    if (!activeId) return;
    const session = sessions.find((s) => s.id === activeId);
    await api.setSessionDocuments(
      activeId,
      (session?.doc_ids ?? []).filter((d) => d !== docId),
    );
    await openSession(activeId);
    refreshSessions();
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        documents={documents}
        webSearchEnabled={webSearchEnabled}
        busy={busy}
        onNew={newSession}
        onSelect={openSession}
        onDelete={deleteSession}
        onUpload={(file) => attach(api.upload(activeId!, file))}
        onArxivImport={(arxivId) => attach(api.arxivImport(arxivId, activeId!))}
        onRemoveDocument={removeDocument}
        onWebSearch={(q) => send(q, "web_search")}
      />
      <ChatPanel
        messages={messages}
        busy={busy}
        status={status}
        error={error}
        hasDocuments={documents.length > 0}
        onSend={(text) => send(text)}
      />
    </div>
  );
}
