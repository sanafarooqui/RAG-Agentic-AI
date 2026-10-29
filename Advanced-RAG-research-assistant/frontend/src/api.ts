import type { ArxivResult, DocumentRecord, Message, Route, Session, SessionDetail } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; web_search: boolean }>("/api/health"),
  listSessions: () => request<Session[]>("/api/sessions"),
  createSession: () => request<Session>("/api/sessions", { method: "POST" }),
  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  deleteSession: (id: string) => request(`/api/sessions/${id}`, { method: "DELETE" }),
  setSessionDocuments: (id: string, doc_ids: string[]) =>
    request<Session>(`/api/sessions/${id}/documents`, json("PUT", { doc_ids })),
  upload: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("file", file);
    return request<DocumentRecord>("/api/documents/upload", { method: "POST", body: form });
  },
  arxivSearch: (query: string) =>
    request<ArxivResult[]>("/api/arxiv/search", json("POST", { query })),
  arxivImport: (arxiv_id: string, session_id: string) =>
    request<DocumentRecord>("/api/arxiv/import", json("POST", { arxiv_id, session_id })),
};

export interface ChatHandlers {
  onStatus: (label: string) => void;
  onToken: (text: string) => void;
  onDone: (message: Message, title: string) => void;
  onError: (detail: string) => void;
}

export async function streamChat(
  sessionId: string,
  message: string,
  forceRoute: Route | undefined,
  h: ChatHandlers,
) {
  const res = await fetch(`/api/sessions/${sessionId}/chat`, json("POST", { message, force_route: forceRoute }));
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    h.onError(body.detail ?? `Request failed (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (!data.length) continue;
      const payload = JSON.parse(data.join("\n"));
      if (event === "status") h.onStatus(payload.label);
      else if (event === "token") h.onToken(payload.text);
      else if (event === "done") h.onDone(payload.message, payload.title);
      else if (event === "error") h.onError(payload.detail);
    }
  }
}
