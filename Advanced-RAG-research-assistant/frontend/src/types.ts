export type Route = "knowledge_base" | "web_search" | "general";

export interface Source {
  id: number;
  kind: "local" | "web";
  title: string;
  location: string;
  snippet: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  route?: Route | "fallback" | null;
}

export interface Session {
  id: string;
  title: string;
  doc_ids: string[];
  updated_at: string;
}

export interface DocumentRecord {
  doc_id: string;
  title: string;
  origin: "upload" | "arxiv";
  n_chunks: number;
  arxiv_id?: string | null;
}

export interface ArxivResult {
  arxiv_id: string;
  title: string;
  authors: string[];
  summary: string;
  published: string;
}

export interface SessionDetail {
  session: Session;
  messages: Message[];
  documents: DocumentRecord[];
}
