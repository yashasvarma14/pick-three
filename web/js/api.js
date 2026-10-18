// Talking to the backend (api/main.py).

import { API } from "./util.js";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `Something went wrong (${response.status}).`;
    try { message = (await response.json()).detail || message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  return response;
}

export const getConfig = async () => (await request("/api/config")).json();
export const getFeatured = async (category) => (await request(`/api/featured/${category}`)).json();
export const locate = async (where) => (await request("/api/locate", { method: "POST", body: JSON.stringify(where) })).json();
export const getWriteup = async (runId) =>
  (await request("/api/writeup", { method: "POST", body: JSON.stringify({ run_id: runId }) })).json();
export const sendFeedback = async (body) =>
  (await request("/api/feedback", { method: "POST", body: JSON.stringify(body) })).json();

// A search streams one JSON object per line: progress..., then the result.
export async function search(body, onEvent) {
  const response = await request("/api/search", { method: "POST", body: JSON.stringify(body) });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}
