from engines.base import ChatEngine

class RagEngine(ChatEngine):
 
 def __init__(self, rag_chain_with_history, history_store):
  self.chain = rag_chain_with_history
  self.history_store = history_store # maps session_id -> ChatMessageHistory

 def answer(self, session_id: str, question: list[str]) -> str:

  resp = self.chain.invoke(
   {"question": question , "history":[]},
  config={"configurable": {"session_id": session_id}},
   )
  return resp
 
 def get_history(self, session_id: str):
  h = self.history_store.get(session_id)
  if not h: return []
  return [("user", m.content) if m.type == "human" else ("assistant",
  m.content) for m in h.messages]