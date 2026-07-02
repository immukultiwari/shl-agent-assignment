from fastapi import FastAPI, HTTPException
import os
import threading
from importlib import import_module
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.agent import handle_chat

app = FastAPI(title="SHL Assessment Recommender API")
executor = ThreadPoolExecutor(max_workers=4)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>SHL Assessment Recommender</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; }
    .container { max-width: 760px; margin: 32px auto; padding: 24px; background: #fff; border-radius: 12px; box-shadow: 0 12px 24px rgba(0,0,0,0.08); }
    h1 { margin-top: 0; }
    .chat-log { min-height: 280px; margin-bottom: 16px; border: 1px solid #d2d8df; border-radius: 8px; padding: 16px; background: #fafbfc; overflow-y: auto; }
    .message { margin-bottom: 12px; }
    .message.user { text-align: right; }
    .message .bubble { display: inline-block; padding: 12px 16px; border-radius: 16px; max-width: 78%; }
    .message.user .bubble { background: #0078d4; color: #fff; }
    .message.assistant .bubble { background: #eef3fb; color: #141414; }
    textarea { width: 100%; min-height: 90px; padding: 12px; border: 1px solid #cfd8e2; border-radius: 10px; resize: vertical; font-size: 14px; }
    button { margin-top: 12px; background: #0078d4; color: #fff; border: none; padding: 12px 20px; border-radius: 10px; cursor: pointer; font-size: 15px; }
    button:disabled { background: #7aaed8; cursor: not-allowed; }
    .footer { margin-top: 16px; font-size: 13px; color: #606770; }
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>SHL Assessment Recommender</h1>
    <p>Ask the agent for the best SHL assessment. The chat history is sent to the API for context.</p>
    <div class=\"chat-log\" id=\"chatLog\"></div>
    <textarea id=\"userInput\" placeholder=\"Type your message here...\"></textarea>
    <button id=\"sendBtn\">Send</button>
    <div class=\"footer\">API endpoint: <code>POST /chat</code></div>
  </div>
  <script>
    const chatLog = document.getElementById('chatLog');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const history = [];

    function addMessage(role, text) {
      const msg = document.createElement('div');
      msg.className = 'message ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      msg.appendChild(bubble);
      chatLog.appendChild(msg);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
      const content = userInput.value.trim();
      if (!content) return;
      addMessage('user', content);
      history.push({ role: 'user', content });
      userInput.value = '';
      userInput.disabled = true;
      sendBtn.disabled = true;

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: history })
        });
        const data = await response.json();
        if (response.ok) {
          addMessage('assistant', data.reply);
          history.push({ role: 'assistant', content: data.reply });
        } else {
          addMessage('assistant', 'Error: ' + (data.detail || response.statusText));
        }
      } catch (error) {
        addMessage('assistant', 'Network error: ' + error.message);
      }

      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>"""

# --- Request Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# --- Response Models (this is what was missing) ---
class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.on_event("startup")
async def ensure_chroma_db():
  """On startup, if the Chroma DB path is missing or empty, run ingest.py in background.

  This helps demo deployments (free Render) where the filesystem starts empty.
  """
  chroma_path = os.getenv("CHROMA_PATH", "./data/shl_chroma_db")
  try:
    exists = os.path.exists(chroma_path) and bool(os.listdir(chroma_path))
  except Exception:
    exists = False

  if not exists:
    def run_ingest():
      try:
        print(f"Chroma DB at {chroma_path} missing or empty — running ingest.py...")
        # Ensure ingest.py uses the same CHROMA_PATH
        os.environ["CHROMA_PATH"] = chroma_path
        ingest_mod = import_module("ingest")
        ingest_mod.main()
        print("Ingest finished successfully.")
      except Exception as e:
        print("Ingest failed:", e)

    t = threading.Thread(target=run_ingest, daemon=True)
    t.start()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        history = [{"role": msg.role, "content": msg.content} 
                   for msg in request.messages]

        if not history:
            raise HTTPException(status_code=400, detail="Message history cannot be empty.")

        # Fix 7a — Hard turn cap: spec says max 8 turns total
        if len(history) > 8:
            return ChatResponse(
                reply="We've reached the maximum conversation length. Here is the current shortlist.",
                recommendations=[],
                end_of_conversation=True
            )

        current_query = history[-1]["content"]
        past_history = history[:-1]

        # Fix 7b — 25s timeout (5s buffer under evaluator's 30s hard limit)
        try:
            loop = asyncio.get_event_loop()
            agent_response = await asyncio.wait_for(
                loop.run_in_executor(executor, handle_chat, current_query, past_history),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                reply="I'm taking too long to respond. Could you simplify the query?",
                recommendations=[],
                end_of_conversation=False
            )

        return agent_response

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error: {e}")
        raise HTTPException(status_code=500, detail="Internal agent error.")