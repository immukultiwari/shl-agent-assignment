Deploying to Render

Overview
- This repository runs a FastAPI app that exposes `/` (chat UI), `/health`, and `/chat`.

Recommended Render setup (Web Service)
1. Create a new Web Service on Render:
   - Connect your GitHub repo containing this project.
   - Environment: Python 3 (default)

2. Build command:

```bash
pip install -r requirements.txt
```

3. Start command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT
```

(You can also use the included `Procfile`.)

4. Environment variables (Add in Render dashboard > Environment):
   - `GOOGLE_API_KEY` = your Google API key
   - `CHROMA_PATH` = `/opt/render/project/data/shl_chroma_db` (if you use a persistent disk)

5. Persistent Disk (optional but recommended):
   - If you want ChromaDB to persist between deploys, attach a Persistent Disk in Render and mount it to `/opt/render/project/data`.
   - Ensure the `data/shl_chroma_db` content is present or set `CHROMA_PATH` accordingly.
      - This project will automatically run `ingest.py` on startup if the ChromaDB path is missing or empty. This helps demo deployments on ephemeral filesystems (free tier).
         Note: the ingest process may take a few minutes depending on the catalog size and embedding model.

6. Deploy and monitor logs.

Notes & troubleshooting
- If you hit rate limits from the Gemini API you'll see quota errors (429). Enable billing or use a different project/key.
- The project currently uses the `google-generativeai` client which is deprecated; consider migrating to `google-genai` for future-proofing.

Local test commands

```bash
# Install deps
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8001

# Or run via gunicorn (production-like)
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8001
```
