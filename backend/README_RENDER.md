# Deploy the LLM (FastAPI) to Render

This folder contains a REST API (FastAPI) in `conv.py`.

## Render setup (Python Web Service)

1. Create a **new Render Web Service** from your repo.
2. **Root Directory**: leave empty (repo root).
3. **Build Command**:

```bash
pip install -r backend/requirements_render.txt
```

4. **Start Command**:

```bash
python -m uvicorn backend.conv:app --host 0.0.0.0 --port $PORT
```

5. Add environment variables:
- `COHERE_API_KEY` = your Cohere key

Optional (recommended):
- `CORS_ORIGINS` = comma-separated list of allowed origins (your Vercel domains)
  - Example: `https://synergiq.vercel.app,https://synergiq-git-main-yourname.vercel.app`

## Frontend setup (Vercel)

In Vercel → Project → Settings → Environment Variables:

- `VITE_API_URL` = your Node/Express API on Render, including `/api`
  - Example: `https://<your-node-service>.onrender.com/api`

- `VITE_LLM_API_URL` = your FastAPI LLM service on Render (NO `/api`)
  - Example: `https://<your-llm-service>.onrender.com`

Then redeploy the Vercel project.

## Notes
- Verify the service is up via `GET /health`.
