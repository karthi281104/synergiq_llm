# Deploy the LLM (FastAPI) to Render

This folder contains:
- A Streamlit UI app (`conv.py`)
- A FastAPI service (`backend.py`)

## Render setup for Streamlit (Python Web Service)

1. Create a **new Render Web Service** from your repo.
2. **Root Directory**: leave empty (repo root).
3. **Build Command**:

```bash
pip install -r backend/requirements.txt
```

4. **Start Command**:

```bash
python -m streamlit run backend/conv.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
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
- For Streamlit, you can verify the service is up by opening the Render URL in your browser.
