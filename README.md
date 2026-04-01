This project is a simple web application that allows users to introduce an arbitrary text into an input field, and receive a joke whose punchline is the given text.

## Stack

- **Backend**: FastAPI (single serverless function)
- **LLM**: Anthropic Claude (`claude-sonnet-4-6`)
- **Observability**: Langfuse v4
- **Frontend**: Vanilla HTML/CSS/JS, Bauhaus palette, Cutive Mono
- **Deployment**: Vercel

## Local dev

```bash
uv venv -p python3.11 && source .venv/bin/activate
uv sync
uvicorn api.index:app --reload
```

Open http://localhost:8000

## Deploy

```bash
vercel deploy
```

Set env vars in Vercel dashboard:
- `ANTHROPIC_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_BASE_URL`
