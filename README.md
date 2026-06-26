# MigrateMate

MigrateMate converts Flask apps to FastAPI. Point it at a GitHub repo or upload a
ZIP, and it rewrites your routes, request handling, and imports using Google
Gemini, then hands back a ZIP of the FastAPI version. It watches each step as it
runs and leaves `# TODO` comments where a human needs to take a look.

## What it converts

- `Blueprint(...)` to `APIRouter(...)` (keeps your original variable names)
- `@app.route(..., methods=[...])` to `@app.get` / `@app.post` / etc.
- `request.json` / `request.data` to `await request.json()` / `await request.body()`
  and adds `request: Request` to the function signature when it's used
- `jsonify(...)` removed (FastAPI returns dicts directly)
- `make_response(...)` to `JSONResponse(...)`
- Flask-JWT-Extended, Flask-Caching, and Flask-SocketIO turned into FastAPI
  equivalents or clearly marked TODOs

Files that aren't Flask-related are copied through unchanged.

## How it works

1. Parse each Python file into chunks (functions, classes, top-level code).
2. Send each Flask chunk to Gemini with a focused prompt.
3. Clean up the output (dedupe imports, fix decorators, organize the file).
4. Package everything into a ZIP with a FastAPI `requirements.txt`.

If no Gemini key is set, it falls back to a pattern-based converter so the tool
still works offline.

## Quick start

### Prerequisites

- Python 3.10+
- Node.js 18+ (frontend)
- A Google Gemini API key ([get one here](https://makersuite.google.com/app/apikey))
- Docker is optional. The core migration needs only the Gemini key; the backend
  starts in about a second without it. Postgres and the vector-search stack are
  off by default and only matter if you set `ENABLE_DATABASE=true` or
  `ENABLE_VECTOR_SERVICES=true` in `.env`.

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env           # then add your GEMINI_API_KEY

uvicorn app.main:app --reload --port 8000
```

Run the tests (offline, no key or Docker needed):

```bash
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

### Optional services (Postgres / Qdrant / Neo4j / Redis)

Only needed with `ENABLE_DATABASE=true` or `ENABLE_VECTOR_SERVICES=true`. The
vector stack also needs the heavier dependencies:

```bash
cd backend
docker compose up -d
pip install -r requirements-rag.txt
```

## Using it

**GitHub:** paste a repo URL (e.g. `https://github.com/username/flask-app`),
choose a branch, and start. The repo is cloned, scanned for Python files, and
migrated.

**ZIP upload:** drop in a ZIP of your Flask project. Same pipeline, local files.

Either way you get a downloadable ZIP with the migrated code, your original file
structure, a FastAPI `requirements.txt`, and TODO comments where manual review is
needed (auth setup, middleware, websockets).

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/batch/github` | Start a GitHub migration (returns a `migration_id`) |
| POST | `/api/v1/batch/upload` | Upload a ZIP (returns a `migration_id`) |
| GET | `/api/v1/batch/status/{id}` | Poll step-by-step progress |
| GET | `/api/v1/batch/result/{id}` | Results plus a download link |
| GET | `/api/v1/batch/download/{id}` | Download the migrated ZIP |

Both start endpoints run in the background and return `202` with a
`migration_id`. Poll `status/{id}` for the live step list
(`queued → fetch → analyze → migrate → package → completed`). Downloads are
resolved from the id on the server, so there's no way to request an arbitrary
file path.

## Configuration

Everything lives in `backend/.env` (see `.env.example` for all options):

```env
GEMINI_API_KEY=your_api_key_here

# Optional, all off by default:
ENABLE_DATABASE=false          # Postgres-backed project store
ENABLE_VECTOR_SERVICES=false   # embeddings + Qdrant + Neo4j
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

The frontend reads `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`).

## Project structure

```
migratemate/
├── backend/
│   ├── app/
│   │   ├── agents/migration_agent.py     # Gemini prompts + conversion
│   │   ├── services/
│   │   │   ├── migration_service.py      # orchestrates the migration
│   │   │   ├── job_store.py              # per-migration progress + steps
│   │   │   ├── code_parser.py            # AST-based parser
│   │   │   └── github_service.py         # clone + file discovery
│   │   ├── api/v1/batch.py              # migration endpoints
│   │   └── main.py
│   ├── tests/                           # offline test suite
│   └── requirements.txt
├── frontend/
│   └── src/components/
│       ├── MigrationPanel.tsx           # main UI + progress
│       └── CodeDiffViewer.tsx           # before/after diff
└── docker-compose.yml                   # optional services
```

## Tech stack

- Backend: FastAPI, Uvicorn, Google Gemini 2.0 Flash, Python AST
- Frontend: Next.js 14, TypeScript, Tailwind CSS, Monaco editor, react-diff-viewer

## Limitations

- WebSocket conversion is partial: SocketIO gets a `ConnectionManager` scaffold
  but needs manual wiring.
- Complex middleware chains are flagged with TODOs rather than fully converted.
- The API has no auth; it's a local dev tool, not production-ready.
- Large projects can hit Gemini rate limits and may need a second run.

## License

MIT.

Built by [Aditya Singh](https://github.com/asing508).
