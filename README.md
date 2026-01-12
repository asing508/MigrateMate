# MigrateMate

MigrateMate is an automated tool that modernizes your Python web applications by converting them from Flask to FastAPI. Instead of rewriting code by hand, you simply upload your project, and the tool uses AI to intelligently translate it. It handles the hard work for you—like updating how your app handles web requests, fixing imports, and making your code run faster with modern features—ensuring the new version works correctly.

Key Simplifications Made:
"Neuro-symbolic" $\rightarrow$ "Combines AI with strict code rules""AST-driven parsing" $\rightarrow$ "Understanding your code's structure""RAG-ready" $\rightarrow$ "Context-aware" (implied)"Deterministic validation" $\rightarrow$ "Double-checks the results""Asynchronous I/O" $\rightarrow$ "Making your code run faster"

## Why I Built This

Code migration is tedious. I've seen teams spend weeks manually converting Flask apps to FastAPI, fixing the same patterns over and over. MigrateMate automates that process using Google's Gemini AI to understand code context and generate proper conversions, not just find-and-replace.

The tool handles real-world Flask patterns:
- Blueprint → APIRouter conversion (preserving your variable names)
- `request.data` / `request.json` → proper async `await request.body()`
- `make_response()` → `JSONResponse`
- Flask-JWT-Extended → python-jose JWT implementation
- SocketIO → FastAPI WebSocket patterns
- Middleware and before_request hooks → TODO comments with migration guidance

## How It Works

1. **Parse** - The code parser breaks your Flask app into logical chunks (functions, classes, imports)
2. **Analyze** - Each chunk is classified (router, controller, middleware, utility)
3. **Migrate** - Gemini AI converts each chunk with context-aware prompts
4. **Post-process** - Cleanup pass fixes common issues, deduplicates imports, organizes code
5. **Package** - Output as a downloadable ZIP with updated requirements.txt

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- Google Gemini API key ([get one here](https://makersuite.google.com/app/apikey))

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open http://localhost:3000 and you're good to go.

## Using MigrateMate

### Option 1: GitHub Repository

Enter a GitHub URL like `https://github.com/username/flask-app.git`, pick a branch, and click migrate. The tool clones the repo, finds all Python files, and processes them.

### Option 2: ZIP Upload

Upload a ZIP file of your Flask project. Same process, just local files instead of cloning.

### What You Get

- A downloadable ZIP with your migrated FastAPI code
- Original file structure preserved
- `requirements.txt` with FastAPI dependencies
- TODO comments where manual review is needed (auth setup, middleware conversion, etc.)

## Project Structure

```
migratemate/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   └── migration_agent.py    # Gemini AI integration & prompts
│   │   ├── services/
│   │   │   ├── migration_service.py  # Orchestrates file-by-file migration
│   │   │   ├── code_parser.py        # AST-based Python parser
│   │   │   └── github_service.py     # Clones repos, finds Python files
│   │   ├── api/v1/
│   │   │   └── batch.py              # REST endpoints for migration
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   └── page.tsx              # Landing page
│   │   └── components/
│   │       ├── MigrationPanel.tsx    # Main migration UI
│   │       └── CodeDiffViewer.tsx    # Side-by-side diff view
│   └── package.json
│
└── docker-compose.yml                 # Optional: PostgreSQL, Redis, Neo4j and qdrant.
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/batch/github` | Start migration from GitHub URL |
| POST | `/api/v1/batch/upload` | Upload ZIP for migration |
| GET | `/api/v1/batch/status/{id}` | Poll migration progress |
| GET | `/api/v1/batch/result/{id}` | Get migration results |
| GET | `/api/v1/batch/download` | Download migrated ZIP |

## Configuration

Create a `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_api_key_here
DATABASE_URL=sqlite:///./migratemate.db  # or PostgreSQL URL
```

## Tech Stack

**Backend:**
- FastAPI + Uvicorn
- Google Gemini 2.0 Flash (for code migration)
- Python AST (for parsing)

**Frontend:**
- Next.js 14 with App Router
- TypeScript
- Tailwind CSS
- Monaco Editor (code viewing)
- react-diff-viewer (side-by-side diffs)

## Current Limitations

This is a portfolio project, so there are some rough edges:

- **WebSocket migration is incomplete** - SocketIO patterns get ConnectionManager scaffolding but need manual wiring
- **Complex middleware chains** need manual review - the tool adds TODOs but doesn't fully convert them
- **No authentication** on the API - this is a local development tool, not production-ready
- **Rate limits** - Gemini API has quotas, large projects may need multiple runs

## What I Learned Building This

- LLM output is unpredictable - you need robust post-processing to catch edge cases
- Chunk-by-chunk migration loses context - future versions should analyze cross-file dependencies
- AST parsing is great for structure, but regex is still useful for quick pattern fixes
- The 80/20 rule applies: 80% of files migrate cleanly, 20% need the most work

## License

MIT - do whatever you want with it.

---

Built by [Aditya Singh](https://github.com/asing508) demonstrating AI-powered code transformation.
