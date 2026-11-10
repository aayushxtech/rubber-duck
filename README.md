# RubberDuck

RubberDuck is an interactive debugging and critical thinking assistant for software engineers. Instead of immediately generating speculative code solutions, RubberDuck analyzes problem descriptions, surfaces hidden assumptions, asks focused Socratic questions, and searches the web for factual evidence when required.

---

## 1. Main Features

- **Socratic Inquiry**: Asks single, plain-English clarifying questions to challenge unexamined assumptions and isolate root causes.
- **Structured Problem Analysis**: Decomposes user inquiries into explicit statements, working assumptions, and unresolved uncertainties.
- **Evidence-Grounded Search**: Automatically queries Tavily for factual claims, deduplicating search results and grounding final responses with source links.
- **Bounded Execution**: Enforces strict limits on internal research iterations (maximum 3) and user clarification turns (maximum 3), guaranteeing clean termination.
- **Interactive Terminal UI**: Full-featured Textual terminal interface with live status updates, conversation history, and quick commands.
- **Stateless API Design**: Clean separation of concerns with client-held state across turns and sanitized error responses.

---

## 2. Technology Stack

- **Runtime**: Python 3.11+
- **API Framework**: FastAPI, Uvicorn, Starlette
- **State Orchestration**: LangGraph, LangChain Core
- **Validation**: Pydantic v2
- **Inference**: Groq API (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for conversation)
- **Web Research**: Tavily Search API
- **Terminal Interface**: Textual, Rich
- **HTTP Client**: HTTPX
- **Testing**: Pytest

---

## 3. High-Level Architecture

The system consists of a Textual CLI client, a FastAPI service layer, and a LangGraph state machine:

```text
[ Developer ]
      │
      ▼
[ Textual CLI (cli/main.py) ]
      │  (HTTPX async POST)
      ▼
[ FastAPI Backend (backend/app/main.py) ]
      │
      ▼
[ LangGraph Engine (backend/app/graph/graph.py) ]
      ├── analyze_node   ──> Groq (openai/gpt-oss-120b)
      ├── route_node     ──> Deterministic state routing
      ├── search_node    ──> Tavily Search API
      ├── socratic_node  ──> Groq (openai/gpt-oss-20b)
      └── final_node     ──> Groq (openai/gpt-oss-20b)
```

For complete details on execution lifecycles, node behaviors, and schema definitions, see the [Architecture Documentation](docs/ARCHITECTURE.md).

---

## 4. Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd rubber-duck
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

To configure a custom backend address for the CLI, set:

```env
RUBBERDUCK_BACKEND_URL=http://localhost:8000
```

---

## 6. Starting the Backend

1. Export environment variables from `backend/.env`:
   ```bash
   set -a
   source backend/.env
   set +a
   ```

2. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Verify the server is running:
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status":"ok"}
   ```

---

## 7. Starting the CLI

Launch the interactive Textual terminal client:

```bash
# Launch with default backend (http://localhost:8000):
python cli/main.py

# Or specify a custom backend URL:
python cli/main.py http://localhost:8000
```

### Available CLI Commands

| Command | Action |
|---|---|
| `/ask <question>` | Start a new reasoning session on a problem |
| `/reply <response>` | Reply to a pending Socratic inquiry |
| `/analysis` | View structured analysis, assumptions, and uncertainties |
| `/evidence` | View collected search snippets and source URLs |
| `/status` | View backend connectivity and active session state |
| `/clear` | Clear conversation history from screen |
| `/help` | Display command guide |
| `/quit` | Exit the CLI application |

When a Socratic question is waiting for a reply, you can type your answer directly into the input box without typing `/reply`.

---

## 8. Running Tests

### Offline Test Suite
Run the full functional and non-functional test suite without external API calls:

```bash
.venv/bin/pytest backend/tests/functional backend/tests/non_functional -v
```

### Live Provider Integration Tests
To test live API integrations with Groq and Tavily, ensure credentials are exported and pass `RUN_LIVE_TESTS=1`:

```bash
set -a
source backend/.env
set +a
RUN_LIVE_TESTS=1 .venv/bin/pytest backend/tests/integration -v
```

---

## 9. Documentation

For an in-depth reference on internal state machines, error recovery, evidence deduplication, and API contracts, refer to:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
