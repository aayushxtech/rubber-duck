# RubberDuck

RubberDuck is an intelligent, AI-powered debugging assistant designed to help developers solve complex problems using the Socratic method. Instead of just giving answers, it acts as a "Rubber Duck," analyzing your problem, asking clarifying questions, and guiding you toward the solution.

The core logic is built using a state graph (LangGraph) that orchestrates the reasoning process, including analysis, search, socratic questioning, and final response generation.

## Features

* **Socratic Debugging**: Asks probing questions to help you clarify your thoughts and isolate the issue.
* **Intelligent Analysis**: Analyzes your initial query and subsequent replies to understand the context and root cause.
* **Automated Search**: Can perform searches to gather evidence or documentation if needed (integrated into the reasoning loop).
* **Stateful Reasoning**: Maintains conversation state across turns to build a complete picture of the problem.

## Powered By

* **[Groq](https://groq.com/)**: Provides ultra-fast inference for the LLM models driving the reasoning and conversation.
* **[Tavily](https://tavily.com/)**: Powers the automated search capabilities to fetch real-time evidence and documentation.

## Getting Started

### Prerequisites

* Python 3.10+
* Dependencies listed in `requirements.txt`
* **API Keys**: You will need API keys for Groq and Tavily.

### Installation

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3.Set up environment variables:
   Create a `.env` file in the `backend` directory and add your keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Health Check

* **URL**: `/health`
* **Method**: `GET`
* **Description**: Checks if the server is running.
* **Response**: `{"status": "ok"}`

### Start Session

* **URL**: `/api/ducks/start`
* **Method**: `POST`
* **Description**: Initiates a new debugging session with an initial problem statement.
* **Request Body**:

```json
    {
      "initial_question": "My Python script is throwing an IndexError."
    }
```

* **Response**: Returns a socratic question, initial analysis, and session state.

### Reply

* **URL**: `/api/ducks/reply`
* **Method**: `POST`
* **Description**: Continues the session with your answer to the previous socratic question.
* **Request Body**:

```json
    {
      "initial_question": "...",
      "prev_socratic_q": "...",
      "user_reply": "It happens in the for loop iterating over the list.",
      "analysis": { ... },
      "turn_count": 1,
      "evidence": { ... }
    }
```

* **Response**: Returns the next socratic question or a final answer if the problem is resolved.

## Architecture

The backend uses a graph-based architecture defined in `app/graph/`:

* **Analyze Node**: Evaluates the current state and user input.
* **Route Node**: Decides the next step (Ask Question, Search, or Final Answer).
* **Socratic Node**: Generates a clarifying question.
* **Search Node**: Fetches external information if required.
* **Final Respond Node**: Provides the solution or summary.
