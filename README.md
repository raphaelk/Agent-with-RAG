# Agent with RAG Platform

An enterprise AI Agent orchestrator and Retrieval-Augmented Generation (RAG) platform with a web-based GUI built on Flask, local vector search with Ollama embeddings, multi-source procedural skills, real-time telemetry, and audit logging.

---

## 1. System Overview

The **Agent with RAG Platform** provides an end-to-end framework for autonomous AI agent workflows grounded in factual documents and dynamic tool procedures:

- **Four Switchable Web GUI Views**:
  1. **Chat & Knowledge Synthesis**: Interactive conversation with multi-step cognitive grounding, step-grouped evidence display, dynamic parameter control (Model, Temperature, Max Tokens, RAG Max Chunks), and support for Google AI Studio active text-generation models dynamically queried via API (or custom OpenAI-compatible endpoints).
  2. **Vector DB Ingestion**: Populate document vector databases from web URLs or local folders with configurable chunk size & overlap, duplicate prevention, active storage status, and an Ollama embedding model manager with catalog inspection and guarded switching.
  3. **Telemetry Dashboard**: Monitor prompt, response, error, and token volumes with multi-interval aggregation (1 min, 15 min, 1 hr, 1 day) across flexible time ranges (Last hr, 1 day, Week, Month, Custom date range) rendered via interactive charts.
  4. **Audit Log & Event Drill-down**: Complete audit tracking of all invocations and responses between specified system components (`user`, `agent`, `skill`, `tool`, `vector database`, `ollamavector model`, `external API call`, `prompts sent to and response received from the model`) with Number of Events displayed per conversation, API key masking, and human-readable JSON payload inspection.

- **Dynamic Skill SOP Architecture**:
  - `skills/time-weather-skill/`: Real-time weather conditions and world time via the public Open-Meteo API (no API keys needed).
  - `skills/person-information-skill/`: 20-sample personnel flat-file CSV lookup (`data/registry.csv`) by name, city, country, or job title.
  - `skills/stock-market-skill/`: Real-time market gainers and decliners ranking based on natural language sentiment and inquiry.
  - `skills/document-retriever-skill/`: Dense cosine similarity search against ingested domain knowledge.

- **Private Local Vector Database**:
  - Independent stores for documents (`database/doc_vectors.json`) and skills (`database/skill_vectors.json`).
  - Powered by local Ollama embeddings (`bge-m3:latest`, `all-minilm:latest`, etc.).
  - Automatic duplicate prevention via cryptographic chunk content hashing.

- **Graceful Lifecycle Management**:
  - Checks on startup whether Ollama is already running.
  - Safely terminates Ollama on shutdown *only* if the application started it. Pre-existing system services remain untouched.

---

## 2. Prerequisites & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- [Ollama](https://ollama.com/) installed locally
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation Steps
1. Clone or navigate into the repository root:
   ```bash
   cd Agent-with-RAG-2
   ```

2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull the recommended embedding model in Ollama:
   ```bash
   ollama pull bge-m3:latest
   ```

5. (Optional) Set your Google AI Studio Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```
   *Note: If no API key is provided, the platform automatically utilizes local synthesis and supports custom OpenAI-compatible endpoints.*

---

## 3. How to Start All Services

1. Run the core orchestrator:
   ```bash
   python3 app.py
   ```
2. The orchestrator will automatically:
   - Verify if Ollama is running (and start `ollama serve` if needed).
   - Scan the `skills/` directory and index any new skills.
   - Start the Flask web server at:
     ```
     http://127.0.0.1:5000
     ```
3. Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 4. How to Shutdown All Services

### Option A: From the Web GUI (Recommended)
1. Click the light red **Shutdown** button located in the top-right corner of the header.
2. A warning modal will appear alerting that all users and services started by the app will be affected.
3. Type the exact confirmation phrase:
   ```
   Shutdown the service
   ```
4. The **Confirm** button will activate. Click **Confirm** to trigger a graceful termination.
5. If Ollama was started by the application, it will be shut down cleanly. If Ollama was running beforehand, it will remain online.

### Option B: From the Terminal
Press `CTRL+C` in the terminal running `app.py`. The registered exit handlers will execute cleanup automatically.

---

## 5. User Guide

### 1. Chat & Knowledge Synthesis
- **Agent Selection**:
  - **Custom Agent (Default)**: The multi-turn cognitive loop described in the specification, incorporating skill vector matching, procedural SOP plans, and tool/document loops.
  - **Google ADK LlmAgent**: Autonomous agent powered by `google.adk.agents.LlmAgent` named `"Chat Agent with RAG"`. Operates with bound skill tools and document search, equipped with fine-grained callback telemetry and payload logging.
- **Model Selection**: Defaults to `gemma-4-26b-a4b-it`. On startup, the application fetches active text-generation models directly from the Google AI Studio API and populates the dropdown menu (e.g., `gemma-4-26b-a4b-it`, `gemini-3.6-flash`, `gemini-flash-lite-latest`, etc.), or choose **Custom Model** to enter your own endpoint URL (e.g., `http://127.0.0.1:8000/v1/chat/completions`).
- **Inference & Orchestration Parameters**:
  - **Temperature**: Adjust generation creativity (0.0 to 2.0).
  - **Max Tokens**: Enforced below model maximum limit.
  - **Max RAG Chunks**: Select maximum number of context chunks to retrieve (1 to 10).
  - **Max Turns**: Multi-turn tool execution loop limit (default: 3, clamped between 1 and 10).
  - **Skills Selection Dropdown**:
    - **Vector Store (Default)**: Automatically queries the skill vector database for skills exceeding the **Threshold** (default: 0.5).
    - **LLM Selected**: Prompts the LLM to inspect all available skills in `skills/` and select the most appropriate skill folder.
    - **Individual Skill**: Explicitly select any skill folder found in `skills/` (e.g., `time-weather-skill`, `person-information-skill`, `stock-market-skill`, `document-retriever-skill`).
  - **Doc Threshold**: Set minimum similarity threshold (default: 0.3) for document chunk retrieval in the *Retrieved Context Evidence* card.
- **Multi-Turn Cognitive Loop**:
  - If no skill matches, answers directly using a simple AI assistant prompt.
  - If a skill matches, prompts the LLM for a structured execution plan or directive.
  - If directed, runs procedural tools or searches the document vector store via `skills/document-retriever-skill/tools/document_search_tool.py`.
  - Feeds results back into the model in a feedback loop up to **Max Turns** until the final answer is synthesized with an assistant system prompt.
- **Interacting with Skills & RAG**:
  - *Weather query*: "What is the weather and local time in Tokyo?"
  - *Registry query*: "Who is the Principal AI Engineer and where are they located?"
  - *Stock market query*: "What are the top gaining stocks in the market today?"
  - *Document RAG query*: "What was our total revenue and net income in FY2025 financial report?"
- **Response Detail Box with "Show Logs" Button & Component Bubbles**:
  - Each response includes an anchored detail box containing bubbles displaying the name of the components that generated logs (`Skills`, `Agent`, `RAG`, `Tools`, `LLM`), appropriate icons (`⚡`, `🤖`, `📚`, `🛠️`, `🧠`), and the elapsed time of each step in milliseconds.
  - A toggle button named **"Show Logs"** strictly anchored at the top-right corner toggles between expand and collapse.
  - Clicking **"Show Logs"** expands the collapsible detail box to reveal the full content of every step including its summary and detailed log records with full JSON payloads, equipped with scroll areas if the content is long.
- **Retrieved Context Evidence**:
  - Inspect the right-hand card to review retrieved evidence grouped by step (**Skill Search** and **Document Search**) with similarity confidence scores and raw snippets. Filtered by the configured **Doc Threshold**.

### 2. Vector DB Ingestion
- **Ingesting Local Docs**: Enter `sample_docs` in the target input and select "Local Directory / File", then click **Populate Vector Database**.
- **Ingesting Web URLs**: Click any of the 5 quick sample URL buttons (or enter a custom URL) and click **Populate Vector Database**.
- **Advanced Chunking**: Expand the collapsed accordion to customize Chunk Size and Overlap characters.
- **Embedding Model Switching**: Use the Embedder dropdown to switch active embedding models. Changing models prompts for confirmation (`Delete Data and Switch`) and automatically re-indexes skills.
- **Reset DB**: Click **Reset DB** on the right card to wipe all document vectors.

### 3. Telemetry Dashboard
- **Metric Cards**: Real-time totals for Prompts, Responses, Errors, Input Tokens, and Output Tokens.
- **Interactive Charts**:
  - Left Chart: Prompts, Responses, and Errors time series.
  - Right Chart: Input and Output Token volume over time.
  - Controls: Toggle aggregation intervals (`1 min`, `15 min`, `1 hr`, `1 day`) and time windows (`Last hr`, `1 day`, `Week`, `Month`, or `Custom` date ranges).
  - Model Filter: Filter graphs by a specific model or view aggregated data across all models.

### 4. Audit Log & Event Trace
- **Conversation Explorer**: Browse all user conversations in reverse chronological order with local timestamps, User Query, Agent Response, and the exact **Number of Events (occured during the conversation)**. Click any conversation row to view its trace.
- **Event Trace Table**: View all fine-grained events across the specified architectural components:
  - `agent` (full log of message sent to agent and response received)
  - `tool` (full log of tool message passed and received including actual payload)
  - `ollama vector` (log first 50 characters of text chunk sent, response from vectorizer, omitting vector floats)
  - `external API call` (full payload passed to and received from external APIs)
  - `LLM` (prompts sent to and response received from model with FULL payload)
  - `user`, `skill`, `vector database`
- **JSON Payload Inspector**: Click any event row to inspect full un-truncated JSON payloads and responses with sensitive API keys redacted.
- **Clear Logs**: Click **Clear Logs** and confirm in the dialog to erase audit records.

---

## 6. Directory Architecture

```
Agent-with-RAG-2/
├── database/                   # Persistent vector databases and audit log
│   ├── log.json                # Audit event logs
│   ├── doc_vectors.json        # Ingested document vectors and metadata
│   └── skill_vectors.json      # Indexed skill vectors and metadata
├── sample_docs/                # Built-in sample domain documents
│   ├── agent_rag_technology.md # Autonomous Agent and RAG architecture
│   ├── marketing_strategy.md   # Global enterprise marketing plan
│   └── financial_report.md     # FY2025 annual financial statement
├── services/                   # Modular backend service layer
│   ├── __init__.py
│   ├── ollama_service.py       # Ollama process management & embeddings
│   ├── vector_store.py         # Cosine similarity store & deduplication
│   ├── skill_manager.py        # Skill scanner, parser, and execution router
│   ├── llm_service.py          # Google AI Studio and custom LLM synthesis
│   ├── agent_orchestrator.py   # Multi-step pipeline coordinator
│   ├── google_adk_agent.py     # Google ADK LlmAgent service implementation
│   ├── telemetry_service.py    # Metric aggregation & interval bucketing
│   └── log_service.py          # Key-redacting JSON audit logger
├── skills/                     # Skill plugins with SOPs and execution scripts
│   ├── time-weather-skill/     # Open-Meteo & world clock
│   │   ├── SKILL.md
│   │   └── scripts/env_tools.py
│   ├── person-information-skill/ # 20-row personnel registry
│   │   ├── SKILL.md
│   │   ├── data/registry.csv
│   │   └── scripts/person_search.py
│   ├── stock-market-skill/     # Top market gainers/losers
│   │   ├── SKILL.md
│   │   └── scripts/stock_search.py
│   └── document-retriever-skill/ # Vector DB RAG retriever
│       ├── SKILL.md
│       └── tools/
│           └── document_search_tool.py # Document vector database retrieval tool
├── static/                     # Frontend static assets
│   ├── css/style.css           # Premium responsive UI styling
│   └── js/app.js               # Frontend application controller
├── templates/
│   └── index.html              # Single-page web application with 4 views
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── test_vector_store.py    # Vector store, chunking, dedup tests
│   ├── test_skills.py          # Skill execution and SOP tests
│   └── test_api.py             # Flask API integration tests
├── app.py                      # Flask orchestrator application
├── config.py                   # Central hyperparameters and configuration
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation and user guide
```

---

## 7. Running the Automated Test Suite

Run the full pytest suite to verify all modules:
```bash
python3 -m pytest tests/ -v
```
All 21 unit and API integration tests will run and validate vector similarity, chunk deduplication, skill execution, key redaction, multi-turn tool loops, document search tools, Google ADK LlmAgent invocation, and API endpoints.
