# Specification Compliance & Updates Walkthrough

This document summarizes the changes implemented across the codebase to fully align with [SPECIFICATION.md](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/SPECIFICATION.md).

---

## 1. Summary of Changes

### A. Individual Document Deletion from Vector Storage
- **Backend Service**:
  - In [services/vector_store.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/vector_store.py), replaced `self.lock = threading.Lock()` with `threading.RLock()` to prevent re-entrant self-deadlock when calling helper methods within write operations.
  - Implemented `delete_document(self, doc_name: str) -> Dict[str, Any]` which removes the document from `self.docs`, deletes all associated vector chunks from `self.chunks`, persists the updated state to `database/doc_vectors.json`, and writes an event to the audit log.
- **API Endpoint**:
  - In [app.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/app.py), added `/api/vector/delete` (supporting `POST` and `DELETE`) receiving `{"doc_name": "..."}` and invoking `vector_store.delete_document()`.
- **Frontend UI**:
  - In [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/templates/index.html), added an `Action` column (`<th>Action</th>`) to the Vector Storage Status table on Page 2.
  - In [static/js/app.js](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/static/js/app.js), updated `loadVectorStorageStatus()` to render a `Delete` button for each ingested document and added `deleteDocument(docName)` to call the backend endpoint and refresh status.

### B. Grouping Retrieved Context Evidence by Document (Page 1)
- Aligned with the updated SPECIFICATION.md ("*Include results from the skills vector store and the documents vector store. Group the results by the documents*"):
  - In [services/agent_orchestrator.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/agent_orchestrator.py) and [services/google_adk_agent.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/google_adk_agent.py):
    - Matched skills retrieved from the skills vector store (`database/skill_vectors.json`) are now appended to `retrieved_evidence` with `source_type="skill_vector_store"`, `store="skills"`, and `document_name="<folder>/SKILL.md"`.
    - Document chunks retrieved from the document vector store (`database/doc_vectors.json`) include `source_type="document_vector_store"`, `store="documents"`, and `document_name`.
  - In [static/js/app.js](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/static/js/app.js):
    - `renderRetrievedEvidence()` groups all retrieved results strictly by their source document (e.g. `▶ Document: <document_name> (<count> items)`).
    - Added dedicated visual tags to distinguish whether the item originated from the **Skills Vector Store** or the **Documents Vector Store**, accompanied by similarity scores and chunk text.

### C. Agent Type Logging & Audit Log Enhancements (Page 4)
- **Agent Type Identification**:
  - In [services/agent_orchestrator.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/agent_orchestrator.py), set and logged `agent_type: "Custom Agent"`.
  - In [services/google_adk_agent.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/google_adk_agent.py), set and logged `agent_type: "Google ADK Agent"`.
- **Log Indexing & Filtering**:
  - In [services/log_service.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/log_service.py), updated `get_conversations()` to extract `agent_type` from logged events (checking payload, metadata, and event details).
- **UI Layout & Headers**:
  - In [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/templates/index.html) and [static/js/app.js](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/static/js/app.js):
    - Added `Agent Type` column to the User Conversations table.
    - Constrained the table container height (`max-height: 280px; overflow-y: auto;`) to display at most 7 items before vertical scrolling.
    - Updated Table 2 header from `Timestamp` to `Time and Date (local time)` displaying localized date and time (`toLocaleString()`).
    - Updated Table 2 title to dynamic format: `Events for Conversation for <Conversation ID>`.

### D. Tool Consolidation to Skill Directory
- Per the updated specification ("*All the python tool should be in the skills/<skill_name>/tools/ folder*"):
  - Consolidated all document search logic into [skills/document-retriever-skill/tools/document_search_tool.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/skills/document-retriever-skill/tools/document_search_tool.py).
  - Updated [skills/document-retriever-skill/tools/__init__.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/skills/document-retriever-skill/tools/__init__.py) to use relative imports.
  - Removed the duplicate root `tools/` directory.
  - Confirmed all orchestrator and Google ADK agent loaders import directly from the skill tool directory.

### E. Sample Web URLs on Page 2
- In [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/templates/index.html), updated the sample buttons to 5 web URLs:
  1. Python Wikipedia
  2. Artificial Intelligence Wikipedia
  3. Retrieval-Augmented Generation Wikipedia
  4. Open-Meteo Documentation
  5. Large Language Model Wikipedia

### F. Decoupling Orchestrator from Hardcoded Skill Logic
- In [services/skill_manager.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/skill_manager.py), added:
  - `get_skill_tool_signature(skill_info: Dict[str, Any]) -> Optional[Dict[str, Any]]`
  - `is_skill_tool_match(skill_info: Dict[str, Any], tool_name: str) -> bool`
- In [services/agent_orchestrator.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/agent_orchestrator.py), eliminated hardcoded `if "weather" in ... elif "person" in ...` branch logic, delegating tool signature building and matching directly to `skill_manager`.

---

## 2. Verification Results

### Automated Unit and Integration Tests
Ran the full test suite with `pytest -v`:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
rootdir: /home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2
collected 25 items

tests/test_api.py::test_api_key_redaction PASSED                         [  4%]
tests/test_api.py::test_tool_payload_not_redacted PASSED                 [  8%]
tests/test_api.py::test_health_endpoint PASSED                           [ 12%]
tests/test_api.py::test_models_endpoint PASSED                           [ 16%]
tests/test_api.py::test_vector_status_endpoint PASSED                    [ 20%]
tests/test_api.py::test_ollama_models_endpoint PASSED                    [ 24%]
tests/test_api.py::test_telemetry_endpoint PASSED                        [ 28%]
tests/test_api.py::test_logs_endpoint PASSED                             [ 32%]
tests/test_api.py::test_chat_endpoint PASSED                             [ 36%]
tests/test_api.py::test_component_logging PASSED                         [ 40%]
tests/test_skills.py::test_skills_list_endpoint PASSED                   [ 44%]
tests/test_api.py::test_chat_with_custom_parameters PASSED               [ 48%]
tests/test_api.py::test_document_search_tool PASSED                      [ 52%]
tests/test_api.py::test_google_adk_agent_chat PASSED                     [ 56%]
tests/test_api.py::test_delete_document_endpoint PASSED                  [ 60%]
tests/test_skills.py::test_parse_skill_markdown PASSED                   [ 64%]
tests/test_skills.py::test_weather_and_time_tool PASSED                  [ 68%]
tests/test_skills.py::test_person_registry_tool PASSED                   [ 72%]
tests/test_skills.py::test_stock_search_tool PASSED                      [ 76%]
tests/test_skills.py::test_parse_tool_call_json PASSED                   [ 80%]
tests/test_skills.py::test_person_registry_tool_with_arguments PASSED    [ 84%]
tests/test_vector_store.py::test_cosine_similarity PASSED                [ 88%]
tests/test_vector_store.py::test_chunk_text PASSED                       [ 92%]
tests/test_vector_store.py::test_vector_store_deduplication PASSED       [ 96%]
tests/test_vector_store.py::test_vector_store_reset PASSED               [100%]

================== 25 passed, 3 warnings in 141.14s (0:02:21) ==================
```

All 25 tests passed cleanly without any regressions.
