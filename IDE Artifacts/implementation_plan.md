# Implementation Plan - Update Codebase to Match Current SPECIFICATION.md

This plan details all updates required to bring the application and codebase into full alignment with the latest [SPECIFICATION.md](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/SPECIFICATION.md).

## Summary of Spec Differences Identified

1. **Delete Any Document in Vector Storage Status Table** (Spec lines 98):
   - Spec: "Allow the user to delete any document from the DB by using the Delete button on the right side of the document row."
   - Current: Only total DB reset was supported; individual document deletion was missing from `vector_store.py`, `app.py`, `index.html`, and `app.js`.

2. **Group Retrieved Evidence Chunks by Skill and Document** (Spec line 71):
   - Spec: "Display the contents of the information retrieved from the vector store. Group chunks by skill and document."
   - Current: Evidence was grouped by processing step name ("Skill Search", "Document Search") rather than explicitly by skill and by document.

3. **Audit Log & Event Updates** (Spec lines 124-136, 156, 179):
   - Spec: Top table displays `timestamp (in local time), conversation ID, User Query, Agent Response, Agent Type, Number of Events (occured during the conversation), etc. Display at most 7 items and allow the user to scroll through all the items. The selected row should be highlighted.`
   - Spec: Set `agent_type` to `"Custom Agent"` and `"Google ADK Agent"` respectively and add to log records.
   - Spec: Bottom table title: `Events for Conversation for <Conversation ID>` with columns `Time and Date (local time)`, `Event Type`, `Invoker`, `Target`, `Short Description`.
   - Current: `Agent Type` was missing from the conversation list and table columns; the display height allowed more than 7 items; bottom table column was titled `Timestamp` instead of `Time and Date (local time)`.

4. **Tools Directory Architecture** (Spec line 190):
   - Spec: "Create the python tool in the tools/ folder called document_search_tool.py that can be used to get the list of text chunks from the document vector database"
   - Current: `document_search_tool.py` only existed in `skills/document-retriever-skill/tools/`. We will create the root-level `tools/` package with `tools/document_search_tool.py`.

5. **5 Web URL Sample Buttons** (Spec line 87):
   - Spec: "Provide 5 buttons for the user to click with samples of web URL that can be imported into the database."
   - Current: Button 5 was a local folder path (`sample_docs`) instead of a web URL.

6. **Minimize Skill-Specific Code in Orchestrator** (Spec line 171):
   - Spec: "Minimize skill-specific code in the orchestrator"
   - Current: `agent_orchestrator.py` contained hardcoded skill-name and tool-function detection logic. We will delegate tool signature inspection and resolution to `skill_manager`.

---

## Proposed Changes

### Vector Storage & Backend Services

#### [MODIFY] [services/vector_store.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/vector_store.py)
- Add `delete_document(self, doc_name: str) -> Dict[str, Any]` to `VectorStore`:
  - Lock store, load records.
  - Delete `doc_name` from `documents` dictionary.
  - Filter out all chunks associated with `doc_name`.
  - Save updated JSON store and log event to `audit_logger`.

#### [MODIFY] [app.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/app.py)
- Add endpoint `/api/vector/delete` (`POST` & `DELETE`):
  - Accepts `document_name` in request body.
  - Calls `doc_vector_store.delete_document(doc_name)`.
  - Returns updated status and confirmation.

#### [MODIFY] [services/log_service.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/log_service.py)
- In `get_conversations()`: Extract `agent_type` from agent invocation logs (defaulting to `"Custom Agent"`) and include `"agent_type"` in the conversation dictionary.

#### [MODIFY] [services/agent_orchestrator.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/agent_orchestrator.py)
- Set `"agent_type": "Custom Agent"` in agent invocation log payload.
- Delegate tool signature generation and execution matching to `skill_manager` to minimize skill-specific code in the orchestrator.

#### [MODIFY] [services/google_adk_agent.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/google_adk_agent.py)
- Set `"agent_type": "Google ADK Agent"` in agent invocation log payload (updating from `"google_adk"`).
- Ensure `skill_threshold` defaults to `config.DEFAULT_SKILL_THRESHOLD` (0.2).

#### [MODIFY] [services/skill_manager.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/services/skill_manager.py)
- Add `get_skill_tool_signature(skill)` helper to provide tool signatures dynamically to orchestrator.

---

### Tools Directory

#### [NEW] [tools/__init__.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/tools/__init__.py)
- Empty package init.

#### [NEW] [tools/document_search_tool.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/tools/document_search_tool.py)
- Provide `search_documents(query, top_k, min_score, conversation_id, invoker)` directly in `tools/` folder as specified in line 190.

---

### Web Frontend (GUI)

#### [MODIFY] [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/templates/index.html)
- **Vector DB Ingestion**:
  - Update sample URL button 5 to a web URL (`https://en.wikipedia.org/wiki/Large_language_model`) so all 5 buttons are samples of web URLs.
  - In Vector Storage Status table, add header `<th>Action</th>` on the right side of the document row.
- **Audit Log & Event**:
  - In User Conversations table, add `<th>Agent Type</th>` column.
  - Set table container height so at most 7 items are displayed comfortably before scrolling.
  - In Conversation Events table, rename header `<th>Timestamp</th>` to `<th>Time and Date (local time)</th>`.
  - Update title container to show `Events for Conversation for <Conversation ID>`.

#### [MODIFY] [static/js/app.js](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/static/js/app.js)
- **Retrieved Evidence**: Update `renderRetrievedEvidence(evidenceList)` to group chunks by Skill and Document (`Document: <doc_name>` and `Skill: <skill_name>`).
- **Vector Storage Status**:
  - In `loadVectorStorageStatus()`, render Delete button in each row:
    `<button class="btn-danger btn-sm" onclick="deleteDocument('${escapeHtml(d.name)}')">Delete</button>`.
  - Add `deleteDocument(docName)` function that prompts confirmation and calls `/api/vector/delete`.
- **Audit Log & Event**:
  - In `fetchLogsData()`, render `Agent Type` column in Table 1.
  - Update Table 2 title to `Events for Conversation for ${selectedId}`.
  - In Table 2, format the first column with `new Date(ev.timestamp).toLocaleString()` displaying both date and local time.

---

### Automated Tests

#### [MODIFY] [tests/test_api.py](file:///home/pi-net/Documents/agent_eng_labs/ai-agent-experiments/Agent-with-RAG-2/tests/test_api.py)
- Ensure test fixture starts Ollama service if not already started.
- Add test for `/api/vector/delete` endpoint.
- Verify `agent_type` in conversation records.
- Verify `tools/document_search_tool.py`.

---

## Verification Plan

### Automated Tests
- Run full pytest test suite:
  ```bash
  pytest -v
  ```
- Verify all endpoints and assertions pass cleanly.

### Manual Verification
- Launch the Flask orchestrator server.
- Verify in browser / API:
  - Deleting an individual document from the Vector Storage Status table works and updates statistics.
  - Evidence display groups chunks by skill and document.
  - Audit Log top table shows Agent Type, highlights active row, and displays at most 7 items before scroll.
  - Audit Log bottom table shows title `Events for Conversation for <ID>` and `Time and Date (local time)`.
