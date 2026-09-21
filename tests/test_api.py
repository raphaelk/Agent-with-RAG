"""
Integration Tests for Flask API endpoints and sensitive key redaction.
"""
import pytest
import json
import config
from app import app
from services.log_service import redact_sensitive_data, audit_logger

@pytest.fixture
def client():
    from services.ollama_service import ollama_service
    ollama_service.ensure_service_started()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_key_redaction():
    payload = {
        "api_key": "AIzaSyD-secret1234567890abcdefghijklm",
        "authorization": "Bearer sec_abcdef1234567890",
        "query": "Hello world with key AIzaSyD-secret1234567890abcdefghijklm embedded in text"
    }
    redacted = redact_sensitive_data(payload)
    assert redacted["api_key"] == "****"
    assert redacted["authorization"] == "****"
    assert "AIzaSyD" not in redacted["query"]
    assert "****" in redacted["query"]

def test_tool_payload_not_redacted():
    tool_payload = {
        "tool": "env_tools.py",
        "api_key": "raw_unredacted_key_or_param",
        "authorization": "Bearer token123",
        "city": "Tokyo"
    }
    logged = audit_logger.log_call(
        event_type="tool",
        invoker="skill",
        recipient="tool",
        call_type="invocation",
        payload=tool_payload
    )
    assert logged["payload"]["api_key"] == "raw_unredacted_key_or_param"
    assert logged["payload"]["authorization"] == "Bearer token123"

    api_payload = {
        "status": "success",
        "api_secret": "raw_secret_data",
        "temperature": 23.5
    }
    logged_api = audit_logger.log_call(
        event_type="external API call",
        invoker="external API call",
        recipient="tool",
        call_type="response",
        payload=api_payload
    )
    assert logged_api["payload"]["api_secret"] == "raw_secret_data"

    # Function call payload
    fn_payload = {
        "function": "calculate_metrics",
        "api_key": "raw_api_key_12345",
        "token": "raw_token_xyz"
    }
    logged_fn = audit_logger.log_call(
        event_type="function call",
        invoker="agent",
        recipient="function",
        call_type="function call",
        payload=fn_payload
    )
    assert logged_fn["payload"]["api_key"] == "raw_api_key_12345"
    assert logged_fn["payload"]["token"] == "raw_token_xyz"

    # Tools API payload
    tools_api_payload = {
        "endpoint": "https://api.example.com/v1/tools",
        "secret_key": "super_secret_raw_key",
        "parameters": {"count": 10}
    }
    logged_tools_api = audit_logger.log_call(
        event_type="tools API",
        invoker="tool",
        recipient="tools API",
        call_type="invocation",
        payload=tools_api_payload
    )
    assert logged_tools_api["payload"]["secret_key"] == "super_secret_raw_key"


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "agent" in data
    assert "ollama" in data

def test_models_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert any(m["id"] == "custom" for m in data["models"])

def test_vector_status_endpoint(client):
    res = client.get("/api/vector/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "total_chunks" in data
    assert "total_documents" in data
    assert "db_size_mb" in data

def test_models_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert "default_model" in data
    assert data["default_model"] == "gemma-4-26b-a4b-it"
    assert len(data["models"]) > 0
    assert data["models"][0]["id"] == "gemma-4-26b-a4b-it"

def test_ollama_models_endpoint(client):
    res = client.get("/api/ollama/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    assert len(data["models"]) > 0
    # Check that model characteristics include dimensions, context_window, size, description, status
    first = data["models"][0]
    assert "dimensions" in first
    assert "context_window" in first
    assert "size" in first
    assert "description" in first
    assert "status" in first

def test_telemetry_endpoint(client):
    res = client.get("/api/telemetry?model=All+Models&time_range=1+day&interval=15+min")
    assert res.status_code == 200
    data = res.get_json()
    assert "totals" in data
    assert "charts" in data

def test_logs_endpoint(client):
    res = client.get("/api/logs")
    assert res.status_code == 200
    data = res.get_json()
    assert "statistics" in data
    assert "conversations" in data

def test_chat_endpoint(client):
    res = client.post(
        "/api/chat",
        json={
            "query": "What is the weather in Tokyo?",
            "model": config.DEFAULT_LLM_MODEL,
            "temperature": 0.5,
            "max_tokens": 1024,
            "max_rag_chunks": 3
        }
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "answer" in data["data"]
    assert "retrieved_evidence" in data["data"]
    assert "steps" in data["data"]
    assert len(data["data"]["steps"]) > 0
    first_step = data["data"]["steps"][0]
    assert "step_name" in first_step
    assert "elapsed_ms" in first_step
    assert "logs" in first_step

def test_component_logging(client):
    # Perform chat inquiry that triggers weather skill, vector search, and model
    res = client.post(
        "/api/chat",
        json={
            "query": "What is the weather in London right now?",
            "model": config.DEFAULT_LLM_MODEL
        }
    )
    assert res.status_code == 200
    conv_id = res.get_json()["data"]["conversation_id"]

    # Query logs for this conversation
    log_res = client.get(f"/api/logs?conversation_id={conv_id}")
    assert log_res.status_code == 200
    log_data = log_res.get_json()

    events = log_data.get("events", [])
    assert len(events) > 0

    invokers = {e.get("invoker") for e in events}
    targets = {e.get("target") for e in events}
    event_types = {e.get("event_type") for e in events}

    # Verify component interactions logged per specification
    assert "user" in invokers or "user" in targets
    assert "agent" in invokers
    assert "skill search" in event_types or "skill" in invokers or "skill" in targets
    assert "tool" in event_types or "tool" in invokers or "tool" in targets
    assert "ollama vector" in event_types or "ollama vector" in targets
    assert "external API call" in event_types or "external API call" in targets
    assert "LLM" in event_types or "LLM" in targets

    # For a weather query, document search MUST NOT be called!
    assert "document search" not in event_types

    # Verify separate invocation and response entries
    call_types = {e.get("call_type") for e in events}
    assert "invocation" in call_types
    assert "response" in call_types

    # Verify conversation list displays Number of Events
    convs = log_data.get("conversations", [])
    matching_conv = next((c for c in convs if c["conversation_id"] == conv_id), None)
    assert matching_conv is not None
    assert matching_conv["total_events"] > 0

    # Test query that should NOT trigger document-retriever-skill or any tool: 'What is the capital city of Japan?'
    res_general = client.post(
        "/api/chat",
        json={
            "query": "What is the capital city of Japan?",
            "model": config.DEFAULT_LLM_MODEL
        }
    )
    assert res_general.status_code == 200
    gen_conv_id = res_general.get_json()["data"]["conversation_id"]
    gen_log_res = client.get(f"/api/logs?conversation_id={gen_conv_id}")
    gen_events = gen_log_res.get_json().get("events", [])
    gen_event_types = {e.get("event_type") for e in gen_events}
    # Verify document search was NOT called for capital city query
    assert "document search" not in gen_event_types
    # Verify tool was NOT called for capital city query
    assert "tool" not in gen_event_types

    # Test query that DOES trigger document-retriever-skill
    res_doc = client.post(
        "/api/chat",
        json={
            "query": "Explain the architecture of Agent and RAG technology from our documents",
            "model": config.DEFAULT_LLM_MODEL,
            "skill_threshold": 0.3
        }
    )
    assert res_doc.status_code == 200
    doc_conv_id = res_doc.get_json()["data"]["conversation_id"]
    doc_log_res = client.get(f"/api/logs?conversation_id={doc_conv_id}")
    doc_events = doc_log_res.get_json().get("events", [])
    doc_event_types = {e.get("event_type") for e in doc_events}
    # Verify document search WAS called when document-retriever-skill matched
    assert "document search" in doc_event_types

def test_skills_list_endpoint(client):
    res = client.get("/api/skills/list")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "skills" in data
    assert len(data["skills"]) >= 4
    folder_names = [s["folder_name"] for s in data["skills"]]
    assert "time-weather-skill" in folder_names
    assert "person-information-skill" in folder_names
    assert "stock-market-skill" in folder_names
    assert "document-retriever-skill" in folder_names

def test_chat_with_custom_parameters(client):
    # Test chat with explicit specific skill mode
    res = client.post(
        "/api/chat",
        json={
            "query": "What is the weather in Tokyo?",
            "model": config.DEFAULT_LLM_MODEL,
            "skills_mode": "time-weather-skill",
            "skill_threshold": 0.4,
            "doc_threshold": 0.3,
            "max_turns": 3
        }
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "answer" in data["data"]
    assert "steps" in data["data"]

def test_document_search_tool():
    import importlib.util
    from pathlib import Path
    from config import SKILLS_DIR
    tool_path = SKILLS_DIR / "document-retriever-skill" / "tools" / "document_search_tool.py"
    assert tool_path.exists()
    spec = importlib.util.spec_from_file_location("document_search_tool", str(tool_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.search_documents(query="Agent RAG technology", top_k=3, min_score=0.1)
    assert isinstance(res, list)
    alias_res = mod.get_document_chunks(query="Agent RAG technology", top_k=3, min_score=0.1)
    assert isinstance(alias_res, list)

def test_google_adk_agent_chat(client):
    res = client.post(
        "/api/chat",
        json={
            "query": "What is the weather in London?",
            "agent_type": "google_adk",
            "model": "gemini-3.6-flash"
        }
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    chat_data = data["data"]
    assert "response" in chat_data or "answer" in chat_data
    assert chat_data.get("agent_type") == "Google ADK Agent"
    assert "steps" in chat_data
    assert len(chat_data["steps"]) >= 3

def test_delete_document_endpoint(client):
    from services.vector_store import doc_vector_store
    # Ingest a temporary dummy text document
    doc_vector_store.ingest_text_document("test_to_delete.txt", "This is a temporary document chunk text for testing deletion.")
    stats_before = doc_vector_store.get_stats()
    assert any(d["name"] == "test_to_delete.txt" for d in stats_before["documents"])

    # Call delete endpoint
    res = client.post("/api/vector/delete", json={"document_name": "test_to_delete.txt"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["deleted_document"] == "test_to_delete.txt"

    # Verify deleted
    stats_after = doc_vector_store.get_stats()
    assert not any(d["name"] == "test_to_delete.txt" for d in stats_after["documents"])

