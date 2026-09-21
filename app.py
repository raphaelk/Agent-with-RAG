"""
Core Orchestrator & Flask Web Application for Agent with RAG.
Provides REST APIs for Chat & Synthesis, Vector DB Ingestion, Telemetry, and Audit Logs.
"""
import atexit
import os
import signal
import sys
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from services.ollama_service import ollama_service
from services.vector_store import doc_vector_store, skill_vector_store
from services.skill_manager import skill_manager
from services.llm_service import llm_service
from services.telemetry_service import telemetry_service
from services.log_service import audit_logger
from services.agent_orchestrator import orchestrator
from services.google_adk_agent import google_adk_agent

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR)
)

def startup_initialization():
    """
    Perform startup checks per specification:
    1. Check if ollama is currently running and start it if not.
    2. Scan skills/ folder and load skills not currently in database.
    3. Get list of ONLY active LLM models for text generation from Google AI Studio API.
    """
    print("[Agent-with-RAG] Checking Ollama service status...")
    ollama_service.ensure_service_started()

    print("[Agent-with-RAG] Scanning skills/ directory...")
    skill_manager.scan_and_load_skills()

    print("[Agent-with-RAG] Fetching active text generation models from Google AI Studio...")
    llm_service.list_available_models()
    print("[Agent-with-RAG] Startup initialization complete.")

def shutdown_handler():
    """
    When the app terminates, shutdown ollama service ONLY if the app started it.
    If it was already running before the app started, do not shutdown ollama.
    """
    print("[Agent-with-RAG] Initiating graceful termination...")
    ollama_service.shutdown_if_started_by_app()

atexit.register(shutdown_handler)

# ----------------- Web GUI Route -----------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------- Health & Shutdown APIs -----------------
@app.route("/api/health", methods=["GET"])
def get_health():
    """
    Periodic health check of backend Agent and supporting services.
    """
    ollama_up = ollama_service.is_running()
    doc_stats = doc_vector_store.get_stats()
    skill_stats = skill_vector_store.get_stats()

    overall_status = "healthy" if ollama_up else "degraded"

    return jsonify({
        "status": overall_status,
        "agent": "online",
        "ollama": {
            "status": "online" if ollama_up else "offline",
            "started_by_app": ollama_service.started_by_app,
            "active_model": ollama_service.current_model
        },
        "vector_db": {
            "document_chunks": doc_stats["total_chunks"],
            "skill_records": skill_stats["total_chunks"]
        }
    })

@app.route("/api/shutdown", methods=["POST"])
def shutdown_app():
    """
    Shutdown all services started by the app.
    Does not shut down supporting services that were running before app started.
    """
    data = request.get_json(silent=True) or {}
    confirm_text = data.get("confirm_text", "").strip()
    if confirm_text != "Shutdown the service":
        return jsonify({"status": "error", "message": "Confirmation text does not match 'Shutdown the service'"}), 400

    audit_logger.log_event(
        event_type="System Shutdown",
        invoker="User",
        target="Server Orchestrator",
        payload={"confirmation": confirm_text},
        response={"action": "shutting_down"},
        description="User triggered full service shutdown"
    )

    # Perform shutdown cleanup
    shutdown_handler()

    # Terminate process after short delay to let response return
    def terminate_server():
        import time
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=terminate_server, daemon=True).start()

    return jsonify({
        "status": "success",
        "message": "Shutdown initiated. Services started by the app have been cleanly terminated."
    })

# ----------------- Models & Chat APIs -----------------
@app.route("/api/models", methods=["GET"])
def get_models():
    """
    List Google AI Studio active models capable of synthesizing text, plus Custom model.
    """
    models = llm_service.list_available_models()
    return jsonify({
        "models": models,
        "default_model": config.DEFAULT_LLM_MODEL,
        "last_custom_endpoint": llm_service.last_custom_endpoint
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint for Page 1.
    Performs RAG retrieval, skill dispatch, context evidence aggregation, and LLM synthesis.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    model = data.get("model", config.DEFAULT_LLM_MODEL)
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 2048))
    max_rag_chunks = int(data.get("max_rag_chunks", 5))
    custom_endpoint = data.get("custom_endpoint")
    conversation_id = data.get("conversation_id")
    skills_mode = data.get("skills_mode", "vector_store")
    skill_threshold = float(data.get("skill_threshold", config.DEFAULT_SKILL_THRESHOLD))
    doc_threshold = float(data.get("doc_threshold", config.DEFAULT_DOC_THRESHOLD))
    max_turns = min(int(data.get("max_turns", config.DEFAULT_MAX_TURNS)), config.MAX_TURNS_LIMIT)
    agent_type = data.get("agent_type", "custom")  # "custom" or "google_adk"

    if agent_type == "google_adk":
        result = google_adk_agent.process_chat(
            query=query,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_rag_chunks=max_rag_chunks,
            custom_endpoint=custom_endpoint,
            conversation_id=conversation_id,
            skills_mode=skills_mode,
            skill_threshold=skill_threshold,
            doc_threshold=doc_threshold,
            max_turns=max_turns
        )
    else:
        result = orchestrator.process_chat(
            query=query,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_rag_chunks=max_rag_chunks,
            custom_endpoint=custom_endpoint,
            conversation_id=conversation_id,
            skills_mode=skills_mode,
            skill_threshold=skill_threshold,
            doc_threshold=doc_threshold,
            max_turns=max_turns
        )

    return jsonify({"status": "success", "data": result})

@app.route("/api/skills/list", methods=["GET"])
def list_skills():
    """
    List all available skills from the skills/ directory.
    """
    skills = skill_manager.get_all_skills()
    return jsonify({
        "status": "success",
        "skills": [
            {
                "folder_name": s["folder_name"],
                "name": s["name"],
                "description": s["description"]
            }
            for s in skills
        ]
    })

# ----------------- Ingestion & Vector Storage APIs -----------------
@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    """
    Populate vector database from a web URL or local directory/file.
    Supports customizable chunk size and overlap, ensuring no duplicate chunks.
    """
    data = request.get_json() or {}
    source_type = data.get("type", "url")  # 'url' or 'local'
    path_or_url = data.get("target", "").strip()
    chunk_size = int(data.get("chunk_size", config.DEFAULT_CHUNK_SIZE))
    overlap = int(data.get("overlap", config.DEFAULT_CHUNK_OVERLAP))

    if not path_or_url:
        return jsonify({"status": "error", "message": "Target URL or path is required"}), 400

    if source_type == "url":
        res = doc_vector_store.ingest_url(path_or_url, chunk_size=chunk_size, overlap=overlap)
    else:
        res = doc_vector_store.ingest_local_path(path_or_url, chunk_size=chunk_size, overlap=overlap)

    if res.get("status") == "error":
        return jsonify(res), 400
    return jsonify(res)

@app.route("/api/vector/status", methods=["GET"])
def vector_status():
    """
    Retrieve document vector storage statistics and ingested documents table.
    """
    stats = doc_vector_store.get_stats()
    return jsonify(stats)

@app.route("/api/vector/reset", methods=["POST"])
def reset_vector_db():
    """
    Reset and wipe document vector database.
    """
    doc_vector_store.reset()
    audit_logger.log_event(
        event_type="Vector DB Reset",
        invoker="User",
        target="VectorStore",
        payload={"action": "reset_doc_db"},
        response={"status": "cleared"},
        description="Reset document vector store"
    )
    return jsonify({"status": "success", "message": "Document database reset successfully"})

@app.route("/api/vector/delete", methods=["POST", "DELETE"])
def delete_vector_document():
    """
    Delete an individual document and its chunks from the vector database.
    Per SPECIFICATION.md: 'Allow the user to delete any document from the DB by using the Delete button on the right side of the document row'
    """
    data = request.get_json(silent=True) or {}
    doc_name = data.get("document_name", "").strip() or request.args.get("document_name", "").strip()
    if not doc_name:
        return jsonify({"status": "error", "message": "document_name is required"}), 400

    res = doc_vector_store.delete_document(doc_name)
    if res.get("status") == "error":
        return jsonify(res), 404
    return jsonify(res)

@app.route("/api/skills/update", methods=["POST"])
def update_skills_db():
    """
    Scan skills/ folder and load new skills not currently in the skills database.
    """
    res = skill_manager.scan_and_load_skills()
    return jsonify({"status": "success", "data": res})

# ----------------- Ollama Embedding Model APIs -----------------
@app.route("/api/ollama/models", methods=["GET"])
def get_ollama_models():
    """
    List supported embedding models with dimensions, context window, size,
    brief description, and status (Installed, Active, Available to Pull).
    """
    catalog = ollama_service.get_embedding_catalog()
    return jsonify({
        "current_model": ollama_service.current_model,
        "models": catalog
    })

@app.route("/api/ollama/switch-model", methods=["POST"])
def switch_ollama_model():
    """
    Switch Ollama embedding model:
    1. Confirm switch
    2. Delete data in document and skill databases
    3. Make Ollama load/pull the new model
    4. Scan skills/ folder and import skills to skill database
    """
    data = request.get_json() or {}
    new_model = data.get("model", "").strip()
    confirm_text = data.get("confirm_text", "").strip()

    if not new_model:
        return jsonify({"status": "error", "message": "New model name is required"}), 400

    if confirm_text != "Delete Data and Switch":
        return jsonify({"status": "error", "message": "Confirmation text does not match 'Delete Data and Switch'"}), 400

    # Step 1: Wipe document & skill vector stores
    doc_vector_store.reset()
    skill_vector_store.reset()

    # Step 2: Ensure model is available in Ollama
    installed = ollama_service.get_installed_models()
    if not any(new_model in m or m in new_model for m in installed):
        pulled = ollama_service.pull_model(new_model)
        if not pulled:
            return jsonify({"status": "error", "message": f"Failed to pull model {new_model} in Ollama"}), 500

    ollama_service.current_model = new_model

    # Step 3: Scan skills and re-import
    skills_imported = skill_manager.scan_and_load_skills()

    audit_logger.log_event(
        event_type="Model Switch",
        invoker="User",
        target="Ollama",
        payload={"new_model": new_model},
        response={"skills_reimported": skills_imported.get("loaded_skills")},
        description=f"Switched embedding model to {new_model} and wiped existing vectors"
    )

    return jsonify({
        "status": "success",
        "current_model": new_model,
        "skills_imported": skills_imported
    })

# ----------------- Telemetry APIs -----------------
@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    """
    Page 3 Telemetry metrics with model filter, aggregation interval, and time range.
    """
    model_filter = request.args.get("model", "All Models")
    time_range = request.args.get("time_range", "1 day")
    interval = request.args.get("interval", "15 min")
    custom_start = request.args.get("custom_start")
    custom_end = request.args.get("custom_end")

    metrics = telemetry_service.get_aggregated_metrics(
        model_filter=model_filter,
        time_range=time_range,
        interval=interval,
        custom_start=custom_start,
        custom_end=custom_end
    )
    return jsonify(metrics)

# ----------------- Audit Log APIs -----------------
@app.route("/api/logs", methods=["GET"])
def get_logs():
    """
    Page 4 Audit Log & Event queries.
    Returns statistics, conversation list, and event details for a selected conversation.
    """
    conversation_id = request.args.get("conversation_id")
    stats = audit_logger.get_statistics()
    conversations = audit_logger.get_conversations()

    selected_events = []
    if conversation_id:
        selected_events = audit_logger.get_conversation_logs(conversation_id)
    elif conversations:
        # Default to first conversation
        selected_events = audit_logger.get_conversation_logs(conversations[0]["conversation_id"])

    return jsonify({
        "statistics": stats,
        "conversations": conversations,
        "selected_conversation_id": conversation_id or (conversations[0]["conversation_id"] if conversations else None),
        "events": selected_events
    })

@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    """
    Clear all audit logs after confirmation.
    """
    data = request.get_json(silent=True) or {}
    confirm = data.get("confirm", False)
    if not confirm:
        return jsonify({"status": "error", "message": "Confirmation required"}), 400

    audit_logger.clear_logs()
    return jsonify({"status": "success", "message": "Audit logs cleared successfully"})


if __name__ == "__main__":
    startup_initialization()
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE
    )
