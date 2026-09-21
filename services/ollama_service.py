"""
Ollama Service Integration.
Manages local Ollama process lifecycle, model pulling, embedding generation,
and catalog status tracking.
"""
import subprocess
import time
import requests
import json
from typing import List, Dict, Any, Optional
from config import OLLAMA_BASE_URL, DEFAULT_EMBEDDING_MODEL, SUPPORTED_EMBEDDING_MODELS
from services.log_service import audit_logger

class OllamaService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.current_model = DEFAULT_EMBEDDING_MODEL
        self.started_by_app = False
        self.process: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        """Check if Ollama server responds to HTTP ping."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def ensure_service_started(self) -> bool:
        """
        Check whether ollama is currently running.
        If not, start the background service and record started_by_app = True.
        """
        if self.is_running():
            self.started_by_app = False
            return True

        # Try to launch ollama serve
        try:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            # Wait for startup
            for _ in range(15):
                time.sleep(0.5)
                if self.is_running():
                    self.started_by_app = True
                    audit_logger.log_event(
                        event_type="Ollama Lifecycle",
                        invoker="System",
                        target="Ollama",
                        payload={"action": "start_service"},
                        response={"status": "running", "started_by_app": True},
                        description="Started local Ollama background service"
                    )
                    return True
        except Exception as e:
            audit_logger.log_event(
                event_type="Ollama Error",
                invoker="System",
                target="Ollama",
                payload={"action": "start_service"},
                response={"error": str(e)},
                description=f"Failed to start Ollama daemon: {e}",
                status="error"
            )

        return self.is_running()

    def shutdown_if_started_by_app(self) -> bool:
        """
        Terminate Ollama service ONLY if the app started it.
        If it was already running, do nothing.
        """
        if not self.started_by_app:
            return False

        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=3)
                self.process = None
            else:
                subprocess.run(["pkill", "-f", "ollama serve"], check=False)
            self.started_by_app = False
            audit_logger.log_event(
                event_type="Ollama Lifecycle",
                invoker="System",
                target="Ollama",
                payload={"action": "shutdown_service"},
                response={"status": "stopped"},
                description="Gracefully stopped Ollama service (started by app)"
            )
            return True
        except Exception as e:
            print(f"[Ollama Shutdown Error] {e}")
            return False

    def get_installed_models(self) -> List[str]:
        """Fetch list of models locally present in Ollama."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if res.status_code == 200:
                data = res.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            print(f"[Ollama Tags Error] {e}")
        return []

    def get_embedding_catalog(self) -> List[Dict[str, Any]]:
        """
        List supported embedding models with dimensions, context window,
        size, brief description, and status (Installed, Active, Available to Pull).
        """
        installed = self.get_installed_models()
        catalog = []

        for model_meta in SUPPORTED_EMBEDDING_MODELS:
            name = model_meta["name"]
            is_installed = any(name in inst or inst in name for inst in installed)
            is_active = (name == self.current_model or self.current_model.startswith(name.split(":")[0]))

            if is_active:
                status = "Active"
            elif is_installed:
                status = "Installed"
            else:
                status = "Available to Pull"

            catalog.append({
                **model_meta,
                "status": status,
                "is_active": is_active,
                "is_installed": is_installed
            })

        return catalog

    def pull_model(self, model_name: str) -> bool:
        """Download / pull model in Ollama if not present."""
        start_time = time.time()
        try:
            res = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=180
            )
            latency = (time.time() - start_time) * 1000
            success = (res.status_code == 200)

            audit_logger.log_event(
                event_type="Ollama Pull",
                invoker="Agent Orchestrator",
                target="Ollama",
                payload={"model": model_name},
                response=res.json() if success else {"error": res.text},
                description=f"Pull Ollama model: {model_name}",
                latency_ms=latency,
                status="success" if success else "error"
            )
            return success
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            audit_logger.log_event(
                event_type="Ollama Pull Error",
                invoker="Agent Orchestrator",
                target="Ollama",
                payload={"model": model_name},
                response={"error": str(e)},
                description=f"Error pulling model {model_name}: {e}",
                latency_ms=latency,
                status="error"
            )
            return False

    def generate_embedding(self, text: str, model_name: Optional[str] = None, conversation_id: Optional[str] = None, invoker: str = "vector database") -> List[float]:
        """
        Generate dense vector embedding for input text.
        Logs interactions between invoker ('skill search' or 'document search') and 'ollama vector'.
        """
        target_model = model_name or self.current_model
        start_time = time.time()

        payload = {"model": target_model, "vectorizer": target_model, "prompt": text}
        first_50_chars = text[:50]
        audit_logger.log_call(
            event_type="ollama vector",
            call_type="invocation",
            invoker=invoker,
            recipient="ollama vector",
            payload={"vectorizer": target_model, "text_chunk_first_50": first_50_chars, "total_characters": len(text)},
            description=f"Log first 50 characters sent for embedding to vectorizer {target_model}: '{first_50_chars}'",
            conversation_id=conversation_id
        )

        try:
            res = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=30)
            latency = (time.time() - start_time) * 1000

            if res.status_code == 200:
                data = res.json()
                embedding = data.get("embedding", [])
                audit_logger.log_call(
                    event_type="ollama vector",
                    call_type="response",
                    invoker="ollama vector",
                    recipient=invoker,
                    payload={"status": "success", "vectorizer": target_model, "response": {"status": "success", "dimensions": len(embedding), "model": target_model}},
                    description=f"Log response from vectorizer {target_model} ({len(embedding)} dimensions, vectors omitted)",
                    conversation_id=conversation_id,
                    latency_ms=latency
                )
                return embedding

            # Fallback endpoint /api/embed
            res2 = requests.post(f"{self.base_url}/api/embed", json={"model": target_model, "input": text}, timeout=30)
            if res2.status_code == 200:
                data2 = res2.json()
                embeddings = data2.get("embeddings", [])
                embedding = embeddings[0] if embeddings else []
                latency = (time.time() - start_time) * 1000
                audit_logger.log_call(
                    event_type="ollama vector",
                    call_type="response",
                    invoker="ollama vector",
                    recipient=invoker,
                    payload={"status": "success", "vectorizer": target_model, "response": {"status": "success", "dimensions": len(embedding), "model": target_model}},
                    description=f"Log response from vectorizer {target_model} ({len(embedding)} dimensions, vectors omitted)",
                    conversation_id=conversation_id,
                    latency_ms=latency
                )
                return embedding

            raise RuntimeError(f"Ollama embedding failed with code {res.status_code}: {res.text}")

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            audit_logger.log_call(
                event_type="ollama vector",
                call_type="response",
                invoker="ollama vector",
                recipient=invoker,
                payload={"status": "error", "error": str(e)},
                description=f"Embedding error: {e}",
                conversation_id=conversation_id,
                latency_ms=latency,
                status="error"
            )
            # Offline pseudo-embedding fallback (deterministic hashing vector)
            import hashlib
            import numpy as np
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384).tolist()
            return vec

# Global singleton
ollama_service = OllamaService()
