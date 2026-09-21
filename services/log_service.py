"""
Audit and Event Logging Service.
Stores all invocations, requests, responses, and errors in database/log.json.
Provides query, aggregation, conversation tracking, and sensitive API key redaction.
"""
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from config import LOG_FILE

_log_lock = threading.Lock()

# Common API key regex patterns to redact
API_KEY_PATTERNS = [
    (re.compile(r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\'])([^"\']{6,})(["\'])'), r'\g<1>****\g<3>'),
    (re.compile(r'(?i)(bearer\s+)([A-Za-z0-9_\-\.]{15,})'), r'\g<1>****'),
    (re.compile(r'(?i)(token["\']?\s*[:=]\s*["\'])([^"\']{6,})(["\'])'), r'\g<1>****\g<3>'),
    (re.compile(r'AIza[0-9A-Za-z_\-]{25,45}'), '****'),
    (re.compile(r'sk-[0-9A-Za-z]{20,60}'), '****'),
]

def redact_sensitive_data(data: Any) -> Any:
    """
    Recursively redact API keys, tokens, and secret credentials with '****'.
    """
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            if any(secret_term in k.lower() for secret_term in ["key", "secret", "token", "password", "authorization"]):
                if isinstance(v, str) and len(v) > 0:
                    redacted[k] = "****"
                else:
                    redacted[k] = v
            else:
                redacted[k] = redact_sensitive_data(v)
        return redacted
    elif isinstance(data, list):
        return [redact_sensitive_data(item) for item in data]
    elif isinstance(data, str):
        text = data
        for regex, repl in API_KEY_PATTERNS:
            text = regex.sub(repl, text)
        return text
    return data

class LogService:
    def __init__(self, log_path=LOG_FILE):
        self.log_path = log_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        with _log_lock:
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)

    def log_call(
        self,
        event_type: str,
        invoker: str,
        recipient: str,
        call_type: str,
        payload: Any,
        description: str = "",
        conversation_id: Optional[str] = None,
        latency_ms: float = 0.0,
        status: str = "success"
    ) -> Dict[str, Any]:
        """
        Record an individual invocation or response event per SPECIFICATION.md:
        - Include time of the call, type of the call, invoker, recipient, and raw payload passed.
        - Each invocation/response is a separate entry in the log.
        """
        now_utc = datetime.now(timezone.utc)
        # Per SPECIFICATION.md and user instruction:
        # "tool - full log of tool message passed to and received from the tool
        # including the actual payload. If the tool makes external API call, log the full payload
        # passed to the API and the full response received from the API. Do not redact or replace any part of the payload and response."
        # "do not redact or replace any part of the payload in the tools API or function call"
        lower_event = (event_type or "").lower()
        lower_invoker = (invoker or "").lower()
        lower_recipient = (recipient or "").lower()
        lower_call_type = (call_type or "").lower()

        is_tool_or_function_call = (
            any(
                kw in field
                for kw in ["tool", "function", "external api", "tools api"]
                for field in [lower_event, lower_invoker, lower_recipient, lower_call_type]
            ) or
            event_type in ["tool", "external API call", "function call", "tools API", "tool API", "function"] or
            invoker in ["tool", "external API call", "function call", "tools API", "tool API", "function"] or
            recipient in ["tool", "external API call", "function call", "tools API", "tool API", "function"] or
            call_type in ["function call", "tool call", "tools API"]
        )
        logged_payload = payload if is_tool_or_function_call else redact_sensitive_data(payload)

        entry = {
            "id": f"log-{int(now_utc.timestamp() * 1000)}-{os.urandom(3).hex()}",
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "call_type": call_type,
            "invoker": invoker,
            "target": recipient,
            "recipient": recipient,
            "description": description or f"{call_type.capitalize()}: {invoker} -> {recipient}",
            "payload": logged_payload,
            "conversation_id": conversation_id or "system",
            "latency_ms": round(latency_ms, 2),
            "status": status
        }

        with _log_lock:
            try:
                logs = []
                if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                logs.append(entry)
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=2)
            except Exception as e:
                print(f"[LogService Error] Failed to write log: {e}")

        return entry

    def log_event(
        self,
        event_type: str,
        invoker: str,
        target: str,
        payload: Any,
        response: Any = None,
        description: str = "",
        conversation_id: Optional[str] = None,
        latency_ms: float = 0.0,
        status: str = "success"
    ) -> Dict[str, Any]:
        """
        Record separate invocation and response entries per SPECIFICATION.md:
        'Do not combine the logs of from the request and response into the same log entry'
        """
        inv_entry = self.log_call(
            event_type=event_type,
            invoker=invoker,
            recipient=target,
            call_type="invocation",
            payload=payload,
            description=description or f"Invocation: {invoker} -> {target}",
            conversation_id=conversation_id,
            status=status
        )

        if response is not None:
            res_entry = self.log_call(
                event_type=event_type,
                invoker=target,
                recipient=invoker,
                call_type="response",
                payload=response,
                description=f"Response from {target} to {invoker}" if not description else f"Response: {description}",
                conversation_id=conversation_id,
                latency_ms=latency_ms,
                status=status
            )
            return res_entry

        return inv_entry

    def get_all_logs(self) -> List[Dict[str, Any]]:
        with _log_lock:
            if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
                return []
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []

    def clear_logs(self) -> bool:
        with _log_lock:
            try:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
                return True
            except Exception as e:
                print(f"[LogService Error] Failed to clear logs: {e}")
                return False

    def get_conversations(self) -> List[Dict[str, Any]]:
        """
        Extract unique conversation sessions with first user query and final agent response.
        """
        logs = self.get_all_logs()
        conversations = {}

        for entry in logs:
            cid = entry.get("conversation_id")
            if not cid or cid == "system":
                continue

            if cid not in conversations:
                conversations[cid] = {
                    "conversation_id": cid,
                    "timestamp": entry.get("timestamp"),
                    "user_query": "",
                    "agent_response": "",
                    "agent_type": "Custom Agent",
                    "total_events": 0,
                    "model_used": "unknown",
                    "status": "completed"
                }

            conversations[cid]["total_events"] += 1

            # Extract user query and agent type from invocation to agent
            if entry.get("event_type") in ["agent", "User Prompt"]:
                if entry.get("call_type") == "invocation" or entry.get("invoker") == "user":
                    p = entry.get("payload", {})
                    if isinstance(p, dict):
                        if p.get("agent_type"):
                            conversations[cid]["agent_type"] = p.get("agent_type")
                        if not conversations[cid]["user_query"]:
                            conversations[cid]["user_query"] = p.get("query") or p.get("message", "")
                    else:
                        if not conversations[cid]["user_query"]:
                            conversations[cid]["user_query"] = str(p)

            # Extract final agent response from response to user
            if entry.get("event_type") in ["agent", "Agent Response", "LLM", "LLM Synthesis"]:
                if entry.get("call_type") == "response" or entry.get("recipient") == "user":
                    p = entry.get("payload", {})
                    if isinstance(p, dict):
                        if p.get("agent_type"):
                            conversations[cid]["agent_type"] = p.get("agent_type")
                        resp = p.get("response") or p.get("content", "")
                        if resp:
                            conversations[cid]["agent_response"] = resp
                    elif isinstance(entry.get("response"), dict):
                        resp = entry.get("response", {}).get("content") or entry.get("response", {}).get("response", "")
                        if resp:
                            conversations[cid]["agent_response"] = resp

            if entry.get("event_type") in ["LLM", "LLM Synthesis"]:
                p = entry.get("payload", {})
                if isinstance(p, dict) and "model" in p:
                    conversations[cid]["model_used"] = p.get("model", "unknown")

        # Sort conversations reverse chronologically
        conv_list = list(conversations.values())
        conv_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return conv_list

    def get_conversation_logs(self, conversation_id: str) -> List[Dict[str, Any]]:
        logs = self.get_all_logs()
        return [l for l in logs if l.get("conversation_id") == conversation_id]

    def get_statistics(self) -> Dict[str, Any]:
        logs = self.get_all_logs()
        total_user_prompts = sum(1 for l in logs if l.get("event_type") in ["User Prompt", "agent"] and (l.get("call_type") == "invocation" or l.get("invoker") == "user"))
        total_model_calls = sum(1 for l in logs if "LLM" in l.get("event_type", "") and (l.get("call_type") == "invocation" or l.get("invoker") == "agent"))
        total_ollama_embeds = sum(1 for l in logs if ("ollama vector" in l.get("event_type", "").lower() or "embed" in l.get("event_type", "").lower()) and l.get("call_type") == "invocation")

        latencies = [l.get("latency_ms", 0) for l in logs if l.get("latency_ms", 0) > 0]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return {
            "total_user_prompts": total_user_prompts,
            "total_model_calls": total_model_calls,
            "total_ollama_embeds": total_ollama_embeds,
            "avg_latency_ms": avg_latency,
            "total_logs": len(logs)
        }

# Global singleton
audit_logger = LogService()
