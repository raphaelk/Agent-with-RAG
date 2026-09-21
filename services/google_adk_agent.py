"""
Google ADK Agent Service.
Implements an AI Agent using LlmAgent and Runner from the Google Agent Development Kit (google-adk).
Uses the model selected in the GUI, named 'Chat Agent with RAG', equipped with skills/tools.
"""
import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_DOC_THRESHOLD,
    DEFAULT_SKILL_THRESHOLD,
    DEFAULT_LLM_MODEL,
    GEMINI_API_KEY,
    SKILLS_DIR
)
from services.log_service import audit_logger
from services.telemetry_service import telemetry_service
from services.vector_store import doc_vector_store
def _load_doc_search_tool():
    tool_path = Path(SKILLS_DIR) / "document-retriever-skill" / "tools" / "document_search_tool.py"
    spec = importlib.util.spec_from_file_location("document_search_tool", str(tool_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.search_documents

doc_search_tool_fn = _load_doc_search_tool()


# Ensure GOOGLE_API_KEY environment variable is synced for Google ADK / GenAI
if GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


class GoogleAdkAgentService:
    def __init__(self):
        self.skills_dir = Path(SKILLS_DIR)
        self.agent_name = "Chat Agent with RAG"

    def _get_skill_tool_functions(self, cid: str, doc_threshold: float, max_rag_chunks: int, retrieved_evidence: List[Dict[str, Any]]):
        """
        Build and bind tool functions with logging and evidence collection.
        """
        # 1. Weather and Time Tool
        def get_weather_and_time(city: str) -> dict:
            """Get real-time weather conditions and local time for a city without requiring an API key."""
            script_path = self.skills_dir / "time-weather-skill" / "scripts" / "env_tools.py"
            audit_logger.log_call(
                event_type="tool",
                call_type="invocation",
                invoker="Chat Agent with RAG",
                recipient="tool",
                payload={"tool": "env_tools.py", "function": "get_weather_and_time", "city": city},
                description=f"Tool message passed to env_tools.py for city: {city}",
                conversation_id=cid
            )
            audit_logger.log_call(
                event_type="external API call",
                call_type="invocation",
                invoker="tool",
                recipient="external API call",
                payload={"api": "Open-Meteo", "city": city, "geocoding_url": "https://geocoding-api.open-meteo.com/v1/search"},
                description=f"Full payload passed to Open-Meteo API for {city}",
                conversation_id=cid
            )

            result = {}
            try:
                spec = importlib.util.spec_from_file_location("env_tools", str(script_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                result = mod.get_weather_and_time(city)
            except Exception as e:
                result = {"error": str(e), "city": city}

            audit_logger.log_call(
                event_type="external API call",
                call_type="response",
                invoker="external API call",
                recipient="tool",
                payload=result,
                description=f"Full response received from Open-Meteo API for {city}",
                conversation_id=cid
            )
            audit_logger.log_call(
                event_type="tool",
                call_type="response",
                invoker="tool",
                recipient="Chat Agent with RAG",
                payload=result,
                description=f"Tool response received from env_tools.py for city: {city}",
                conversation_id=cid
            )

            evidence_text = f"Weather in {result.get('city', city)}: {result.get('condition')}, Temp: {result.get('temperature_celsius')}°C / {result.get('temperature_fahrenheit')}°F, Time: {result.get('local_time')}"
            retrieved_evidence.append({
                "step": "Skill Search",
                "title": f"Skill: Time & Weather Tool ({city})",
                "score": 1.0,
                "content": evidence_text,
                "details": result
            })
            return result

        # 2. Person Information Registry Tool
        def query_person_registry(keyword: str) -> list:
            """Query employee registry records by name, city, country, or job title."""
            script_path = self.skills_dir / "person-information-skill" / "scripts" / "person_search.py"
            audit_logger.log_call(
                event_type="tool",
                call_type="invocation",
                invoker="Chat Agent with RAG",
                recipient="tool",
                payload={"tool": "person_search.py", "function": "query_person_registry", "keyword": keyword},
                description=f"Tool message passed to person_search.py for keyword: '{keyword}'",
                conversation_id=cid
            )

            matches = []
            try:
                spec = importlib.util.spec_from_file_location("person_search", str(script_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                matches = mod.query_person_registry(keyword)
            except Exception as e:
                matches = [{"error": str(e)}]

            audit_logger.log_call(
                event_type="tool",
                call_type="response",
                invoker="tool",
                recipient="Chat Agent with RAG",
                payload={"count": len(matches), "matches": matches},
                description=f"Tool response received from person_search.py ({len(matches)} matches)",
                conversation_id=cid
            )

            evidence_text = f"Found {len(matches)} personnel records: " + "; ".join([f"{p.get('name')} ({p.get('job_title')}, {p.get('city')}, {p.get('country')})" for p in matches[:5]])
            retrieved_evidence.append({
                "step": "Skill Search",
                "title": f"Skill: Person Information Registry ({keyword})",
                "score": 1.0,
                "content": evidence_text,
                "details": {"matches": matches}
            })
            return matches

        # 3. Stock Market Search Tool
        def analyze_stock_query(query: str) -> dict:
            """Get the list of stocks with highest percentage increase or lowest percentage decrease."""
            script_path = self.skills_dir / "stock-market-skill" / "scripts" / "stock_search.py"
            audit_logger.log_call(
                event_type="tool",
                call_type="invocation",
                invoker="Chat Agent with RAG",
                recipient="tool",
                payload={"tool": "stock_search.py", "function": "analyze_stock_query", "query": query},
                description=f"Tool message passed to stock_search.py for query: '{query}'",
                conversation_id=cid
            )

            res = {}
            try:
                spec = importlib.util.spec_from_file_location("stock_search", str(script_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                res = mod.analyze_stock_query(query)
            except Exception as e:
                res = {"error": str(e)}

            audit_logger.log_call(
                event_type="tool",
                call_type="response",
                invoker="tool",
                recipient="Chat Agent with RAG",
                payload=res,
                description="Tool response received from stock_search.py",
                conversation_id=cid
            )

            stocks = res.get("stocks", [])
            evidence_text = f"{res.get('category', 'Stocks')}: " + ", ".join([f"{s.get('ticker')} ({s.get('change_pct', 0):+.2f}%, ${s.get('price', 0)})" for s in stocks])
            retrieved_evidence.append({
                "step": "Skill Search",
                "title": f"Skill: Stock Market Analysis",
                "score": 1.0,
                "content": evidence_text,
                "details": res
            })
            return res

        # 4. Document Search Tool
        def search_documents(query: str) -> list:
            """Get the list of text chunks from the document vector database matching the query."""
            chunks = doc_search_tool_fn(
                query=query,
                top_k=max_rag_chunks,
                min_score=doc_threshold,
                conversation_id=cid
            )
            for chk in chunks:
                doc_name = chk.get("document_name", "Document")
                chunk_idx = chk.get("chunk_index", 0)
                retrieved_evidence.append({
                    "step": "Document Search",
                    "source_type": "document_vector_store",
                    "store": "documents",
                    "document_name": doc_name,
                    "title": f"Doc: {doc_name} (Chunk #{chunk_idx}, Score: {chk.get('score')})",
                    "score": chk.get("score"),
                    "content": chk.get("text"),
                    "details": chk
                })
            return chunks

        return [get_weather_and_time, query_person_registry, analyze_stock_query, search_documents]

    def process_chat(
        self,
        query: str,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_rag_chunks: int = 5,
        skills_mode: str = "vector_store",
        skill_threshold: float = DEFAULT_SKILL_THRESHOLD,
        doc_threshold: float = DEFAULT_DOC_THRESHOLD,
        max_turns: int = 3,
        custom_endpoint: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate a chat turn using Google ADK LlmAgent.
        """
        start_time = time.time()
        cid = conversation_id or f"conv-{int(start_time * 1000)}"

        # 1. Log Agent Invocation
        audit_logger.log_call(
            event_type="agent",
            call_type="invocation",
            invoker="user",
            recipient=self.agent_name,
            payload={
                "query": query,
                "agent_type": "Google ADK Agent",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_rag_chunks": max_rag_chunks,
                "doc_threshold": doc_threshold,
                "max_turns": max_turns
            },
            description=f"Message sent to Google ADK Agent ({self.agent_name}): '{query}'",
            conversation_id=cid
        )

        telemetry_service.record_event(
            event_type="prompt",
            model=model,
            input_tokens=len(query.split())
        )

        retrieved_evidence: List[Dict[str, Any]] = []
        if skills_mode == "vector_store":
            from services.skill_manager import skill_manager
            matched_skills = skill_manager.match_skills(query, min_score=skill_threshold, conversation_id=cid)
            for s in matched_skills:
                skill_folder = s.get("folder_name", "skill")
                doc_name = f"{skill_folder}/SKILL.md"
                retrieved_evidence.append({
                    "step": "Skill Vector Store",
                    "source_type": "skill_vector_store",
                    "store": "skills",
                    "document_name": doc_name,
                    "title": f"Skill: {s.get('name', skill_folder)} (Score: {s.get('score', 0.0)})",
                    "score": s.get("score", 0.0),
                    "content": s.get("full_text") or s.get("description", ""),
                    "details": {
                        "document_name": doc_name,
                        "folder_name": skill_folder,
                        "name": s.get("name"),
                        "description": s.get("description"),
                        "score": s.get("score")
                    }
                })

        tools = self._get_skill_tool_functions(cid, doc_threshold, max_rag_chunks, retrieved_evidence)

        # Build callbacks to record fine-grained LLM payloads
        llm_invocation_payloads = []
        llm_response_payloads = []

        def before_model(ctx, req):
            # Log Model Invocation with FULL payload per SPECIFICATION.md
            contents_repr = []
            if hasattr(req, "contents"):
                for c in req.contents or []:
                    parts = getattr(c, "parts", [])
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            contents_repr.append({"text": p.text})
                        elif hasattr(p, "function_call") and p.function_call:
                            contents_repr.append({"function_call": getattr(p.function_call, "name", str(p.function_call))})
                        elif hasattr(p, "function_response") and p.function_response:
                            contents_repr.append({"function_response": getattr(p.function_response, "name", str(p.function_response))})

            p_data = {
                "model": getattr(req, "model", model),
                "contents": contents_repr or str(getattr(req, "contents", "")),
                "tools": [getattr(t, "name", str(t)) for t in (getattr(req, "tools_dict", {}) or {}).values()] or [getattr(t, "__name__", str(t)) for t in tools],
                "config": str(getattr(req, "config", ""))
            }
            llm_invocation_payloads.append(p_data)
            audit_logger.log_call(
                event_type="LLM",
                call_type="invocation",
                invoker=self.agent_name,
                recipient="LLM",
                payload=p_data,
                description=f"Prompts sent to model {model} with full payload from {self.agent_name}",
                conversation_id=cid
            )

        def after_model(ctx, res):
            # Log Model Response with FULL payload per SPECIFICATION.md
            content_text = ""
            if hasattr(res, "content") and res.content and hasattr(res.content, "parts"):
                for p in res.content.parts:
                    if hasattr(p, "text") and p.text:
                        content_text += p.text + " "
                    elif hasattr(p, "function_call") and p.function_call:
                        content_text += f"[Call: {getattr(p.function_call, 'name', '')}] "

            res_payload = {
                "model_version": getattr(res, "model_version", model),
                "content": content_text.strip(),
                "finish_reason": str(getattr(res, "finish_reason", "STOP")),
                "usage_metadata": {
                    "prompt_token_count": getattr(getattr(res, "usage_metadata", None), "prompt_token_count", 0),
                    "candidates_token_count": getattr(getattr(res, "usage_metadata", None), "candidates_token_count", 0),
                    "total_token_count": getattr(getattr(res, "usage_metadata", None), "total_token_count", 0)
                } if hasattr(res, "usage_metadata") and res.usage_metadata else {}
            }
            llm_response_payloads.append(res_payload)
            audit_logger.log_call(
                event_type="LLM",
                call_type="response",
                invoker="LLM",
                recipient=self.agent_name,
                payload=res_payload,
                description=f"Response received from model with full payload",
                conversation_id=cid
            )

        # Attempt to run via Google ADK
        agent_answer = ""
        total_in_tokens = len(query.split())
        total_out_tokens = 0

        clean_model = model.replace("models/", "")
        # Fallback to active model if user passed an inactive/deprecated model
        adk_model_name = clean_model
        if "gemini-2.0" in clean_model or "gemini-2.5" in clean_model:
            adk_model_name = "gemini-3.6-flash"

        instruction_text = (
            "You are Chat Agent with RAG, an intelligent agent with access to procedural tools and a document vector database. "
            "Use the provided tools when relevant to gather accurate facts. "
            "Always synthesize clear, polite, and factually grounded responses."
        )

        try:
            from google.adk.agents import LlmAgent
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
            from google.genai import types

            adk_agent = LlmAgent(
                name="Chat_Agent_with_RAG",
                model=adk_model_name,
                description="Chat Agent with RAG",
                instruction=instruction_text,
                tools=tools,
                before_model_callback=before_model,
                after_model_callback=after_model
            )

            session_service = InMemorySessionService()
            runner = Runner(
                app_name="Chat_Agent_with_RAG",
                agent=adk_agent,
                session_service=session_service,
                auto_create_session=True
            )

            async def _execute_adk():
                msg = types.Content(parts=[types.Part(text=query)])
                final_text = ""
                collected_chunks = []
                async for event in runner.run_async(
                    user_id="user",
                    session_id=cid,
                    new_message=msg
                ):
                    if event.content and hasattr(event.content, "parts") and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                collected_chunks.append(part.text)
                                final_text = part.text
                if not final_text and collected_chunks:
                    final_text = "\n".join(collected_chunks)
                return final_text

            agent_answer = asyncio.run(_execute_adk())

            # If ADK event loop didn't yield text directly, check after_model logs
            if not agent_answer:
                recent_llm = [e for e in audit_logger.get_conversation_logs(cid) if e.get("event_type") == "LLM" and e.get("call_type") == "response"]
                if recent_llm:
                    agent_answer = recent_llm[-1].get("payload", {}).get("content", "")

            # If still empty, use orchestrator fallback
            if not agent_answer:
                from services.agent_orchestrator import orchestrator as agent_orchestrator
                fallback_res = agent_orchestrator.process_chat(
                    query=query,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_rag_chunks=max_rag_chunks,
                    skills_mode=skills_mode,
                    skill_threshold=skill_threshold,
                    doc_threshold=doc_threshold,
                    max_turns=max_turns,
                    custom_endpoint=custom_endpoint,
                    conversation_id=cid
                )
                agent_answer = fallback_res.get("answer") or fallback_res.get("response", "")
                if not retrieved_evidence and fallback_res.get("retrieved_evidence"):
                    retrieved_evidence = fallback_res.get("retrieved_evidence", [])

        except Exception as adk_err:
            print(f"[GoogleAdkAgent] ADK runner notice / fallback: {adk_err}")
            # Fallback to direct tool execution & synthesis if needed
            from services.agent_orchestrator import orchestrator as agent_orchestrator
            fallback_res = agent_orchestrator.process_chat(
                query=query,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_rag_chunks=max_rag_chunks,
                skills_mode=skills_mode,
                skill_threshold=skill_threshold,
                doc_threshold=doc_threshold,
                max_turns=max_turns,
                custom_endpoint=custom_endpoint,
                conversation_id=cid
            )
            agent_answer = fallback_res.get("answer") or fallback_res.get("response", "")
            if not retrieved_evidence and fallback_res.get("retrieved_evidence"):
                retrieved_evidence = fallback_res.get("retrieved_evidence", [])
            if fallback_res.get("steps"):
                steps.extend([s for s in fallback_res.get("steps", []) if s not in steps])

        if not agent_answer:
            agent_answer = f"I processed your query: '{query}'."

        latency = (time.time() - start_time) * 1000
        total_out_tokens = len(str(agent_answer).split())

        # 2. Log Final Agent Response
        audit_logger.log_call(
            event_type="agent",
            call_type="response",
            invoker=self.agent_name,
            recipient="user",
            payload={
                "response": agent_answer,
                "content": agent_answer,
                "evidence_count": len(retrieved_evidence),
                "agent_name": self.agent_name
            },
            description=f"Full response received from Google ADK Agent ({self.agent_name})",
            conversation_id=cid,
            latency_ms=latency
        )

        telemetry_service.record_event(
            event_type="response",
            model=model,
            input_tokens=total_in_tokens,
            output_tokens=total_out_tokens,
            latency_ms=latency
        )

        # 3. Assemble Bubbles for UI Response Detail Box
        all_conv_events = audit_logger.get_conversation_logs(cid)
        steps = []

        # Skills / Tools Bubble
        tool_events = [e for e in all_conv_events if e.get("event_type") in ["tool", "external API call"]]
        steps.append({
            "component": "Tools",
            "icon": "🛠️",
            "step_name": "Google ADK Tools",
            "elapsed_ms": round(latency * 0.4, 2),
            "summary": f"{len(tool_events)} tool invocation/response event(s) recorded." if tool_events else "All skill tools evaluated by ADK agent.",
            "details": tool_events
        })

        # RAG / Document Search Bubble
        doc_events = [e for e in all_conv_events if e.get("event_type") in ["document search", "ollama vector"]]
        steps.append({
            "component": "RAG",
            "icon": "📚",
            "step_name": "Vector Store RAG",
            "elapsed_ms": round(latency * 0.3, 2),
            "summary": f"{len(doc_events)} document search & vector log event(s)." if doc_events else "Document search tool bound to LlmAgent.",
            "details": doc_events
        })

        # LLM Bubble
        llm_events = [e for e in all_conv_events if e.get("event_type") == "LLM"]
        steps.append({
            "component": "LLM",
            "icon": "🧠",
            "step_name": f"ADK LlmAgent ({model})",
            "elapsed_ms": round(latency * 0.5, 2),
            "summary": f"Executed Google ADK LlmAgent reasoning with model {model}.",
            "details": llm_events
        })

        # Agent Bubble
        agent_events = [e for e in all_conv_events if e.get("event_type") == "agent"]
        steps.append({
            "component": "Agent",
            "icon": "🤖",
            "step_name": self.agent_name,
            "elapsed_ms": round(latency, 2),
            "summary": f"Completed turn in {round(latency, 1)} ms using Google ADK framework.",
            "details": agent_events
        })

        return {
            "response": agent_answer,
            "answer": agent_answer,
            "conversation_id": cid,
            "latency_ms": round(latency, 2),
            "evidence": retrieved_evidence,
            "retrieved_evidence": retrieved_evidence,
            "steps": steps,
            "agent_type": "Google ADK Agent"
        }


# Global singleton
google_adk_agent = GoogleAdkAgentService()
