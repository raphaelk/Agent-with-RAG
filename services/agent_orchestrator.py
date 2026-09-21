"""
Agent Orchestrator Service.
Coordinates intent routing, skill selection, RAG document search,
context evidence grouping by step, LLM synthesis, logging, and telemetry.
"""
import uuid
import time
from typing import Dict, Any, List, Optional

from config import (
    MIN_SKILL_SCORE,
    MIN_RAG_DOC_SCORE,
    DEFAULT_LLM_MODEL,
    DEFAULT_SKILL_THRESHOLD,
    DEFAULT_DOC_THRESHOLD,
    DEFAULT_MAX_TURNS,
    MAX_TURNS_LIMIT,
    SKILLS_DIR
)
import json
import re
import importlib.util
from pathlib import Path
from services.skill_manager import skill_manager
from services.vector_store import doc_vector_store
from services.llm_service import llm_service
from services.log_service import audit_logger
from services.telemetry_service import telemetry_service
from services.ollama_service import ollama_service

def _load_doc_search_tool():
    tool_path = Path(SKILLS_DIR) / "document-retriever-skill" / "tools" / "document_search_tool.py"
    spec = importlib.util.spec_from_file_location("document_search_tool", str(tool_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.search_documents

search_documents = _load_doc_search_tool()

def _parse_tool_call_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON tool execution directive per SPECIFICATION.md:
    {
      "tool": "person_search.query_person_registry",
      "arguments": {
        "keyword": "Lucas Dubois",
        "field": "name"
      }
    }
    Includes fallback heuristics for markdown codeblocks and text directives.
    """
    clean_text = (text or "").strip()
    # 1. Search for fenced code blocks ```json ... ``` or ``` ... ```
    cb_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if cb_match:
        try:
            data = json.loads(cb_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 2. Try direct json.loads
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 3. Regex search for JSON object with "tool" key
    match = re.search(r"\{\s*\"tool\"[\s\S]*?\}", clean_text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 4. Regex search for any JSON object
    for m in re.finditer(r"\{[^{}]*\}", clean_text):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "tool" in data:
                return data
        except Exception:
            pass

    # Fallback heuristics for text/directive responses
    lower = clean_text.lower()
    if "no_tool_needed" in lower or "no tool needed" in lower or "no tool required" in lower:
        return {"tool": "none", "arguments": {}}
    if "execute_document_search" in lower or "document search" in lower:
        return {"tool": "document_search_tool.search_documents", "arguments": {}}
    if "execute_tool:" in clean_text:
        parts = clean_text.split("EXECUTE_TOOL:", 1)
        tool_name = parts[1].strip().split("\n")[0].strip()
        return {"tool": tool_name, "arguments": {}}
    if "execute_tool" in lower:
        return {"tool": "execute_tool", "arguments": {}}

    return {"tool": "none", "arguments": {}}

class AgentOrchestrator:
    def process_chat(
        self,
        query: str,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_rag_chunks: int = 5,
        custom_endpoint: Optional[str] = None,
        conversation_id: Optional[str] = None,
        skills_mode: str = "vector_store",
        skill_threshold: float = DEFAULT_SKILL_THRESHOLD,
        doc_threshold: float = DEFAULT_DOC_THRESHOLD,
        max_turns: int = DEFAULT_MAX_TURNS
    ) -> Dict[str, Any]:
        """
        Main multi-step cognitive pipeline:
        1. Log user prompt & record prompt telemetry
        2. Skill resolution based on skills_mode:
           - 'vector_store': Query skill vector store with skill_threshold
           - 'llm_selected': Ask LLM to select relevant skill from skills/ directory
           - specific skill folder: Use that skill directly
        3. If no skill found: send user query to LLM using simple system prompt as assistant
        4. If skill found: send user message and skill to LLM to get tool execution plan
        5. If procedural tool or document search is directed, execute tool/search, feed results back to LLM
           Repeat until final answer is received, limited to max_turns loops (<= MAX_TURNS_LIMIT)
        6. Return grounded response with steps and logs
        """
        cid = conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        steps: List[Dict[str, Any]] = []

        # Step 1: Log User Message to Agent (Invocation)
        audit_logger.log_call(
            event_type="agent",
            call_type="invocation",
            invoker="user",
            recipient="agent",
            payload={
                "agent_type": "Custom Agent",
                "query": query,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_rag_chunks": max_rag_chunks,
                "skills_mode": skills_mode,
                "skill_threshold": skill_threshold,
                "doc_threshold": doc_threshold,
                "max_turns": max_turns
            },
            description=f"Message sent to agent: '{query}'",
            conversation_id=cid
        )

        telemetry_service.record_event(
            event_type="prompt",
            model=model,
            input_tokens=len(query.split())
        )

        retrieved_evidence: List[Dict[str, Any]] = []
        skill_context_snippets = []
        doc_context_snippets = []
        step_skill_elapsed = 0.0
        matched_skills: List[Dict[str, Any]] = []
        vectorizer_name = ollama_service.current_model

        # Step 2: Skill Resolution Mode
        if skills_mode == "vector_store":
            step_skill_start = time.time()
            audit_logger.log_call(
                event_type="skill search",
                call_type="invocation",
                invoker="agent",
                recipient="skill search",
                payload={"query": query, "min_score": skill_threshold, "vectorizer": vectorizer_name},
                description=f"Message sent to skill search for: '{query}' using vectorizer {vectorizer_name} (threshold: {skill_threshold})",
                conversation_id=cid
            )

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

            audit_logger.log_call(
                event_type="skill search",
                call_type="response",
                invoker="skill search",
                recipient="agent",
                payload={
                    "vectorizer": vectorizer_name,
                    "vectorizer_response": {"status": "success", "model": vectorizer_name},
                    "matched_skills": [{"name": s.get("name"), "folder_name": s.get("folder_name"), "score": s.get("score")} for s in matched_skills],
                    "count": len(matched_skills)
                },
                description=f"Response received from skill search: {len(matched_skills)} skill(s) matched using {vectorizer_name}",
                conversation_id=cid
            )
            step_skill_elapsed = round((time.time() - step_skill_start) * 1000, 2)

        elif skills_mode == "llm_selected":
            step_skill_start = time.time()
            all_skills = skill_manager.get_all_skills()
            skills_listing = "\n".join([f"- Folder: {s['folder_name']} | Name: {s['name']} | Description: {s['description']}" for s in all_skills])

            sel_prompt = (
                f"User Question: {query}\n\n"
                f"Available Skills:\n{skills_listing}\n\n"
                f"Determine which skill (if any) should be used to answer the question. "
                f"If the question is general knowledge or none of the skills are relevant, respond with 'NONE'. "
                f"Otherwise, respond with ONLY the exact folder name of the selected skill."
            )

            sel_res = llm_service.generate_response(
                prompt=sel_prompt,
                system_instruction="You are a skill router. Select the best matching skill folder name or respond with NONE.",
                model=model,
                temperature=0.1,
                max_tokens=64,
                custom_endpoint=custom_endpoint,
                conversation_id=cid
            )
            sel_text = sel_res.get("content", "").strip()

            chosen_skill = None
            for s in all_skills:
                if s["folder_name"].lower() in sel_text.lower() or s["name"].lower() in sel_text.lower():
                    chosen_skill = skill_manager.get_skill_by_folder(s["folder_name"])
                    break

            if chosen_skill:
                matched_skills = [chosen_skill]
                skill_folder = chosen_skill.get("folder_name", "skill")
                doc_name = f"{skill_folder}/SKILL.md"
                retrieved_evidence.append({
                    "step": "Skill Selection",
                    "source_type": "skill_vector_store",
                    "store": "skills",
                    "document_name": doc_name,
                    "title": f"Skill: {chosen_skill.get('name', skill_folder)}",
                    "score": chosen_skill.get("score", 1.0),
                    "content": chosen_skill.get("full_text") or chosen_skill.get("description", ""),
                    "details": {
                        "document_name": doc_name,
                        "folder_name": skill_folder,
                        "name": chosen_skill.get("name"),
                        "description": chosen_skill.get("description"),
                        "score": chosen_skill.get("score", 1.0)
                    }
                })
            step_skill_elapsed = round((time.time() - step_skill_start) * 1000, 2)

        else:
            # Specific skill folder selected by user
            chosen_skill = skill_manager.get_skill_by_folder(skills_mode)
            if chosen_skill:
                matched_skills = [chosen_skill]
                skill_folder = chosen_skill.get("folder_name", "skill")
                doc_name = f"{skill_folder}/SKILL.md"
                retrieved_evidence.append({
                    "step": "Skill Selection",
                    "source_type": "skill_vector_store",
                    "store": "skills",
                    "document_name": doc_name,
                    "title": f"Skill: {chosen_skill.get('name', skill_folder)}",
                    "score": chosen_skill.get("score", 1.0),
                    "content": chosen_skill.get("full_text") or chosen_skill.get("description", ""),
                    "details": {
                        "document_name": doc_name,
                        "folder_name": skill_folder,
                        "name": chosen_skill.get("name"),
                        "description": chosen_skill.get("description"),
                        "score": chosen_skill.get("score", 1.0)
                    }
                })

        # Step 3: Handle Case Where No Skill is Found / Selected
        step_plan_elapsed = 0.0
        step_tool_elapsed = 0.0
        step_rag_elapsed = 0.0
        step_syn_elapsed = 0.0
        tool_plan_text = ""
        llm_result: Dict[str, Any] = {}
        top_skills = matched_skills[:2] if matched_skills else []
        highest_skill = top_skills[0] if top_skills else None

        if not top_skills:
            # Per SPECIFICATION.md: "If no skill is found, send the user query to the LLM using the simple system prompt as an assistant to answer the question."
            step_syn_start = time.time()
            llm_result = llm_service.generate_response(
                prompt=f"User Question: {query}\n\nAnswer the user directly and concisely.",
                system_instruction="You are a helpful AI assistant. Answer the question directly and concisely.",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                custom_endpoint=custom_endpoint,
                conversation_id=cid
            )
            step_syn_elapsed = round((time.time() - step_syn_start) * 1000, 2)
            agent_answer = llm_result.get("content", "")

        else:
            # Step 4: Skills Found - Send top 2 skills to LLM to get instruction or plan for tool execution
            # Per SPECIFICATION.md:
            # - The system prompt should ask the LLM to determine if any of the tools should be invoked to gather more information to answer the question.
            # - The system prompt should ask the LLM to respond with JSON format indicating the tool to be executed and the arguments to be passed to the tool.
            step_plan_start = time.time()
            skills_text_blocks = []
            for i, s in enumerate(top_skills, 1):
                folder = s.get("folder_name", "")
                tool_func = skill_manager.get_skill_tool_signature(s)

                skills_text_blocks.append(
                    f"- Skill #{i}: {s['name']} (Score: {s.get('score', 0.0)})\n"
                    f"  Folder: {folder}\n"
                    f"  Available Tool Function: {tool_func}\n"
                    f"  Description: {s['description']}\n"
                    f"  SOP / Instructions:\n{s.get('full_text', '')[:450]}"
                )
            skills_text = "\n\n".join(skills_text_blocks)

            plan_prompt = (
                f"User Question: {query}\n\n"
                f"Top Matching Skills:\n{skills_text}\n\n"
                f"Determine if any of the tools from the matching skills should be invoked to gather more information to answer the question.\n"
                f"Respond with JSON format indicating the tool to be executed and the arguments to be passed to the tool.\n"
                f"For example:\n"
                f"{{\n"
                f'  "tool": "person_search.query_person_registry",\n'
                f'  "arguments": {{\n'
                f'    "keyword": "Lucas Dubois",\n'
                f'    "field": "name"\n'
                f'  }}\n'
                f"}}\n"
                f"If document retrieval from the document vector store is required, use:\n"
                f"{{\n"
                f'  "tool": "document_search_tool.search_documents",\n'
                f'  "arguments": {{\n'
                f'    "query": "{query}",\n'
                f'    "top_k": {max_rag_chunks}\n'
                f'  }}\n'
                f"}}\n"
                f"If no tool should be invoked, respond with:\n"
                f"{{\n"
                f'  "tool": "none",\n'
                f'  "arguments": {{}}\n'
                f"}}"
            )

            plan_system_instruction = (
                "You are an AI Agent Orchestrator. Determine if any of the tools should be invoked to gather more information to answer the question. "
                "Respond with JSON format indicating the tool to be executed and the arguments to be passed to the tool."
            )

            plan_res = llm_service.generate_response(
                prompt=plan_prompt,
                system_instruction=plan_system_instruction,
                model=model,
                temperature=temperature,
                max_tokens=2048,
                custom_endpoint=custom_endpoint,
                conversation_id=cid
            )
            tool_plan_text = plan_res.get("content", "")
            step_plan_elapsed = round((time.time() - step_plan_start) * 1000, 2)

            # Execution loop limited to max_turns (no more than MAX_TURNS_LIMIT)
            turns_limit = min(max(1, int(max_turns)), MAX_TURNS_LIMIT)
            current_turn = 1
            has_doc_skill = any(
                ("retriever" in s.get("folder_name", "").lower() or "document" in s.get("folder_name", "").lower())
                for s in top_skills
            )
            latest_plan_text = tool_plan_text
            agent_answer = ""
            executed_skills = set()
            executed_tools = set()

            while current_turn <= turns_limit:
                plan_data = _parse_tool_call_json(latest_plan_text)
                tool_to_execute = (plan_data.get("tool") or "").strip()
                tool_args = plan_data.get("arguments") or {}
                tool_executed_this_turn = False

                # Check if tool is none
                if not tool_to_execute or tool_to_execute.lower() in ["none", "null", "false", "no_tool_needed"]:
                    pass
                elif has_doc_skill and any(term in tool_to_execute.lower() for term in ["document", "retriever", "search_documents"]):
                    # 1. Document Search if directed
                    step_rag_start = time.time()
                    doc_query = tool_args.get("query") or query
                    doc_top_k = int(tool_args.get("top_k") or max_rag_chunks)
                    doc_chunks = search_documents(
                        query=doc_query,
                        top_k=doc_top_k,
                        min_score=doc_threshold,
                        conversation_id=cid,
                        invoker="agent"
                    )
                    for chunk in doc_chunks:
                        doc_name = chunk.get("document_name", "Document")
                        chunk_idx = chunk.get("chunk_index", 0)
                        evidence_item = {
                            "step": "Document Search",
                            "source_type": "document_vector_store",
                            "store": "documents",
                            "document_name": doc_name,
                            "title": f"Doc: {doc_name} (Chunk #{chunk_idx}, Score: {chunk['score']})",
                            "score": chunk["score"],
                            "content": chunk["text"],
                            "details": chunk
                        }
                        retrieved_evidence.append(evidence_item)
                        doc_context_snippets.append(f"[From {doc_name}]:\n{chunk['text']}")
                    step_rag_elapsed += round((time.time() - step_rag_start) * 1000, 2)
                    tool_executed_this_turn = True
                    executed_tools.add(tool_to_execute)

                else:
                    # 2. Procedural Tools from matching skills
                    for s in top_skills:
                        folder_name = s.get("folder_name", "")
                        if "retriever" in folder_name.lower() or "document" in folder_name.lower():
                            continue
                        if folder_name in executed_skills:
                            continue
                        is_match = skill_manager.is_skill_tool_match(s, tool_to_execute) or not latest_plan_text
                        if is_match:
                            executed_skills.add(folder_name)
                            executed_tools.add(tool_to_execute)
                            step_tool_start = time.time()
                            exec_result = skill_manager.execute_skill(
                                s,
                                query,
                                conversation_id=cid,
                                arguments=tool_args
                            )
                            evidence_item = {
                                "step": "Tool Execution",
                                "source_type": "procedural_tool",
                                "document_name": f"{folder_name}/tool_output",
                                "title": f"Tool Output: {s['name']}",
                                "score": s.get("score", 0.0),
                                "content": exec_result.get("evidence_text", ""),
                                "details": exec_result.get("result_data", {})
                            }
                            retrieved_evidence.append(evidence_item)
                            skill_context_snippets.append(f"[{s['name']} Execution Output]: {exec_result.get('evidence_text', '')}")
                            step_tool_elapsed += round((time.time() - step_tool_start) * 1000, 2)
                            tool_executed_this_turn = True
                            break

                # Build cumulative context from tool execution outputs
                context_block = ""
                if tool_plan_text:
                    context_block += f"--- Orchestrator Tool Plan ---\n{tool_plan_text.strip()}\n\n"
                if skill_context_snippets:
                    context_block += "--- Skill Tool Execution Outputs ---\n" + "\n\n".join(skill_context_snippets) + "\n\n"
                if doc_context_snippets:
                    context_block += "--- Document Vector DB Chunks ---\n" + "\n\n".join(doc_context_snippets) + "\n"

                # If a tool was executed and we have turns left, check if the LLM needs another tool
                if tool_executed_this_turn and current_turn < turns_limit:
                    next_check_prompt = (
                        f"User Question: {query}\n\n"
                        f"Top Matching Skills:\n{skills_text}\n\n"
                        f"Results from tool execution so far:\n"
                        f"{context_block.strip()}\n\n"
                        f"Determine if any additional tool should be invoked to gather more information to answer the question. "
                        f"Respond with JSON format indicating the tool and arguments.\n"
                        f"If sufficient information has been gathered to formulate the final answer, respond with:\n"
                        f'{{"tool": "none", "arguments": {{}}}}'
                    )
                    next_res = llm_service.generate_response(
                        prompt=next_check_prompt,
                        system_instruction=plan_system_instruction,
                        model=model,
                        temperature=temperature,
                        max_tokens=512,
                        custom_endpoint=custom_endpoint,
                        conversation_id=cid
                    )
                    next_plan_data = _parse_tool_call_json(next_res.get("content", ""))
                    next_tool = (next_plan_data.get("tool") or "").strip()
                    if next_tool and next_tool.lower() not in ["none", "null", "false", "no_tool_needed"] and next_tool not in executed_tools:
                        latest_plan_text = next_res.get("content", "")
                        current_turn += 1
                        continue

                # Per SPECIFICATION.md:
                # "The last llm call should use typical system prompt as an assistant to answer the question.
                # The final output of the agent is the response from this last llm call."
                system_instruction = "You are a helpful assistant. Answer the user's question accurately based on the provided context."

                if context_block.strip():
                    full_prompt = (
                        f"User Question: {query}\n\n"
                        f"=== RETRIEVED CONTEXT EVIDENCE ===\n"
                        f"{context_block.strip()}\n"
                        f"=== END CONTEXT ===\n\n"
                        f"Provide a helpful, precise, and factually grounded response to the user's question."
                    )
                else:
                    full_prompt = f"User Question: {query}\n\nAnswer the user directly and concisely."

                step_syn_start = time.time()
                llm_result = llm_service.generate_response(
                    prompt=full_prompt,
                    system_instruction=system_instruction,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    custom_endpoint=custom_endpoint,
                    conversation_id=cid
                )
                step_syn_elapsed += round((time.time() - step_syn_start) * 1000, 2)
                agent_answer = llm_result.get("content", "")
                break

        latency = (time.time() - start_time) * 1000

        # Step 5: Log final Agent response
        audit_logger.log_call(
            event_type="agent",
            call_type="response",
            invoker="agent",
            recipient="user",
            payload={
                "response": agent_answer,
                "model": model,
                "evidence_count": len(retrieved_evidence),
                "top_skills": [s.get("name") for s in top_skills]
            },
            description=f"Full response received from the agent ({len(retrieved_evidence)} evidence items, {len(top_skills)} skill(s) evaluated)",
            conversation_id=cid,
            latency_ms=latency
        )

        # Step 6: Telemetry Recording
        total_in_tokens = llm_result.get("input_tokens", 0)
        total_out_tokens = llm_result.get("output_tokens", 0)
        telemetry_service.record_event(
            event_type="response",
            model=model,
            input_tokens=total_in_tokens,
            output_tokens=total_out_tokens,
            latency_ms=latency
        )

        # Step 7: Assemble Event Grouping / Collapsible Steps
        all_conv_events = audit_logger.get_conversation_logs(cid)
        steps = []

        # 1. Skills Component
        skill_logs = [e for e in all_conv_events if e.get("event_type") in ["skill search", "ollama vector"] and (not tool_plan_text or e.get("timestamp") <= all_conv_events[min(len(all_conv_events)-1, 3)].get("timestamp"))]
        top_skills_str = ", ".join([f"{s['name']} ({s.get('score', 0.0)})" for s in top_skills])
        steps.append({
            "component": "Skills",
            "icon": "⚡",
            "step_name": "Skills",
            "elapsed_ms": step_skill_elapsed,
            "summary": f"Scanned skill database ({vectorizer_name}). Top matches: {top_skills_str}" if top_skills else "No skills matched above threshold.",
            "logs": [e for e in all_conv_events if e.get("event_type") == "skill search"]
        })

        # 2. Agent Component (tool planning by orchestrator using top scoring skills)
        if top_skills:
            steps.append({
                "component": "Agent",
                "icon": "🤖",
                "step_name": "Agent",
                "elapsed_ms": step_plan_elapsed,
                "summary": f"Agent evaluated top {len(top_skills)} skill(s) ({', '.join(s['name'] for s in top_skills)}) and planned tool directives with {model}.",
                "logs": [e for e in all_conv_events if e.get("event_type") == "LLM" and ("Top Matching Skills" in str(e.get("payload", "")) or "Highest Matching Skill" in str(e.get("payload", "")))]
            })

        # 3. RAG Component (if document search was performed)
        if step_rag_elapsed > 0:
            rag_logs = [e for e in all_conv_events if e.get("event_type") == "document search"]
            steps.append({
                "component": "RAG",
                "icon": "📚",
                "step_name": "RAG",
                "elapsed_ms": step_rag_elapsed,
                "summary": f"Retrieved {len(doc_chunks)} chunk(s) from document vector store exceeding {MIN_RAG_DOC_SCORE}.",
                "logs": rag_logs
            })

        # 4. Tools Component (if procedural tool was executed)
        if step_tool_elapsed > 0:
            tool_logs = [e for e in all_conv_events if e.get("event_type") in ["tool", "external API call"]]
            executed_names = ", ".join([s["name"] for s in top_skills if s.get("folder_name") in executed_skills]) or (top_skills[0]["name"] if top_skills else "Tool")
            steps.append({
                "component": "Tools",
                "icon": "🛠️",
                "step_name": "Tools",
                "elapsed_ms": step_tool_elapsed,
                "summary": f"Executed procedural tool for '{executed_names}'.",
                "logs": tool_logs
            })

        # 5. LLM Synthesis Component
        synthesis_logs = [e for e in all_conv_events if e.get("event_type") == "LLM" and ("RETRIEVED CONTEXT EVIDENCE" in str(e.get("payload", "")) or "Answer the user directly" in str(e.get("payload", "")))]
        if not synthesis_logs:
            synthesis_logs = [e for e in all_conv_events if e.get("event_type") == "LLM"][-1:]
        steps.append({
            "component": "LLM",
            "icon": "🧠",
            "step_name": "LLM",
            "elapsed_ms": step_syn_elapsed,
            "summary": f"Synthesized final grounded response with {model}.",
            "logs": synthesis_logs
        })

        return {
            "conversation_id": cid,
            "answer": agent_answer,
            "model_used": model,
            "retrieved_evidence": retrieved_evidence,
            "steps": steps,
            "latency_ms": round(latency, 2),
            "tokens": {
                "input": llm_result.get("input_tokens", 0),
                "output": llm_result.get("output_tokens", 0)
            }
        }

# Global singleton
orchestrator = AgentOrchestrator()
agent_orchestrator = orchestrator

