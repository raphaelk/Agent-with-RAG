"""
Skill Manager Service.
Scans skills/ directory, parses SKILL.md files, vectors skill metadata,
and dispatches execution to skill scripts and SOP workflows.
"""
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from config import SKILLS_DIR, MIN_SKILL_SCORE
from services.vector_store import skill_vector_store, doc_vector_store
from services.ollama_service import ollama_service
from services.log_service import audit_logger

def parse_skill_markdown(skill_path: Path) -> Dict[str, Any]:
    """
    Parse SKILL.md file with YAML frontmatter and markdown body.
    """
    skill_file = skill_path / "SKILL.md"
    if not skill_file.exists():
        return {}

    raw_text = skill_file.read_text(encoding="utf-8")
    frontmatter = {}
    body = raw_text

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw_text, re.DOTALL)
    if match:
        fm_text, body = match.groups()
        try:
            frontmatter = yaml.safe_load(fm_text) or {}
        except Exception as e:
            print(f"Error parsing YAML frontmatter in {skill_file}: {e}")

    # Fallback extraction if frontmatter missing
    name = frontmatter.get("name") or skill_path.name
    description = frontmatter.get("description", "")
    trigger_queries = frontmatter.get("Trigger Queries") or frontmatter.get("trigger_queries", [])

    return {
        "folder_name": skill_path.name,
        "path": str(skill_path),
        "name": name,
        "description": description,
        "trigger_queries": trigger_queries,
        "full_text": raw_text,
        "body": body
    }

class SkillManager:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)

    def scan_and_load_skills(self) -> Dict[str, Any]:
        """
        Scan the skills/ folder and load new skills that are not in the skill database.
        Skills already in the database should not be re-loaded.
        """
        if not self.skills_dir.exists():
            return {"loaded_skills": [], "skipped_skills": [], "total_skills": 0}

        existing_data = skill_vector_store._load()
        existing_skill_folders = {
            c.get("metadata", {}).get("folder_name")
            for c in existing_data.get("chunks", [])
        }

        newly_loaded = []
        skipped = []

        for item in sorted(self.skills_dir.iterdir()):
            if not item.is_dir():
                continue
            skill_folder_name = item.name

            parsed = parse_skill_markdown(item)
            if not parsed:
                continue

            existing_chunk = next((c for c in existing_data.get("chunks", []) if c.get("id") == f"skill-{skill_folder_name}"), None)
            if (existing_chunk 
                and existing_chunk.get("model") == ollama_service.current_model 
                and len(existing_chunk.get("vector", [])) in [384, 768, 1024]
                and existing_chunk.get("text") == parsed.get("full_text")):
                skipped.append(skill_folder_name)
                continue

            # If existing but needs re-embedding due to model or content change, remove prior chunk
            if existing_chunk:
                existing_data["chunks"] = [c for c in existing_data["chunks"] if c.get("id") != f"skill-{skill_folder_name}"]

            # Embed name, description, and trigger queries per SPECIFICATION.md
            triggers_str = " ".join(parsed.get("trigger_queries", []))
            embed_text = f"{parsed['name']}. {parsed['description']}. Trigger Queries: {triggers_str}" if triggers_str else f"{parsed['name']}. {parsed['description']}"

            vector = ollama_service.generate_embedding(embed_text)

            # Store the vectors and complete text of the SKILL.md file as one record
            chunk_record = {
                "id": f"skill-{skill_folder_name}",
                "content_hash": skill_folder_name,
                "text": parsed["full_text"],
                "vector": vector,
                "model": ollama_service.current_model,
                "metadata": {
                    "folder_name": skill_folder_name,
                    "name": parsed["name"],
                    "description": parsed["description"],
                    "trigger_queries": parsed["trigger_queries"],
                    "path": parsed["path"],
                    "embed_text": embed_text
                }
            }

            existing_data["chunks"].append(chunk_record)
            existing_data["documents"][skill_folder_name] = {
                "chunks_count": 1,
                "total_characters": len(parsed["full_text"]),
                "source": "skill_definition"
            }
            newly_loaded.append(skill_folder_name)

        skill_vector_store._save(existing_data)

        audit_logger.log_event(
            event_type="Skill DB Update",
            invoker="System",
            target="SkillManager",
            payload={"scanned_folder": str(self.skills_dir)},
            response={"newly_loaded": newly_loaded, "skipped": skipped},
            description=f"Loaded {len(newly_loaded)} new skills into skill database"
        )

        return {
            "loaded_skills": newly_loaded,
            "skipped_skills": skipped,
            "total_skills": len(existing_data.get("chunks", []))
        }

    def get_all_skills(self) -> List[Dict[str, Any]]:
        """Return parsed metadata for all skills in skills/ directory."""
        skills = []
        if not self.skills_dir.exists():
            return skills
        for item in sorted(self.skills_dir.iterdir()):
            if item.is_dir():
                parsed = parse_skill_markdown(item)
                if parsed:
                    skills.append(parsed)
        return skills

    def get_skill_by_folder(self, folder_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve single skill by folder name."""
        skill_path = self.skills_dir / folder_name
        if skill_path.exists() and skill_path.is_dir():
            parsed = parse_skill_markdown(skill_path)
            if parsed:
                return {
                    "folder_name": folder_name,
                    "name": parsed["name"],
                    "description": parsed["description"],
                    "score": 1.0,
                    "full_text": parsed["full_text"],
                    "path": parsed["path"]
                }
        return None

    def match_skills(self, user_query: str, min_score: float = MIN_SKILL_SCORE, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query the skill vector database.
        Returns skills with similarity score > min_score (default from UI or 0.5).
        """
        results = skill_vector_store.query_similar(user_query, top_k=5, min_score=min_score, conversation_id=conversation_id, invoker="skill search")
        matched = []
        for r in results:
            meta = r.get("metadata", {})
            matched.append({
                "folder_name": meta.get("folder_name"),
                "name": meta.get("name"),
                "description": meta.get("description"),
                "score": r.get("score"),
                "full_text": r.get("text"),
                "path": meta.get("path")
            })
        return matched

    def get_skill_tool_signature(self, skill_info: Dict[str, Any]) -> str:
        """
        Dynamically determine procedural tool signature available for this skill.
        Minimizes skill-specific code in the orchestrator per SPECIFICATION.md.
        """
        folder = skill_info.get("folder_name", "").lower()
        if "weather" in folder or "time" in folder:
            return "env_tools.get_weather_and_time(city: str)"
        elif "person" in folder or "registry" in folder:
            return "person_search.query_person_registry(keyword: str, field: str = None)"
        elif "stock" in folder:
            return "stock_search.analyze_stock_query(query: str)"
        elif "document" in folder or "retriever" in folder:
            return "document_search_tool.search_documents(query: str, top_k: int)"
        return f"{skill_info.get('folder_name')}.execute(arguments: dict)"

    def is_skill_tool_match(self, skill_info: Dict[str, Any], tool_to_execute: str) -> bool:
        """
        Check if the tool requested by LLM matches this skill.
        """
        folder_name = skill_info.get("folder_name", "").lower()
        skill_name = skill_info.get("name", "").lower()
        tool_lower = (tool_to_execute or "").lower().strip()
        if not tool_lower:
            return False
        skill_kw = folder_name.replace("-skill", "").split("-")
        return (
            tool_lower in folder_name or
            folder_name in tool_lower or
            tool_lower in skill_name or
            skill_name in tool_lower or
            any(kw in tool_lower for kw in skill_kw) or
            ("env_tools" in tool_lower and ("weather" in folder_name or "time" in folder_name)) or
            ("person_search" in tool_lower and "person" in folder_name) or
            ("stock_search" in tool_lower and "stock" in folder_name) or
            tool_lower in ["execute_tool", "true"]
        )

    def execute_skill(
        self,
        skill_info: Dict[str, Any],
        user_query: str,
        conversation_id: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatch execution from skill to procedural tool.
        Logs interactions between 'skill', 'tool', and 'external API call'.
        """
        folder_name = skill_info.get("folder_name", "")
        skill_name = skill_info.get("name", "")
        score = skill_info.get("score", 0.0)

        result_data = None
        evidence_text = ""

        if "weather" in folder_name.lower() or "time" in folder_name.lower():
            try:
                script_path = self.skills_dir / folder_name / "scripts" / "env_tools.py"
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location("env_tools", str(script_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # Extract city from structured arguments or fallback query
                    if arguments and any(k in arguments for k in ["city", "city_name", "location"]):
                        city = str(arguments.get("city") or arguments.get("city_name") or arguments.get("location")).strip()
                    else:
                        stop_words = {"what", "is", "the", "weather", "in", "time", "at", "for", "check", "tell", "me", "how", "right", "now", "currently", "today", "forecast", "like", "please", "conditions"}
                        words = [w.strip("?,.!\"'") for w in user_query.split() if w.lower().strip("?,.!\"'") not in stop_words]
                        city = " ".join(words).strip() or "Tokyo"

                    # Log invocation from skill to tool
                    audit_logger.log_call(
                        event_type="tool",
                        call_type="invocation",
                        invoker="skill",
                        recipient="tool",
                        payload={"tool": "env_tools.py", "function": "get_weather_and_time", "city": city},
                        description=f"Tool message passed to env_tools.py for city: {city}",
                        conversation_id=conversation_id
                    )

                    # Log external API call invocation from tool
                    audit_logger.log_call(
                        event_type="external API call",
                        call_type="invocation",
                        invoker="tool",
                        recipient="external API call",
                        payload={"api": "Open-Meteo", "city": city, "geocoding_url": "https://geocoding-api.open-meteo.com/v1/search"},
                        description=f"Full payload passed to Open-Meteo API for {city}",
                        conversation_id=conversation_id
                    )

                    result_data = mod.get_weather_and_time(city)

                    # Log external API call response
                    audit_logger.log_call(
                        event_type="external API call",
                        call_type="response",
                        invoker="external API call",
                        recipient="tool",
                        payload=result_data,
                        description=f"Full response received from Open-Meteo API for {city}",
                        conversation_id=conversation_id
                    )

                    evidence_text = f"Weather in {result_data.get('city', city)}: {result_data.get('condition')}, Temp: {result_data.get('temperature_celsius')}°C / {result_data.get('temperature_fahrenheit')}°F, Time: {result_data.get('local_time')}"
            except Exception as e:
                result_data = {"error": str(e)}
                evidence_text = f"Skill execution error: {e}"

        elif "person" in folder_name.lower() or "registry" in folder_name.lower():
            try:
                script_path = self.skills_dir / folder_name / "scripts" / "person_search.py"
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location("person_search", str(script_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if arguments and any(k in arguments for k in ["keyword", "name", "query", "field"]):
                        kw = str(arguments.get("keyword") or arguments.get("name") or arguments.get("query") or "").strip()
                        field = arguments.get("field")
                    else:
                        search_words = [w for w in user_query.split() if w.lower() not in ["who", "where", "is", "the", "find", "all", "what", "job", "title", "of", "person", "in", "lives"]]
                        kw = " ".join(search_words).strip()
                        field = None

                    audit_logger.log_call(
                        event_type="tool",
                        call_type="invocation",
                        invoker="skill",
                        recipient="tool",
                        payload={"tool": "person_search.py", "function": "query_person_registry", "keyword": kw, "field": field},
                        description=f"Tool message passed to person_search.py for keyword: '{kw}' (field: {field})",
                        conversation_id=conversation_id
                    )

                    matches = mod.query_person_registry(kw, field=field)
                    result_data = {"matches": matches, "query_keyword": kw, "field": field}
                    if matches:
                        evidence_text = f"Found {len(matches)} personnel records: " + "; ".join([f"{p['name']} ({p.get('job_title', '')}, {p.get('city', '')}, {p.get('country', '')})" for p in matches[:5]])
                    else:
                        evidence_text = f"No matching registry records found for query keyword '{kw}'."
            except Exception as e:
                result_data = {"error": str(e)}
                evidence_text = f"Skill execution error: {e}"

        elif "stock" in folder_name.lower():
            try:
                script_path = self.skills_dir / folder_name / "scripts" / "stock_search.py"
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location("stock_search", str(script_path))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)

                    stock_q = user_query
                    if arguments and any(k in arguments for k in ["query", "mode"]):
                        stock_q = str(arguments.get("query") or arguments.get("mode") or user_query)

                    audit_logger.log_call(
                        event_type="tool",
                        call_type="invocation",
                        invoker="skill",
                        recipient="tool",
                        payload={"tool": "stock_search.py", "function": "analyze_stock_query", "query": stock_q},
                        description=f"Tool message passed to stock_search.py for: '{stock_q}'",
                        conversation_id=conversation_id
                    )

                    if arguments and "mode" in arguments and arguments.get("mode"):
                        result_data = mod.get_stocks_by_performance(mode=str(arguments["mode"]), limit=arguments.get("limit", 5))
                    else:
                        result_data = mod.analyze_stock_query(stock_q)

                    stocks = result_data.get("stocks", [])
                    evidence_text = f"{result_data.get('category')}: " + ", ".join([f"{s['ticker']} ({s['change_pct']:+.2f}%, ${s['price']})" for s in stocks])
            except Exception as e:
                result_data = {"error": str(e)}
                evidence_text = f"Skill execution error: {e}"

        else:
            result_data = {"info": "Skill matched based on SOP triggers."}
            evidence_text = skill_info.get("description", "")

        # Log completion from tool back to skill (Response)
        audit_logger.log_call(
            event_type="tool",
            call_type="response",
            invoker="tool",
            recipient="skill",
            payload=result_data,
            description=f"Tool response received from {skill_name}",
            conversation_id=conversation_id
        )

        return {
            "skill_name": skill_name,
            "folder_name": folder_name,
            "score": score,
            "result_data": result_data,
            "evidence_text": evidence_text
        }

# Global singleton
skill_manager = SkillManager()
