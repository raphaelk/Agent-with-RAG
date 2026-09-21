"""
Tests for Skills and SOP Execution.
"""
import importlib.util
import pytest
from pathlib import Path
from services.skill_manager import parse_skill_markdown, skill_manager
import config

def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_parse_skill_markdown():
    skill_dir = config.SKILLS_DIR / "time-weather-skill"
    parsed = parse_skill_markdown(skill_dir)
    assert parsed["name"] == "Time and Weather Skill"
    assert "Open-Meteo" in parsed["description"]
    assert len(parsed["trigger_queries"]) > 0

def test_weather_and_time_tool():
    env_tools_path = config.SKILLS_DIR / "time-weather-skill" / "scripts" / "env_tools.py"
    env_tools = load_module_from_path("env_tools", env_tools_path)
    result = env_tools.get_weather_and_time("London")
    assert "status" in result
    assert result["status"] in ["success", "partial_success"]
    assert "city" in result
    assert "local_time" in result

def test_person_registry_tool():
    person_search_path = config.SKILLS_DIR / "person-information-skill" / "scripts" / "person_search.py"
    person_search = load_module_from_path("person_search", person_search_path)
    matches = person_search.query_person_registry("Tokyo")
    assert len(matches) >= 1
    assert any("Kenji Sato" in m["name"] for m in matches)

    matches_name = person_search.query_person_registry("Alex Morgan")
    assert len(matches_name) >= 1
    assert matches_name[0]["job_title"] == "Principal AI Engineer"

def test_stock_search_tool():
    stock_search_path = config.SKILLS_DIR / "stock-market-skill" / "scripts" / "stock_search.py"
    stock_search = load_module_from_path("stock_search", stock_search_path)

    gainers = stock_search.analyze_stock_query("top gaining stocks")
    assert gainers["status"] == "success"
    assert "Gainers" in gainers["category"]
    assert len(gainers["stocks"]) > 0
    assert gainers["stocks"][0]["change_pct"] > 0

    losers = stock_search.analyze_stock_query("worst drop and losers")
    assert losers["status"] == "success"
    assert "Losers" in losers["category"] or "Decliners" in losers["category"]
    assert len(losers["stocks"]) > 0
    assert losers["stocks"][0]["change_pct"] < 0

def test_parse_tool_call_json():
    from services.agent_orchestrator import _parse_tool_call_json
    raw_json = '''{
      "tool": "person_search.query_person_registry",
      "arguments": {
        "keyword": "Lucas Dubois",
        "field": "name"
      }
    }'''
    res = _parse_tool_call_json(raw_json)
    assert res["tool"] == "person_search.query_person_registry"
    assert res["arguments"]["keyword"] == "Lucas Dubois"
    assert res["arguments"]["field"] == "name"

    # Markdown codeblock wrapping
    codeblock_json = "```json\n" + raw_json + "\n```"
    res_cb = _parse_tool_call_json(codeblock_json)
    assert res_cb["tool"] == "person_search.query_person_registry"

def test_person_registry_tool_with_arguments():
    parsed_skill = {"folder_name": "person-information-skill", "name": "Person Information Skill", "score": 0.9}
    res = skill_manager.execute_skill(
        skill_info=parsed_skill,
        user_query="Tell me about Lucas",
        arguments={"keyword": "Lucas Dubois", "field": "name"}
    )
    assert res["result_data"]["query_keyword"] == "Lucas Dubois"
    assert res["result_data"]["field"] == "name"
    assert len(res["result_data"]["matches"]) >= 1
    assert "Lucas Dubois" in res["evidence_text"]

