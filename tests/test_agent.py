import pytest

from ragagent.agent import Agent, FinalAnswer, ToolCall
from ragagent.tools import calculate_pto


@pytest.mark.parametrize(
    ("years", "expected"), [(0, 15), (1.9, 15), (2, 20), (4.5, 20), (5, 25), (12, 25)]
)
def test_calculate_pto_tiers(years: float, expected: int) -> None:
    assert calculate_pto(years)["annual_days"] == expected


def test_calculate_pto_prorates_rounds_down_and_caps_carryover() -> None:
    r = calculate_pto(3, months_worked_this_year=7, days_used=2)
    assert r["accrued_days"] == 11.5  # 20 * 7/12 = 11.67 -> 11.5
    assert r["remaining_days"] == 9.5
    assert r["max_carryover_days"] == 5


def test_calculate_pto_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        calculate_pto(-1)
    with pytest.raises(ValueError):
        calculate_pto(1, months_worked_this_year=13)


def test_agent_routes_pto_math_to_calculator(agent: Agent) -> None:
    result = agent.run("How many PTO days do I have left with 6 years of service if I used 4 days?")
    assert [s.call.name for s in result.steps] == ["calculate_pto"]
    assert result.steps[0].call.arguments == {"years_of_service": 6.0, "days_used": 4.0}
    assert "25 PTO days per year" in result.answer
    assert "21 remain" in result.answer


def test_agent_routes_policy_question_to_search(agent: Agent) -> None:
    result = agent.run("How long is parental leave for the primary caregiver?")
    assert [s.call.name for s in result.steps] == ["search_docs"]
    assert "16 weeks" in result.answer
    assert result.citations and result.citations[0].doc_id == "parental-leave"


class _BadPlanner:
    def next_action(self, question, steps, tools):  # type: ignore[no-untyped-def]
        if not steps:
            return ToolCall("calculate_pto", {"years_of_service": -3})
        return FinalAnswer(steps[-1].result.get("error", ""))


def test_agent_turns_tool_errors_into_results(agent: Agent) -> None:
    bad = Agent(agent.pipeline, _BadPlanner())
    result = bad.run("anything")
    assert "years_of_service must be >= 0" in result.answer


class _LoopingPlanner:
    def next_action(self, question, steps, tools):  # type: ignore[no-untyped-def]
        return ToolCall("search_docs", {"query": question})


def test_agent_stops_at_step_limit(agent: Agent) -> None:
    result = Agent(agent.pipeline, _LoopingPlanner(), max_steps=2).run("holidays")
    assert len(result.steps) == 2
    assert result.answer.startswith("Stopped")
