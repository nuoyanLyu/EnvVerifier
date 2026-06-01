from agentfly.rewards.math_reward import (
    math_equal_reward_think_no_tool_bonus,
    math_equal_reward_think_tool,
)


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _tool(content: str = "15") -> dict:
    return {"role": "tool", "content": content}


def test_math_equal_reward_think_tool_correct_with_tool() -> None:
    trajectory = [
        _assistant(
            '<think>Need a calculator.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}}</tool_call>'
        ),
        _tool(),
        _assistant("<think>The result is known.</think><answer>15</answer>"),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 1.0
    assert result["acc"] == 1.0


def test_math_equal_reward_think_tool_wrong_with_tool() -> None:
    trajectory = [
        _assistant(
            '<think>Need a calculator.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}}</tool_call>'
        ),
        _tool(),
        _assistant("<think>The result is known.</think><answer>14</answer>"),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.1
    assert result["acc"] == 0.0


def test_math_equal_reward_think_tool_no_answer_with_tool() -> None:
    trajectory = [
        _assistant(
            '<think>Need a calculator.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}}</tool_call>'
        ),
        _tool(),
        _assistant(
            '<think>I should still verify once more.</think><tool_call>{"name": "calculator", "arguments": {"expression": "15+0"}}</tool_call>'
        ),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.1
    assert result["acc"] == 0.0


def test_math_equal_reward_think_tool_correct_without_tool_reward() -> None:
    trajectory = [
        _assistant("<think>I can solve this mentally.</think><answer>15</answer>"),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.0
    assert result["acc"] == 1.0


def test_math_equal_reward_think_no_tool_bonus_correct_answer() -> None:
    trajectory = [
        _assistant("<think>I can solve this mentally.</think><answer>15</answer>"),
    ]

    result = math_equal_reward_think_no_tool_bonus(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 1.0
    assert result["acc"] == 1.0


def test_math_equal_reward_think_no_tool_bonus_wrong_answer() -> None:
    trajectory = [
        _assistant("<think>I can solve this mentally.</think><answer>14</answer>"),
    ]

    result = math_equal_reward_think_no_tool_bonus(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.1
    assert result["acc"] == 0.0


def test_math_equal_reward_think_no_tool_bonus_no_answer() -> None:
    trajectory = [
        _assistant(
            '<think>I should verify once more.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}}</tool_call>'
        ),
        _tool(),
    ]

    result = math_equal_reward_think_no_tool_bonus(
        final_response=trajectory[0]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.1
    assert result["acc"] == 0.0


def test_math_equal_reward_think_rejects_extra_text_outside_tags() -> None:
    trajectory = [
        _assistant(
            'prefix<think>Need a calculator.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}}</tool_call>'
        ),
        _tool(),
        _assistant("<think>The result is known.</think><answer>15</answer>"),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.0
    assert result["acc"] == 0.0


def test_math_equal_reward_think_rejects_malformed_tool_call_json() -> None:
    trajectory = [
        _assistant(
            '<think>Need a calculator.</think><tool_call>{"name": "calculator", "arguments": {"expression": "3*5"}</tool_call>'
        ),
    ]

    result = math_equal_reward_think_tool(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.0
    assert result["acc"] == 0.0


def test_math_equal_reward_think_rejects_missing_tool_call_fields() -> None:
    trajectory = [
        _assistant(
            '<think>Need a calculator.</think><tool_call>{"name": "calculator"}</tool_call>'
        ),
    ]

    result = math_equal_reward_think_no_tool_bonus(
        final_response=trajectory[-1]["content"],
        answer="15",
        trajectory=trajectory,
    )

    assert result["reward"] == 0.0
    assert result["acc"] == 0.0
