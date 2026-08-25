import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulate import extract_action


def test_json_wrapped_in_code_fence_and_explanation():
    content = (
        "Based on the provided information, here is a JSON representation of my "
        "decisions regarding willingness to work and consumption:\n\n"
        '```json\n{\n  "work": 0.85,\n  "consumption": 0.60\n}\n```\n\n'
        "### Explanation:\n- **Work (0.85)**: My willingness to work is high at 0.85 "
        "because I have been unemployed and the opportunity to earn a monthly salary "
        "of $2200.90 is significant."
    )
    assert extract_action(content) == [0.85, 0.60]


def test_curly_quote_in_surrounding_prose():
    content = (
        "Based on the provided information, here’s a JSON representation of my "
        "decisions regarding willingness to work and consumption:\n\n"
        '```json\n{\n  "work": 0.85,\n  "consumption": 0.70\n}\n```\n\n'
        "### Explanation:\n- **Work (0.85)**: My willingness to work is high at 0.85 "
        "because I have just been offered a job as a Mail Carrier with a decent "
        "salary of $7004.40."
    )
    assert extract_action(content) == [0.85, 0.70]


def test_prose_before_json_without_code_fence():
    content = (
        "Based on the information provided, here is a JSON format reflecting your "
        "willingness to work and your planned consumption expenditures:\n\n"
        '```json\n{\n  "work": 0.9,\n  "consumption": 0.8\n}\n```\n\n'
        "### Explanation:\n- **Work (0.9)**: Given that you have a job offer with a "
        "substantial salary of $31,458.73, your willingness to work is high."
    )
    assert extract_action(content) == [0.9, 0.8]


def test_plain_json_without_wrapping_text():
    content = '{"work": 1, "consumption": 0.5}'
    assert extract_action(content) == [1, 0.5]


def test_no_json_object_raises_value_error():
    with pytest.raises(ValueError):
        extract_action("I would rather not share a decision this month.")


def test_truncated_response_before_json_raises_value_error():
    # Respuesta cortada por max_tokens antes de llegar al bloque JSON prometido.
    content = (
        "Based on the information provided, here’s how I would assess my "
        "willingness to work and plan my expenditures on essential goods:\n\n"
        "1. **Willingness to Work**: Given that I have been unemployed and now "
        "have a job offer with a substantial salary of $72,188.82 per month, my "
        "willingness to work would be high. This is a significant income that "
        "would allow me to regain financial stability after a period of "
        "unemployment. Therefore, I would rate my willingness to work at 1."
    )
    with pytest.raises(ValueError):
        extract_action(content)


def test_extra_keys_are_ignored():
    content = '{"work": 0.7, "consumption": 0.4, "reasoning": "confident"}'
    assert extract_action(content) == [0.7, 0.4]
