"""Tests for LLM prompt templates."""

from __future__ import annotations

from comfio.llm.prompts import (
    DIAGNOSTIC_PROMPT_TEMPLATE,
    EDGE_SYSTEM_PROMPT,
    format_prompt,
)


class TestEdgeSystemPrompt:
    def test_is_string(self) -> None:
        assert isinstance(EDGE_SYSTEM_PROMPT, str)
        assert len(EDGE_SYSTEM_PROMPT) > 0

    def test_contains_rules(self) -> None:
        assert "CRITICAL RULES" in EDGE_SYSTEM_PROMPT
        assert "Never invent physical metrics" in EDGE_SYSTEM_PROMPT

    def test_has_placeholders(self) -> None:
        assert "{context}" in EDGE_SYSTEM_PROMPT
        assert "{query}" in EDGE_SYSTEM_PROMPT


class TestDiagnosticPromptTemplate:
    def test_is_string(self) -> None:
        assert isinstance(DIAGNOSTIC_PROMPT_TEMPLATE, str)

    def test_has_placeholders(self) -> None:
        assert "{ieq_report}" in DIAGNOSTIC_PROMPT_TEMPLATE
        assert "{complaint}" in DIAGNOSTIC_PROMPT_TEMPLATE


class TestFormatPrompt:
    def test_basic_format(self) -> None:
        result = format_prompt("Hello {name}!", name="World")
        assert result == "Hello World!"

    def test_multiple_kwargs(self) -> None:
        result = format_prompt("{a} and {b}", a="foo", b="bar")
        assert result == "foo and bar"

    def test_edge_prompt_format(self) -> None:
        result = format_prompt(
            EDGE_SYSTEM_PROMPT,
            context="IEQ score: 75/100",
            query="Why is it hot?",
        )
        assert "IEQ score: 75/100" in result
        assert "Why is it hot?" in result
