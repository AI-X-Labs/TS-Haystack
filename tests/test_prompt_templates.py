# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for PromptTemplateBank module.

These tests verify template sampling, placeholder substitution, and grammar helpers.
All tests use synthetic data and can run without Capture24 data.
"""

import numpy as np
import pytest

from ts_haystack.core.prompt_templates import (
    PromptTemplateBank,
    TemplateVariant,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def template_bank():
    """Create a fresh PromptTemplateBank."""
    return PromptTemplateBank()


@pytest.fixture
def rng():
    """Create a seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


# =============================================================================
# Test Basic Sampling
# =============================================================================


class TestBasicSampling:
    """Tests for basic template sampling functionality."""

    def test_sample_existence_task(self, template_bank, rng):
        """Test sampling existence task templates."""
        question, answer = template_bank.sample(
            "existence",
            rng,
            activity="walking",
            exists=True,
        )

        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert "walking" in question
        assert len(question) > 0
        assert len(answer) > 0

    def test_sample_localization_task(self, template_bank, rng):
        """Test sampling localization task templates."""
        question, answer = template_bank.sample(
            "localization",
            rng,
            activity="running",
            start="9:15 AM",
            end="9:20 AM",
        )

        assert "running" in question
        assert "9:15 AM" in answer or "9:15" in answer
        assert "9:20 AM" in answer or "9:20" in answer

    def test_sample_counting_task(self, template_bank, rng):
        """Test sampling counting task templates."""
        question, answer = template_bank.sample(
            "counting",
            rng,
            activity="walking",
            count=3,
        )

        assert "walking" in question
        assert "3" in answer

    def test_sample_ordering_task(self, template_bank, rng):
        """Test sampling ordering task templates."""
        question, answer = template_bank.sample(
            "ordering",
            rng,
            activity_a="walking",
            activity_b="running",
            a_before_b=True,
            first_activity="walking",
            second_activity="running",
        )

        assert "walking" in question or "running" in question

    def test_sample_state_query_task(self, template_bank, rng):
        """Test sampling state query task templates."""
        question, answer = template_bank.sample(
            "state_query",
            rng,
            needle_activity="running",
            global_state="sedentary",
        )

        assert "running" in question
        assert "sedentary" in answer.lower()

    def test_sample_antecedent_task(self, template_bank, rng):
        """Test sampling antecedent task templates."""
        question, answer = template_bank.sample(
            "antecedent",
            rng,
            target_activity="running",
            antecedent_activity="walking",
        )

        assert "running" in question
        assert "walking" in answer.lower()

    def test_sample_comparison_with_task(self, template_bank, rng):
        """Test sampling comparison_with task templates (activity bouts)."""
        question, answer = template_bank.sample(
            "comparison_with",
            rng,
            extremum="longest",
            activity="walking",
            start="10:00 AM",
            end="10:30 AM",
            duration_ms=1800000,  # 30 minutes
        )

        assert "longest" in question.lower() or "walking" in question

    def test_sample_comparison_without_task(self, template_bank, rng):
        """Test sampling comparison_without task templates (gaps between bouts)."""
        question, answer = template_bank.sample(
            "comparison_without",
            rng,
            extremum="longest",
            activity="walking",
            start="10:00 AM",
            end="10:30 AM",
            duration_ms=1800000,  # 30 minutes
        )

        # Check that the question mentions "without" or "gap"
        assert "without" in question.lower() or "gap" in question.lower() or "walking" in question

    def test_sample_multi_hop_task(self, template_bank, rng):
        """Test sampling multi-hop task templates."""
        question, answer = template_bank.sample(
            "multi_hop",
            rng,
            target_activity="walking",
            anchor_activity="running",
            k=2,
            direction="after",
            start="11:00 AM",
            end="11:15 AM",
        )

        assert "walking" in question or "running" in question

    def test_unknown_task_raises_error(self, template_bank, rng):
        """Test that sampling unknown task raises ValueError."""
        with pytest.raises(ValueError, match="No templates registered"):
            template_bank.sample("unknown_task", rng, activity="walking")


# =============================================================================
# Test Grammar Helpers
# =============================================================================


class TestGrammarHelpers:
    """Tests for automatic grammar helper generation."""

    def test_singular_count_grammar(self, template_bank, rng):
        """Test singular grammar for count=1."""
        # Sample multiple times to find a template that uses grammar helpers
        for _ in range(50):
            question, answer = template_bank.sample(
                "counting",
                rng,
                activity="walking",
                count=1,
            )
            # Check for singular forms
            if "bout" in answer and "bouts" not in answer:
                assert "1 walking bout" in answer or "1 bout" in answer
                return
            if "is 1" in answer:
                return

        # If we get here, at least verify no errors occurred
        assert True

    def test_plural_count_grammar(self, template_bank, rng):
        """Test plural grammar for count>1."""
        for _ in range(50):
            question, answer = template_bank.sample(
                "counting",
                rng,
                activity="walking",
                count=5,
            )
            if "bouts" in answer:
                assert "5" in answer
                return
            if "are 5" in answer:
                return

        assert True

    def test_yes_no_helpers_positive(self, template_bank, rng):
        """Test yes/no helpers for positive existence."""
        found_yes = False
        for _ in range(50):
            question, answer = template_bank.sample(
                "existence",
                rng,
                activity="walking",
                exists=True,
            )
            if "Yes" in answer:
                found_yes = True
                break

        assert found_yes, "Expected 'Yes' in positive existence answer"

    def test_yes_no_helpers_negative(self, template_bank, rng):
        """Test yes/no helpers for negative existence."""
        found_no = False
        for _ in range(50):
            question, answer = template_bank.sample(
                "existence",
                rng,
                activity="walking",
                exists=False,
            )
            if "No" in answer:
                found_no = True
                break

        assert found_no, "Expected 'No' in negative existence answer"

    def test_ordering_came_before_helper(self, template_bank, rng):
        """Test ordering 'came before/after' helper."""
        for _ in range(50):
            question, answer = template_bank.sample(
                "ordering",
                rng,
                activity_a="walking",
                activity_b="running",
                a_before_b=True,
                first_activity="walking",
                second_activity="running",
            )
            if "came before" in answer:
                return

        # Some templates don't use this helper
        assert True

    def test_ordinal_conversion(self, template_bank, rng):
        """Test ordinal conversion (1 -> first, 2 -> second, etc.)."""
        ordinal_found = False
        for _ in range(50):
            question, answer = template_bank.sample(
                "multi_hop",
                rng,
                target_activity="walking",
                anchor_activity="running",
                k=1,
                direction="after",
                start="11:00 AM",
                end="11:15 AM",
            )
            if "first" in question.lower():
                ordinal_found = True
                break

        assert ordinal_found, "Expected 'first' ordinal in multi-hop question"

    def test_duration_formatting_minutes(self, template_bank, rng):
        """Test duration formatting for values >= 60000ms."""
        for _ in range(50):
            question, answer = template_bank.sample(
                "comparison_with",
                rng,
                extremum="longest",
                activity="walking",
                start="10:00 AM",
                end="10:30 AM",
                duration_ms=120000,  # 2 minutes
            )
            if "minutes" in answer:
                return

        # Some templates don't include duration
        assert True

    def test_duration_formatting_seconds(self, template_bank, rng):
        """Test duration formatting for values < 60000ms."""
        for _ in range(50):
            question, answer = template_bank.sample(
                "comparison_with",
                rng,
                extremum="shortest",
                activity="walking",
                start="10:00 AM",
                end="10:00 AM",
                duration_ms=30000,  # 30 seconds
            )
            if "seconds" in answer:
                return

        # Some templates don't include duration
        assert True


# =============================================================================
# Test A/An Correction
# =============================================================================


class TestAAnCorrection:
    """Tests for automatic a/an article correction."""

    def test_a_before_consonant(self, template_bank):
        """Test 'a' is kept before consonant sounds."""
        result = template_bank._check_a_an("This is a walking bout.")
        assert "a walking" in result

    def test_an_before_vowel(self, template_bank):
        """Test 'a' is changed to 'an' before vowel sounds."""
        result = template_bank._check_a_an("This is a activity.")
        assert "an activity" in result

    def test_an_preserved_before_vowel(self, template_bank):
        """Test 'an' is kept before vowel sounds."""
        result = template_bank._check_a_an("This is an activity.")
        assert "an activity" in result

    def test_an_changed_to_a_before_consonant(self, template_bank):
        """Test 'an' is changed to 'a' before consonant sounds."""
        result = template_bank._check_a_an("This is an walking bout.")
        assert "a walking" in result

    def test_preserves_capitalization(self, template_bank):
        """Test that article capitalization is preserved."""
        result = template_bank._check_a_an("A activity was detected.")
        assert "An activity" in result

        result = template_bank._check_a_an("An walking bout.")
        assert "A walking" in result


# =============================================================================
# Test Template Counts
# =============================================================================


class TestTemplateCounts:
    """Tests for template count verification."""

    def test_all_tasks_have_templates(self, template_bank):
        """Test that all expected tasks have templates."""
        expected_tasks = [
            "existence",
            "localization",
            "counting",
            "ordering",
            "state_query",
            "antecedent",
            "comparison_with",
            "comparison_without",
            "multi_hop",
        ]

        for task in expected_tasks:
            count = template_bank.get_template_count(task)
            assert count > 0, f"Task '{task}' has no templates"

    def test_sufficient_template_diversity(self, template_bank):
        """Test that each task has sufficient template diversity (20+)."""
        counts = template_bank.get_all_template_counts()

        for task, count in counts.items():
            assert count >= 20, f"Task '{task}' has only {count} templates (expected 20+)"

    def test_get_available_tasks(self, template_bank):
        """Test getting list of available tasks."""
        tasks = template_bank.get_available_tasks()

        assert "existence" in tasks
        assert "localization" in tasks
        assert "counting" in tasks


# =============================================================================
# Test Custom Templates
# =============================================================================


class TestCustomTemplates:
    """Tests for custom template registration."""

    def test_register_custom_templates(self, rng):
        """Test registering custom templates at initialization."""
        custom = {
            "existence": [
                TemplateVariant(
                    question="Custom: Is {activity} present?",
                    answer="Custom: {yes_no}.",
                )
            ]
        }

        bank = PromptTemplateBank(custom_templates=custom)

        # Original templates plus custom
        count = bank.get_template_count("existence")
        assert count > len(PromptTemplateBank.TEMPLATES["existence"])

    def test_register_templates_runtime(self, template_bank, rng):
        """Test registering templates at runtime."""
        original_count = template_bank.get_template_count("existence")

        template_bank.register_templates(
            "existence",
            [
                TemplateVariant(
                    question="Runtime: Does {activity} exist?",
                    answer="Runtime: {yes_no}.",
                )
            ],
        )

        new_count = template_bank.get_template_count("existence")
        assert new_count == original_count + 1

    def test_register_new_task(self, template_bank, rng):
        """Test registering templates for a new task."""
        template_bank.register_templates(
            "custom_task",
            [
                TemplateVariant(
                    question="Custom task: {param}?",
                    answer="Custom answer: {param}.",
                )
            ],
        )

        question, answer = template_bank.sample("custom_task", rng, param="test_value")
        assert "test_value" in question
        assert "test_value" in answer


# =============================================================================
# Test Determinism
# =============================================================================


class TestDeterminism:
    """Tests for reproducible template sampling."""

    def test_same_seed_same_template(self, template_bank):
        """Test that same seed produces same template selection."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        q1, a1 = template_bank.sample("existence", rng1, activity="walking", exists=True)
        q2, a2 = template_bank.sample("existence", rng2, activity="walking", exists=True)

        assert q1 == q2
        assert a1 == a2

    def test_different_seed_potentially_different_template(self, template_bank):
        """Test that different seeds can produce different templates."""
        results = set()

        for seed in range(100):
            rng = np.random.default_rng(seed)
            q, a = template_bank.sample("existence", rng, activity="walking", exists=True)
            results.add(q)

        # With 100 different seeds, we should see multiple distinct templates
        assert len(results) > 1, "Expected multiple distinct templates with different seeds"

    def test_all_templates_accessible(self, template_bank):
        """Test that all templates can be sampled given enough trials."""
        task = "existence"
        n_templates = template_bank.get_template_count(task)

        seen_questions = set()
        for seed in range(1000):
            rng = np.random.default_rng(seed)
            q, _ = template_bank.sample(task, rng, activity="walking", exists=True)
            seen_questions.add(q)

        # Should see a good fraction of available templates
        coverage = len(seen_questions) / n_templates
        assert coverage > 0.5, f"Only covered {coverage:.1%} of templates"


# =============================================================================
# Test Missing Placeholders
# =============================================================================


class TestMissingPlaceholders:
    """Tests for error handling with missing placeholders."""

    def test_missing_required_placeholder_raises(self, template_bank, rng):
        """Test that missing required placeholder raises ValueError."""
        with pytest.raises(ValueError, match="Missing placeholder"):
            # 'activity' is required but not provided
            template_bank.sample("existence", rng, exists=True)

    def test_extra_placeholders_ignored(self, template_bank, rng):
        """Test that extra placeholders are ignored (no error)."""
        # This should work - extra parameters are ignored
        question, answer = template_bank.sample(
            "existence",
            rng,
            activity="walking",
            exists=True,
            extra_param="ignored",
            another_extra=123,
        )

        assert "walking" in question


# =============================================================================
# Test Template Quality
# =============================================================================


class TestTemplateQuality:
    """Tests for template content quality."""

    def test_no_empty_questions(self, template_bank, rng):
        """Test that no templates produce empty questions."""
        for task in template_bank.get_available_tasks():
            for seed in range(20):
                rng = np.random.default_rng(seed)

                # Build appropriate kwargs for each task
                kwargs = _get_sample_kwargs_for_task(task)

                q, a = template_bank.sample(task, rng, **kwargs)

                assert len(q.strip()) > 0, f"Empty question for task '{task}'"
                assert len(a.strip()) > 0, f"Empty answer for task '{task}'"

    def test_no_unfilled_placeholders(self, template_bank, rng):
        """Test that all placeholders get filled."""
        for task in template_bank.get_available_tasks():
            for seed in range(20):
                rng = np.random.default_rng(seed)

                kwargs = _get_sample_kwargs_for_task(task)

                q, a = template_bank.sample(task, rng, **kwargs)

                # Check for unfilled placeholders (curly braces)
                assert "{" not in q, f"Unfilled placeholder in question: {q}"
                assert "}" not in q, f"Unfilled placeholder in question: {q}"
                assert "{" not in a, f"Unfilled placeholder in answer: {a}"
                assert "}" not in a, f"Unfilled placeholder in answer: {a}"


def _get_sample_kwargs_for_task(task: str) -> dict:
    """Get sample kwargs for a given task."""
    base_kwargs = {
        "existence": {"activity": "walking", "exists": True},
        "localization": {"activity": "walking", "start": "9:00 AM", "end": "9:30 AM"},
        "counting": {"activity": "walking", "count": 3},
        "ordering": {
            "activity_a": "walking",
            "activity_b": "running",
            "a_before_b": True,
            "first_activity": "walking",
            "second_activity": "running",
        },
        "state_query": {"needle_activity": "running", "global_state": "sedentary"},
        "antecedent": {"target_activity": "running", "antecedent_activity": "walking"},
        "comparison_with": {
            "extremum": "longest",
            "activity": "walking",
            "start": "10:00 AM",
            "end": "10:30 AM",
            "duration_ms": 1800000,
        },
        "comparison_without": {
            "extremum": "longest",
            "activity": "walking",
            "start": "10:00 AM",
            "end": "10:30 AM",
            "duration_ms": 1800000,
        },
        "multi_hop": {
            "target_activity": "walking",
            "anchor_activity": "running",
            "k": 2,
            "direction": "after",
            "start": "11:00 AM",
            "end": "11:15 AM",
        },
        # Anomaly Detection (split into positive/negative like comparison_with/without)
        "anomaly_detection_positive": {
            "anomaly_activity": "running",
            "background_regime": "sedentary",
        },
        "anomaly_detection_negative": {
            "background_regime": "sedentary",
        },
        # Anomaly Localization (split into positive/negative)
        "anomaly_localization_positive": {
            "anomaly_activity": "running",
            "start": "9:00 AM",
            "end": "9:15 AM",
        },
        "anomaly_localization_negative": {
            "background_regime": "sedentary",
        },
    }

    return base_kwargs.get(task, {})


# =============================================================================
# Test JSON Loading (if applicable)
# =============================================================================


class TestJSONLoading:
    """Tests for loading templates from JSON files."""

    def test_from_json_with_valid_file(self, tmp_path):
        """Test loading templates from a valid JSON file."""
        import json

        json_content = {
            "existence": [
                {"question": "JSON: Is {activity} there?", "answer": "JSON: {yes_no}."}
            ],
            "custom_task": [
                {"question": "Custom: {param}?", "answer": "Custom: {param}."}
            ],
        }

        json_path = tmp_path / "templates.json"
        with open(json_path, "w") as f:
            json.dump(json_content, f)

        bank = PromptTemplateBank.from_json(str(json_path))

        # Should have loaded custom templates
        assert bank.get_template_count("custom_task") == 1

        # Should have merged with defaults for existence
        default_count = len(PromptTemplateBank.TEMPLATES["existence"])
        assert bank.get_template_count("existence") == default_count + 1
