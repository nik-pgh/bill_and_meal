import pytest

from bill_and_meal.evaluate import ingredient_coverage, hallucination_rate


class TestIngredientCoverage:
    def test_full_coverage(self):
        items = ["chicken", "garlic", "rice"]
        text = "Cook the chicken with garlic and serve over rice."
        assert ingredient_coverage(items, text) == 1.0

    def test_partial_coverage(self):
        items = ["chicken", "garlic", "rice", "lemon"]
        text = "Cook the chicken with garlic."
        assert ingredient_coverage(items, text) == 0.5

    def test_no_coverage(self):
        items = ["chicken", "garlic"]
        text = "Boil some pasta with tomato sauce."
        assert ingredient_coverage(items, text) == 0.0

    def test_case_insensitive(self):
        items = ["Chicken Breast", "GARLIC"]
        text = "cook the chicken breast with garlic"
        assert ingredient_coverage(items, text) == 1.0


class TestHallucinationRate:
    def test_no_hallucination(self):
        receipt_items = ["chicken", "garlic", "rice"]
        text = "Cook chicken with garlic over rice."
        known = ["chicken", "garlic", "rice", "pasta", "beef"]
        assert hallucination_rate(receipt_items, text, known) == 0.0

    def test_some_hallucination(self):
        receipt_items = ["chicken", "garlic"]
        text = "Cook chicken with garlic and pasta."
        known = ["chicken", "garlic", "pasta", "rice"]
        # mentioned: chicken, garlic, pasta (3 total). hallucinated: pasta (1)
        assert hallucination_rate(receipt_items, text, known) == pytest.approx(1 / 3)

    def test_all_hallucinated(self):
        receipt_items = ["chicken"]
        text = "Make pasta with beef and cream."
        known = ["chicken", "pasta", "beef", "cream"]
        # mentioned: pasta, beef, cream (3). hallucinated: all 3
        assert hallucination_rate(receipt_items, text, known) == 1.0

    def test_empty_mentions_returns_zero(self):
        receipt_items = ["chicken"]
        text = "A delightful meal."
        known = ["chicken", "pasta"]
        assert hallucination_rate(receipt_items, text, known) == 0.0
