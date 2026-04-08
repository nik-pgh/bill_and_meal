import base64
import pytest
from pathlib import Path

from bill_and_meal.teacher import validate_teacher_output, encode_image, TEACHER_SYSTEM_PROMPT


class TestValidateTeacherOutput:
    def test_valid_output_passes(self, sample_labeled_record):
        assert validate_teacher_output(sample_labeled_record) is True

    def test_rejects_missing_recipe_marker(self, sample_labeled_record):
        record = {**sample_labeled_record, "teacher_output": "Here are some ideas for dinner."}
        assert validate_teacher_output(record) is False

    def test_rejects_low_ingredient_coverage(self, sample_labeled_record):
        record = {
            **sample_labeled_record,
            "ingredients": ["chicken", "garlic", "olive oil", "lemon", "rice",
                            "butter", "cream", "flour", "eggs", "milk"],
            "teacher_output": (
                "RECIPE 1: Toast\n"
                "USES: bread\n"
                "TIME: 5 minutes\n"
                "DIFFICULTY: easy\n"
                "STEPS:\n"
                "1. Toast the bread."
            ),
        }
        assert validate_teacher_output(record) is False

    def test_rejects_missing_steps(self, sample_labeled_record):
        record = {
            **sample_labeled_record,
            "teacher_output": "RECIPE 1: Chicken\nUSES: chicken breast\n",
        }
        assert validate_teacher_output(record) is False

    def test_accepts_30_percent_coverage(self):
        record = {
            "ingredients": ["chicken", "garlic", "olive oil", "lemon", "rice",
                            "onion", "pepper", "tomato", "basil", "salt"],
            "teacher_output": (
                "RECIPE 1: Simple Chicken\n"
                "USES: chicken, garlic, olive oil\n"
                "TIME: 20 minutes\n"
                "DIFFICULTY: easy\n"
                "STEPS:\n"
                "1. Cook the chicken with garlic and olive oil."
            ),
        }
        assert validate_teacher_output(record) is True


class TestEncodeImage:
    def test_returns_base64_string(self, sample_receipt_image):
        result = encode_image(sample_receipt_image)
        decoded = base64.standard_b64decode(result)
        assert len(decoded) > 0

    def test_encoded_data_is_string(self, sample_receipt_image):
        result = encode_image(sample_receipt_image)
        assert isinstance(result, str)


class TestTeacherPrompt:
    def test_prompt_exists_and_has_key_instructions(self):
        assert "recipe" in TEACHER_SYSTEM_PROMPT.lower()
        assert "RECIPE 1:" in TEACHER_SYSTEM_PROMPT
        assert "STEPS:" in TEACHER_SYSTEM_PROMPT
        assert "USES:" in TEACHER_SYSTEM_PROMPT
