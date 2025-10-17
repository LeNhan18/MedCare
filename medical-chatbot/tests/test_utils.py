import unittest
from src.utils.text_processing import normalize_text, tokenize_text
from src.utils.medical_utils import get_medication_suggestions

class TestUtils(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")
        self.assertEqual(normalize_text("   Multiple   Spaces   "), "multiple spaces")
        self.assertEqual(normalize_text("1234"), "1234")

    def test_tokenize_text(self):
        self.assertEqual(tokenize_text("Hello world"), ["Hello", "world"])
        self.assertEqual(tokenize_text("This is a test."), ["This", "is", "a", "test"])
        self.assertEqual(tokenize_text(""), [])

    def test_get_medication_suggestions(self):
        self.assertIn("Aspirin", get_medication_suggestions("headache"))
        self.assertIn("Ibuprofen", get_medication_suggestions("pain"))
        self.assertEqual(get_medication_suggestions("unknown symptom"), [])

if __name__ == '__main__':
    unittest.main()