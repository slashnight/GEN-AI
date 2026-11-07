import unittest
from unittest.mock import Mock

from google.genai.errors import ServerError

from llm_client import LLMClient
from main import generate_answer

# Fake provider for testing
class FakeProvider:

    def generate(self, prompt):
        return "AI stands for Artificial Intelligence."

class TestLLMClient(unittest.TestCase):

    def test_generate(self):

        provider = FakeProvider()
        client = LLMClient(provider)

        result = client.generate("Explain AI")

        self.assertEqual(
            result,
            "AI stands for Artificial Intelligence."
        )

    def test_generate_answer_reports_gemini_unavailable(self):
        client = Mock()
        client.generate.side_effect = ServerError(
            503, {"error": {"message": "temporarily unavailable"}}
        )

        with self.assertRaisesRegex(SystemExit, "Gemini is temporarily unavailable"):
            generate_answer(client, "What is 20 * 2?")

    def test_generate_answer_reraises_other_server_errors(self):
        error = ServerError(500, {"error": {"message": "server error"}})
        client = Mock()
        client.generate.side_effect = error

        with self.assertRaises(ServerError) as raised:
            generate_answer(client, "What is 20 * 2?")

        self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
