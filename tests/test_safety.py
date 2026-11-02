import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("TELEGRAM_CHAT_ID", "1")
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("YANDEX_TOKEN", "test")
os.environ.setdefault("VK_REDIRECT_URI", "https://example.test/vk/callback")
os.environ.setdefault("VK_GROUP_ID", "1")

import main


class SafetyTests(unittest.TestCase):
    def test_pkce_is_s256_and_url_safe(self):
        verifier, challenge = main.pkce_pair()
        expected = main.base64.urlsafe_b64encode(main.hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
        self.assertEqual(challenge, expected)
        self.assertNotIn("=", challenge)

    def test_prompt_has_separate_negative_line(self):
        prompt = main.build_standard_prompt({"composition": {}, "camera": {}})
        self.assertIn("Негативный промпт:", prompt)
        self.assertIn("Промпт для генерации", prompt)

    def test_state_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            main.save_json_atomic(path, {"safe": True})
            self.assertEqual(main.load_json(path, {}), {"safe": True})
            self.assertFalse((Path(directory) / "state.json.tmp").exists())

    def test_default_env_disables_scheduler(self):
        old = os.environ.pop("SCHEDULER_ENABLED", None)
        try:
            self.assertFalse(main.env_bool("SCHEDULER_ENABLED", False))
        finally:
            if old is not None:
                os.environ["SCHEDULER_ENABLED"] = old


if __name__ == "__main__":
    unittest.main()

