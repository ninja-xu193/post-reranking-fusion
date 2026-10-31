"""Release hygiene checks without real private data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_check", Path(__file__).resolve().parents[1] / "tools" / "check_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_checkout(self):
        files, errors = release.inspect(release.ROOT)
        self.assertFalse(errors, errors)
        self.assertTrue(files)

    def test_rejects_unexpected_and_private_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "predictions.json").write_text("{}", encoding="utf-8")
            (root / "README.md").write_text("/" + "Users/fictional/private", encoding="utf-8")
            _, errors = release.inspect(root)
            self.assertTrue(any("outside the release allowlist" in e for e in errors))
            self.assertTrue(any("machine path" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
