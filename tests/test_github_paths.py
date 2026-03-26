import unittest

from github_paths import build_github_path, normalize_github_repo_path


class GitHubPathTests(unittest.TestCase):
    def test_build_github_path_accepts_nested_relative_paths(self) -> None:
        self.assertEqual(
            build_github_path("metanova-labs", "nova", "main", "data/results"),
            "metanova-labs/nova/main/data/results",
        )

    def test_build_github_path_rejects_invalid_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain path separators"):
            build_github_path("metanova/labs", "nova", "main", "")

    def test_normalize_github_repo_path_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain empty, '.', '..'"):
            normalize_github_repo_path("../results")

    def test_normalize_github_repo_path_rejects_leading_slash(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be relative"):
            normalize_github_repo_path("/data/results")


if __name__ == "__main__":
    unittest.main()
