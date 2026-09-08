import pytest


@pytest.fixture(autouse=True)
def run_in_a_temporary_directory(tmp_path, monkeypatch):
    """Command-line runs checkpoint to runs/ in the working directory by default; keep that out of the repo."""
    monkeypatch.chdir(tmp_path)
