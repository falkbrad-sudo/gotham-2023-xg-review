"""Run the Streamlit app headlessly against the cached pipeline outputs.

Marked integration because it needs data/processed/ from `python -m src.pipeline`;
skipped if those caches are missing.
"""
import pytest

from src.config import resolve_path

pytestmark = pytest.mark.integration


def test_app_renders_every_tab_without_errors():
    if not resolve_path("data/processed/scored_shots.parquet").exists():
        pytest.skip("Run `python -m src.pipeline` first.")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(resolve_path("app/streamlit_app.py")), default_timeout=180).run()
    assert not at.exception
    assert not at.error
    assert [t.label for t in at.tabs] == ["Season", "Finishing", "Shots & model", "Team shape"]
    # Every result sentence is computed; none should carry an unformatted placeholder.
    assert all("nan" not in m.value for m in at.markdown)
