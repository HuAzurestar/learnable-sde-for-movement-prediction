"""The CI test extra must include existing resource and rendering imports."""
import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_existing_pirc17_imports_are_declared_in_the_test_extra():
    source = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    match = re.search(r'(?ms)^test\s*=\s*(\[[^]]+\])', source)
    assert match is not None
    requirements = ast.literal_eval(match.group(1))
    names = {re.split(r'[<>=!~\[]', requirement, maxsplit=1)[0].lower()
             for requirement in requirements}
    assert {'pytest', 'psutil', 'threadpoolctl', 'pymupdf', 'matplotlib', 'pillow'} <= names


def test_ci_installs_the_declared_test_extra_without_dropping_tests():
    workflow = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
    assert 'pip install -e ".[test]"' in workflow
    assert 'python -m pytest -q' in workflow
