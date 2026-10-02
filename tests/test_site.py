import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "docs" / "template.html"


def test_every_element_id_used_by_the_site_script_exists():
    """A missing container made Plotly throw and silently blanked half the page once."""
    html = TEMPLATE.read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("([A-Za-z_0-9]+)"\)', html)) | set(re.findall(r'Plotly\.newPlot\("([A-Za-z_0-9]+)"', html))
    missing = sorted(u for u in used if f'id="{u}"' not in html)
    assert not missing, f"template.html is missing element ids: {missing}"
