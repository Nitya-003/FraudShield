import re
from pathlib import Path

docs = [Path("README.md"), Path("docs/UI_NOTES.md")]
for doc in docs:
    if doc.exists():
        text = doc.read_text(encoding="utf-8")
        matches = re.findall(r"reports/figures/screens/([a-zA-Z0-9_\-\.]+)", text)
        print(f"=== {doc.name} ===")
        for m in sorted(set(matches)):
            print(f"  {m}")
