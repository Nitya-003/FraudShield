import json
import subprocess
from pathlib import Path

OUT_DIR = Path(r"C:\Users\ARNAV ARYA\.gemini\antigravity\brain\9d48f82a-9772-420e-bc78-fb7ea758cd3d\lighthouse")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("landing", "http://localhost:8000/"),
    ("overview", "http://localhost:8000/overview"),
    ("livecheck", "http://localhost:8000/live-check"),
]

def run_lh():
    results = {}
    for name, url in TARGETS:
        for mode in ["desktop", "mobile"]:
            key = f"{name}_{mode}"
            json_out = OUT_DIR / f"{key}.json"
            preset_flag = "--preset=desktop" if mode == "desktop" else "--form-factor=mobile --throttling.rttMs=40 --throttling.throughputKbps=10240 --throttling.cpuSlowdownMultiplier=1"
            cmd = f'cmd /c "cd frontend-web && npx lighthouse {url} --output=json --output-path=\"{json_out}\" --chrome-flags=\"--headless\" {preset_flag} --only-categories=performance,accessibility"'
            print(f"Running Lighthouse on {key}...")
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if json_out.exists():
                try:
                    with open(json_out, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    perf = int(round(data["categories"]["performance"]["score"] * 100))
                    a11y = int(round(data["categories"]["accessibility"]["score"] * 100))
                    cls = data["audits"]["cumulative-layout-shift"]["numericValue"]
                    tbt = data["audits"]["total-blocking-time"]["numericValue"]
                    results[key] = {
                        "performance": perf,
                        "accessibility": a11y,
                        "cls": round(cls, 3),
                        "tbt_ms": round(tbt, 1),
                    }
                    print(f"[{key}] Perf: {perf}, A11y: {a11y}, CLS: {round(cls, 3)}, TBT: {round(tbt, 1)}ms")
                except Exception as e:
                    print(f"Error reading {json_out}: {e}")
            else:
                print(f"Failed to generate {json_out}: {res.stderr}")

    summary_file = OUT_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Completed Lighthouse audits.")

if __name__ == "__main__":
    run_lh()
