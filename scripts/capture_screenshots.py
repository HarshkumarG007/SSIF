"""
capture_screenshots.py — High-Fidelity UI Screenshot Capturer via Chrome DevTools Protocol (CDP)
Student Success Intelligence Framework (SSIF)
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUTPUT_DIR = Path("docs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PORT = 9222

try:
    import websocket
except ImportError:
    import pip
    pip.main(["install", "websocket-client"])
    import websocket

def run():
    print("Launching Chrome in headless mode with remote debugging...", flush=True)
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--remote-allow-origins=*",
        r"--user-data-dir=C:\Users\Lenovo\AppData\Local\Temp\ssif_chrome_snap",
        "--window-size=1680,1150",
        "--hide-scrollbars",
        "http://localhost:8502",
    ])

    time.sleep(7)

    try:
        tabs_raw = urllib.request.urlopen(f"http://localhost:{PORT}/json/list").read()
        tabs = json.loads(tabs_raw)
        page_tabs = [t for t in tabs if t.get("type") == "page"]
        if not page_tabs:
            raise RuntimeError(f"No page tab found in Chrome tabs: {tabs}")
        
        target_tab = next((t for t in page_tabs if "8502" in t.get("url", "")), page_tabs[0])
        ws_url = target_tab["webSocketDebuggerUrl"]
        print(f"Connected to CDP page tab: {target_tab.get('title')} at {ws_url}", flush=True)

        ws = websocket.create_connection(ws_url, timeout=15)
        msg_id = 0

        def send_cdp(method, params=None):
            nonlocal msg_id
            msg_id += 1
            payload = {"id": msg_id, "method": method, "params": params or {}}
            ws.send(json.dumps(payload))
            while True:
                resp = json.loads(ws.recv())
                if resp.get("id") == msg_id:
                    return resp.get("result", {})

        send_cdp("Page.enable")
        send_cdp("Runtime.enable")
        send_cdp("Emulation.setDeviceMetricsOverride", {
            "width": 1680,
            "height": 1150,
            "deviceScaleFactor": 1.5,
            "mobile": False,
        })

        time.sleep(4)

        def take_screenshot(filename):
            res = send_cdp("Page.captureScreenshot", {"format": "png"})
            data = base64.b64decode(res["data"])
            filepath = OUTPUT_DIR / filename
            filepath.write_bytes(data)
            print(f"  --> Saved: {filename} ({len(data):,} bytes)", flush=True)

        def select_radio_index(idx, wait_sec=4):
            js = f"""
            (() => {{
                const radios = Array.from(document.querySelectorAll('[data-testid="stRadioOption"]'));
                if (radios.length > {idx}) {{
                    const el = radios[{idx}].querySelector('input') || radios[{idx}];
                    el.click();
                    return "Selected radio index {idx}: " + radios[{idx}].innerText.substring(0, 30);
                }}
                return "Index {idx} not found (found: " + radios.length + ")";
            }})()
            """
            res = send_cdp("Runtime.evaluate", {"expression": js, "returnByValue": True})
            print(f"Navigation: {res.get('result', {}).get('value')}", flush=True)
            time.sleep(wait_sec)

        def scroll_to(y, wait_sec=2):
            send_cdp("Runtime.evaluate", {"expression": f"window.scrollTo(0, {y});"})
            time.sleep(wait_sec)

        # ─── Page 0: Executive Overview ──────────────────────────────────────────
        print("\n--- 1. Executive Overview ---", flush=True)
        select_radio_index(0, wait_sec=3)
        scroll_to(0)
        take_screenshot("01_executive_overview.png")
        
        scroll_to(480)
        take_screenshot("01b_leaderboard_pillars.png")

        # ─── Page 1: Data Audit ──────────────────────────────────────────────────
        print("\n--- 2. Data Audit & Missingness ---", flush=True)
        select_radio_index(1, wait_sec=4)
        scroll_to(0)
        take_screenshot("02_data_audit.png")

        # ─── Page 2: Academic Retention & Trajectory Intelligence ────────────────
        print("\n--- 3. Academic Retention & Simulator ---", flush=True)
        select_radio_index(2, wait_sec=4)
        scroll_to(250)
        take_screenshot("03a_retention_simulator.png")

        # 3D Risk Surface
        print("  Capturing 3D Risk Surface...", flush=True)
        scroll_to(1100, wait_sec=3)
        take_screenshot("03b_retention_3d_surface.png")

        # Click Tab 2: Trajectory Ribbons
        print("  Switching to 3D Trajectory Ribbons Tab...", flush=True)
        send_cdp("Runtime.evaluate", {"expression": """
            (() => {
                const tabs = Array.from(document.querySelectorAll('button[data-baseweb="tab"]'));
                const ribbonTab = tabs.find(t => t.textContent.includes("Trajectory Ribbons"));
                if (ribbonTab) { ribbonTab.click(); return "clicked ribbon tab"; }
                return "tab not found";
            })()
        """})
        time.sleep(4)
        take_screenshot("03c_retention_3d_ribbons.png")

        # ─── Page 3: Survival Analysis ───────────────────────────────────────────
        print("\n--- 4. Survival Analysis & Hazard ---", flush=True)
        select_radio_index(3, wait_sec=4)
        scroll_to(0)
        take_screenshot("04_survival_analysis.png")

        # ─── Page 4: Career Placement ────────────────────────────────────────────
        print("\n--- 5. Career Placement & Diagnostics ---", flush=True)
        select_radio_index(4, wait_sec=4)
        scroll_to(0)
        take_screenshot("05_career_placement.png")

        # ─── Page 5: Explainable AI & SHAP ───────────────────────────────────────
        print("\n--- 6. Explainable AI & SHAP ---", flush=True)
        select_radio_index(5, wait_sec=4)
        scroll_to(220)
        take_screenshot("06a_explainability_shap.png")

        # 3D Feature Space
        print("  Capturing 3D Multivariate Feature Space...", flush=True)
        scroll_to(950, wait_sec=3)
        take_screenshot("06b_multivariate_3d_scatter.png")

        # ─── Page 6: DLSM Compatibility & Construct Bridge ───────────────────────
        print("\n--- 7. DLSM Compatibility & Construct Bridge ---", flush=True)
        select_radio_index(6, wait_sec=4)
        scroll_to(0)
        take_screenshot("07_dlsm_construct_bridge.png")

        print("\nAll high-fidelity screenshots successfully captured!", flush=True)
        ws.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    run()
