"""M8: smoke test of the deployed demo.

  python scripts/smoke_public.py                 free checks only (UI, health, status, documents, auth)
  python scripts/smoke_public.py --ask           also asks ONE question twice (~$0.001; 2nd answer from cache)

Never sends the admin token: it only checks that admin actions WITHOUT the token are refused.
"""
import argparse
import sys
import time

import requests

URL = "https://ai-investigation-engine-fqlm.onrender.com"


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--ask", action="store_true")
    a = ap.parse_args()
    B, results = a.url.rstrip("/"), []

    t = time.time()
    for _ in range(12):  # Render free sleeps after 15 min: the first request may take ~1 min
        try:
            h = requests.get(f"{B}/health", timeout=90)
            if h.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    results.append(check("health", h.ok and h.json().get("db") == "ok", f"{h.text} after {time.time() - t:.0f}s"))
    r = requests.get(B + "/", timeout=30)
    results.append(check("UI page", r.ok and "AI Investigation Engine" in r.text))
    s = requests.get(f"{B}/api/status", timeout=30).json()
    results.append(check("status", "cap_usd" in s, str(s)))
    d = requests.get(f"{B}/api/documents", params={"uploaded_only": "true", "limit": 100}, timeout=30).json()
    results.append(check("synthetic documents present (data survived restarts)", d["total"] >= 17, f"total={d['total']}"))
    d = requests.get(f"{B}/api/documents", params={"q": "Scott Derrickson"}, timeout=30).json()
    results.append(check("title search", any(x["title"] == "Scott Derrickson" for x in d["documents"])))
    r = requests.delete(f"{B}/api/documents/1", timeout=30)
    results.append(check("delete without token refused", r.status_code == 401, str(r.status_code)))
    r = requests.post(f"{B}/api/documents", files={"file": ("a.txt", b"hello")}, headers={"X-Admin-Token": "wrong"}, timeout=30)
    results.append(check("upload with wrong token refused", r.status_code == 401, str(r.status_code)))

    if a.ask:
        q = {"question": "Which team owns the fraud-check service?"}
        for i in (1, 2):
            t = time.time()
            r = requests.post(f"{B}/api/query", json=q, timeout=180)
            j = r.json()
            results.append(check(f"question #{i}", r.ok and j.get("status") == "answered",
                                 f"{time.time() - t:.1f}s cached={j.get('cached')} cost=${j.get('cost_usd', 0):.5f} "
                                 f"answer={j.get('answer', j.get('detail'))!r:.120} sources={[x['title'] for x in j.get('sources', [])]}"))
        print(requests.get(f"{B}/api/status", timeout=30).json())

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
