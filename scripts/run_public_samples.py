#!/usr/bin/env python3
"""POST every public sample case to a running service and judge the responses.

Usage:
    python scripts/run_public_samples.py                          # http://localhost:8000
    python scripts/run_public_samples.py --base-url https://your-deployment.example.com
    python scripts/run_public_samples.py --case SAMPLE-03 --show

Exit code 0 when every case passes schema, interpretation, validity, and cost checks.
Standard library only, so it runs from any Python 3.9+ without installing anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import judge_case  # noqa: E402

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        body = err.read()
        try:
            return err.code, json.loads(body)
        except ValueError:
            return err.code, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", action="append", help="only run these case ids (repeatable)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--show", action="store_true", help="print each response's interpretation")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
    url = args.base_url.rstrip("/") + "/optimize-energy"

    passed, latencies = 0, []
    for case in cases:
        started = time.monotonic()
        status, body = post_json(url, case["input"], args.timeout)
        latency = time.monotonic() - started
        latencies.append(latency)
        if status != 200 or body is None:
            print(f"FAIL {case['id']:<10} HTTP {status} {json.dumps(body)[:300] if body else ''}")
            continue
        result = judge_case(case, body)
        problems = [f"{k}: {m}" for k, msgs in result.items() for m in msgs]
        ref = case["expected_output"]["total_cost_bdt"]
        line = f"{case['id']:<10} cost={body['total_cost_bdt']:>10.2f} ref={ref:>10.2f} latency={latency:5.2f}s"
        if problems:
            print(f"FAIL {line}")
            for p in problems:
                print(f"       - {p}")
        else:
            passed += 1
            print(f"PASS {line}")
        if args.show:
            for e in body["directive_interpretation"]:
                print(f"       note {e['note_index']}: {e['directive_type']} {json.dumps(e['structured_adjustment'])}")

    if latencies:
        p95 = sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)]
        print(f"\n{passed}/{len(cases)} cases passed | latency median={statistics.median(latencies):.2f}s p95={p95:.2f}s")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
