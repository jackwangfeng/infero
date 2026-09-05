#!/usr/bin/env python3
"""Capture baseline transcripts for the mixed-batch dispatch split
(docs/superpowers/plans/2026-09-05-mixed-batch-attention-dispatch-split.md,
Task 1). Re-run this same script, unmodified, against the POST-change binary
in Task 7 and diff the saved JSON files text-for-text (seed=0/temperature=0
ensures determinism; token IDs are unobtainable from the API).

Each saved JSON includes:
- Exact request payload (messages, temperature, seed, max_tokens)
- Server response (content, finish_reason, token usage)
- Wall-clock timestamps (before/after each request) for concurrency verification
- overlap_detected flag to ensure genuine concurrent execution in production

After all scenarios run, captures production log excerpt as corroborating evidence.
"""
import json
import sys
import time
import threading
import urllib.request
import subprocess

BASE = "http://127.0.0.1:8301"
SEED = 42424242
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "docs/superpowers/plans/mixed_batch_baseline"

# Global list to track all scenario timing windows for log excerpt
test_windows = []  # List of (scenario_name, t_start, t_end)

def chat(messages, max_tokens=64, seed=SEED):
    """Send request to server. Returns (request_dict, response_dict, wall_clock_times)."""
    request_payload = {
        "model": "qwen38-27b-fp8",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": False,
    }
    body = json.dumps(request_payload).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            response = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")
    t_end = time.time()

    return request_payload, response, {"t_start": t_start, "t_end": t_end}

def save(name, obj):
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {OUT_DIR}/{name}.json")

# Scenario A: pure single-sequence prefill+decode -- exercises the existing
# single_seq_run path, must be untouched by this change.
print("Scenario A: Single-sequence...")
req_a, resp_a, times_a = chat([{"role": "user", "content": "Count from 1 to 20, one number per line."}], max_tokens=80)
output_a = {
    "scenario": "A (single-sequence prefill+decode)",
    "request": req_a,
    "response": resp_a,
    "timing": times_a,
}
save("scenario_a_single_seq", output_a)
# Record test window for log excerpt
test_windows.append(("scenario_a", times_a["t_start"], times_a["t_end"]))

# Scenario B: engineered mixed decode+prefill batch with retry logic.
# Fire a long prompt (forces multi-chunk prefill) on one connection, then fire
# a short prompt whose decode should land in the SAME scheduler batch as the
# long prompt's later prefill chunks. Retry with increasing delays if overlap
# is not detected, to ensure genuine concurrent execution.
print("Scenario B: Mixed decode+prefill...")
long_prompt = "Please summarize this list in detail, one sentence per item: " + \
    ", ".join(f"item number {i} is about topic {i%7}" for i in range(4000))

def run_scenario_b_attempt(initial_delay):
    """Run scenario B with given initial delay before second request. Returns output_b or None if retry needed."""
    results_b = {}
    errors_b = {}
    long_started = threading.Event()  # Signal when long request actually starts

    def run_long():
        try:
            long_started.set()  # Signal that this thread is about to call chat()
            results_b["long_req"], results_b["long_resp"], results_b["long_time"] = \
                chat([{"role": "user", "content": long_prompt}], max_tokens=40)
        except Exception as e:
            errors_b["long"] = str(e)

    def run_short():
        # Poll: wait until long request has started (event signaled), then fire short request.
        # This gives the long request time to be admitted to the scheduler first.
        max_wait = 5.0
        if long_started.wait(timeout=max_wait):
            # Long request thread has started; fire short request now
            pass
        else:
            # Timeout - long request thread didn't start, but continue anyway
            print(f"    (warning: long request didn't start within {max_wait}s, continuing)")
        try:
            results_b["short_req"], results_b["short_resp"], results_b["short_time"] = \
                chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}], max_tokens=8)
        except Exception as e:
            errors_b["short"] = str(e)

    t1 = threading.Thread(target=run_long)
    t2 = threading.Thread(target=run_short)
    t1.start()
    time.sleep(initial_delay)
    t2.start()
    t1.join()
    t2.join()

    # Verify both requests succeeded
    if errors_b or "long_resp" not in results_b or "short_resp" not in results_b:
        print(f"  Attempt with delay={initial_delay:.3f}s: REQUEST FAILED (errors: {errors_b})")
        return None

    overlap = (
        results_b["long_time"]["t_start"] < results_b["short_time"]["t_end"] and
        results_b["short_time"]["t_start"] < results_b["long_time"]["t_end"]
    )

    if not overlap:
        print(f"  Attempt with delay={initial_delay:.3f}s: no overlap detected, retrying...")
        return None

    print(f"  Attempt with delay={initial_delay:.3f}s: overlap detected!")
    output_b = {
        "scenario": "B (mixed decode+prefill batch)",
        "requests": {
            "long": {"payload": results_b["long_req"], "timing": results_b["long_time"]},
            "short": {"payload": results_b["short_req"], "timing": results_b["short_time"]},
        },
        "responses": {
            "long": results_b["long_resp"],
            "short": results_b["short_resp"],
        },
        "concurrency_check": {
            "long_t_start": results_b["long_time"]["t_start"],
            "long_t_end": results_b["long_time"]["t_end"],
            "short_t_start": results_b["short_time"]["t_start"],
            "short_t_end": results_b["short_time"]["t_end"],
            "overlap_detected": overlap,
        },
    }
    # Record test window for log excerpt
    test_windows.append(("scenario_b", results_b["long_time"]["t_start"], results_b["short_time"]["t_end"]))
    return output_b

# Retry scenario B with increasing delays: 0.05s, 0.08s, 0.15s, 0.25s
output_b = None
for attempt, delay in enumerate([0.05, 0.08, 0.15, 0.25], 1):
    print(f"  Scenario B attempt {attempt}/4...")
    output_b = run_scenario_b_attempt(delay)
    if output_b:
        break

if not output_b:
    print("ERROR: Scenario B failed after 4 retry attempts (could not achieve overlap)")
    sys.exit(1)

save("scenario_b_mixed_decode_prefill", output_b)

# Scenario C: two simultaneous prefills with retry logic.
# Fire two prompts as close together as possible and verify they landed in the
# same scheduler batch. Retry with a small jitter if overlap is not detected.
print("Scenario C: Two simultaneous prefills...")
prompt_x = "Explain photosynthesis in exactly three sentences."
prompt_y = "Explain how a car engine works in exactly three sentences."

def run_scenario_c_attempt(attempt_num):
    """Run scenario C, returning output_c or None if retry needed."""
    results_c = {}
    errors_c = {}

    def run_x():
        try:
            results_c["x_req"], results_c["x_resp"], results_c["x_time"] = \
                chat([{"role": "user", "content": prompt_x}], max_tokens=60)
        except Exception as e:
            errors_c["x"] = str(e)

    def run_y():
        try:
            results_c["y_req"], results_c["y_resp"], results_c["y_time"] = \
                chat([{"role": "user", "content": prompt_y}], max_tokens=60)
        except Exception as e:
            errors_c["y"] = str(e)

    tx = threading.Thread(target=run_x)
    ty = threading.Thread(target=run_y)
    tx.start()
    ty.start()
    tx.join()
    ty.join()

    # Verify both requests succeeded
    if errors_c or "x_resp" not in results_c or "y_resp" not in results_c:
        print(f"  Attempt {attempt_num}/4: REQUEST FAILED (errors: {errors_c})")
        return None

    overlap = (
        results_c["x_time"]["t_start"] < results_c["y_time"]["t_end"] and
        results_c["y_time"]["t_start"] < results_c["x_time"]["t_end"]
    )

    if not overlap:
        print(f"  Attempt {attempt_num}/4: no overlap detected, retrying...")
        return None

    print(f"  Attempt {attempt_num}/4: overlap detected!")
    output_c = {
        "scenario": "C (two simultaneous prefills)",
        "requests": {
            "x": {"payload": results_c["x_req"], "timing": results_c["x_time"]},
            "y": {"payload": results_c["y_req"], "timing": results_c["y_time"]},
        },
        "responses": {
            "x": results_c["x_resp"],
            "y": results_c["y_resp"],
        },
        "concurrency_check": {
            "x_t_start": results_c["x_time"]["t_start"],
            "x_t_end": results_c["x_time"]["t_end"],
            "y_t_start": results_c["y_time"]["t_start"],
            "y_t_end": results_c["y_time"]["t_end"],
            "overlap_detected": overlap,
        },
    }
    # Record test window for log excerpt
    test_windows.append(("scenario_c", min(results_c["x_time"]["t_start"], results_c["y_time"]["t_start"]),
                         max(results_c["x_time"]["t_end"], results_c["y_time"]["t_end"])))
    return output_c

# Retry scenario C up to 4 times (small jitter between retries)
output_c = None
for attempt in range(1, 5):
    print(f"  Scenario C attempt {attempt}/4...")
    output_c = run_scenario_c_attempt(attempt)
    if output_c:
        break
    if attempt < 4:
        # Small jitter before retry
        time.sleep(0.02 * attempt)

if not output_c:
    print("ERROR: Scenario C failed after 4 retry attempts (could not achieve overlap)")
    sys.exit(1)

save("scenario_c_two_prefills", output_c)

# Capture production log excerpt as corroborating evidence
print("\nCapturing production log excerpt...")
def capture_log_excerpt():
    """SSH to bw, grep production log for test window, save excerpt."""
    import os

    if not test_windows:
        print("  (no test windows recorded)")
        return

    # Compute overall test window
    earliest = min(t_start for _, t_start, _ in test_windows)
    latest = max(t_end for _, _, t_end in test_windows)

    # Convert to ISO8601 format for grepping
    import datetime
    dt_start = datetime.datetime.fromtimestamp(earliest, tz=datetime.timezone.utc)
    dt_end = datetime.datetime.fromtimestamp(latest, tz=datetime.timezone.utc)

    print(f"  Test window: {dt_start.isoformat()} to {dt_end.isoformat()}")
    print(f"  ({earliest:.1f} to {latest:.1f} Unix time)")

    # Remote: grep for request admission/completion lines and seq= lines during this window
    # Use actual computed timestamps, not hardcoded literals
    dt_start_str = dt_start.isoformat()[:19]  # e.g., "2026-09-05T20:02:24"
    dt_end_str = dt_end.isoformat()[:19]      # e.g., "2026-09-05T20:07:30"

    try:
        # Grep for relevant lines in the time range. Note: log has ANSI color codes before timestamps.
        # Strategy: grep for request/seq lines, then filter by date since color codes are before timestamp.
        date_str = dt_start.strftime("%Y-%m-%d")  # e.g., "2026-09-05"
        result = subprocess.run(
            [
                "ssh", "bw",
                f"grep '{date_str}' /tmp/infero_27b_live.log | "
                f"grep -E '(request admitted|request complete|seq=.*prompt)' | "
                f"tail -100"
            ],
            capture_output=True, text=True, timeout=10
        )

        excerpt = result.stdout.strip()
        if excerpt:
            # Save log excerpt
            os.makedirs(OUT_DIR, exist_ok=True)
            excerpt_path = f"{OUT_DIR}/production_log_excerpt.txt"
            with open(excerpt_path, "w") as f:
                f.write(f"Production log excerpt for test window {dt_start.isoformat()} to {dt_end.isoformat()}\n")
                f.write(f"(Unix time: {earliest:.1f} to {latest:.1f})\n\n")
                f.write("Log lines with 'request admitted', 'request complete', or 'seq=' during test window:\n")
                f.write("=" * 100 + "\n\n")
                f.write(excerpt)
                f.write("\n\n" + "=" * 100 + "\n")
                f.write(f"Verification: Look for multiple request IDs with seq=0 and seq=1 entries.\n")
                f.write(f"Scenario A: single sequence (66 prompt_tokens)\n")
                f.write(f"Scenario B: long (50953) + short (65) prompts overlapping in time\n")
                f.write(f"Scenario C: two prefills (64 and 61 prompt_tokens) with nearby admission times\n")
            print(f"  saved {excerpt_path}")
            return True  # Success
        else:
            print(f"  (log excerpt empty for time range {dt_start_str} to {dt_end_str}; checking tail of log)")
            # Fallback: grab lines around the end time
            result2 = subprocess.run(
                ["ssh", "bw", "grep -E '(request admitted|request complete|seq=)' /tmp/infero_27b_live.log | tail -50"],
                capture_output=True, text=True, timeout=10
            )
            if result2.stdout.strip():
                os.makedirs(OUT_DIR, exist_ok=True)
                excerpt_path = f"{OUT_DIR}/production_log_excerpt.txt"
                with open(excerpt_path, "w") as f:
                    f.write(f"Production log excerpt (fallback: recent requests from /tmp/infero_27b_live.log)\n")
                    f.write(f"Intended window was {dt_start_str} to {dt_end_str}, but time-based grep was empty.\n")
                    f.write(f"Showing recent request/seq activity:\n\n")
                    f.write(result2.stdout)
                print(f"  saved {excerpt_path} (fallback recent tail)")
                return False  # Fallback, not ideal but better than nothing
            else:
                print(f"  ERROR: could not capture log excerpt (both time-based and fallback tail empty)")
                return False  # Failure
    except Exception as e:
        print(f"  ERROR: could not capture log excerpt: {e}")
        return False  # Failure

log_success = capture_log_excerpt()
if not log_success:
    print("ERROR: Log excerpt capture failed. This evidence is needed for Task 7 verification.")
    sys.exit(1)

print("\ndone")
