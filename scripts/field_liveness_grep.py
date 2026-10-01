#!/usr/bin/env python3
"""Field-mention recall list for manual buffer-liveness tracing.

This is Candidate B from the 2026-10-01 auto-fusion-tooling scoping pass
(see project memory `project_infero_autofusion_goal.md` and
`project_infero_perf_gap.md` for the full writeup). It exists because every
real fusion landed that day required a BY-HAND grep + control-flow trace to
answer "does anything else read/write this buffer between the write I'm
eliminating and the read I'm eliminating" -- repetitive, easy to under-scope
by eye, worth a recall aid.

WHAT THIS IS: a plain, line-based recall list -- every line in a given
range that mentions a field path, plus (best-effort, NOT structurally
verified) which `if`/`else if`/`else` branch each mention's own line sits
inside, found by indentation, not by parsing Rust.

WHAT THIS IS NOT, AND MUST NEVER BE TREATED AS: a safety proof. It cannot
see through a reference/alias (`let b = &self.act; b.xb`), a destructuring
bind, a macro expansion, or a field access spelled through a helper method.
Much more importantly, even when it DOES find every textual mention (its
real, validated job -- see the real 897ee7c/44c1f68 case studies this was
checked against, in project memory), it has no idea whether two mentions
are on MUTUALLY EXCLUSIVE branches (e.g. one guarded by `w_gate.ty == Q4K`,
the other by `w_gate.ty == F4E2M1`) -- that is a semantic judgment call about
the governing enum's possible values, not a text-matching problem, and this
script does not attempt it. Treat its output as "don't forget to look at
these lines," never as "these lines are safe" or "these lines conflict." A
human still reads every line this prints and decides reachability/mutual-
exclusion by hand, the same as before -- this only saves the first,
mechanical pass of finding the candidate lines, not the real judgment call
about them.

A REAL, CONCRETE FALSE-NEGATIVE PATTERN TO WATCH FOR (found in this
codebase's own real fusion commits, not hypothetical): a field can be
SLICED INTO A LOCAL before the call that actually reads/writes it, e.g.

    let mut out_view = if cond { self.act.x.slice_mut(..n * d) }
                        else    { self.act.proj.slice_mut(..n * d) };
    Self::matmul_pre(..., &mut out_view, ...)?;   // the REAL access is HERE

If your trace window starts after the `let` line (a natural mistake, since
that's where the field name `.act.proj` textually last appears), this
script reports zero mentions for a window that contains the real access --
a false "nothing here," exactly where a human might stop reading because
the tool found nothing. Rule of thumb: when you see a field sliced or
aliased into a local (`let v = self.act.X...`), re-run this script for that
LOCAL's own name too, and never end a trace window between the alias site
and the call that actually consumes it.

Usage:
    python3 scripts/field_liveness_grep.py <file> <start_line> <end_line> <field_path> [field_path...]

Example (one of the real cases this script was validated against):
    python3 scripts/field_liveness_grep.py \\
        crates/model/src/lib.rs 6992 7415 '.act.xb' '.act.proj'
"""

import re
import sys
from dataclasses import dataclass


@dataclass
class Mention:
    lineno: int
    text: str
    branch_guard: str | None  # best-effort, see module doc comment


def branch_guard_for(lines: list[str], idx: int, range_start_idx: int) -> str | None:
    """Best-effort "which branch is this line inside" -- walks upward by
    INDENTATION only (never brace-counts or parses Rust), stopping at the
    first shallower line that looks like a branch header. This is exactly
    the kind of thing that is cheap and approximate by design: see the
    module doc comment for why it must not be read as a safety claim.
    """
    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    own_indent = indent_of(lines[idx])
    branch_re = re.compile(r"^\s*(?:\}\s*)?(?:else\s+)?if\b|^\s*\}\s*else\s*\{")
    i = idx - 1
    while i >= range_start_idx:
        line = lines[i]
        stripped = line.strip()
        if stripped and indent_of(line) < own_indent:
            if branch_re.match(line):
                return f"line {i + 1}: {stripped}"
            # A shallower non-branch line (e.g. the enclosing `match` arm's
            # own opener, or the function signature) means we've walked out
            # of any branch -- stop rather than keep climbing past it.
            own_indent = indent_of(line)
            if own_indent == 0:
                break
        i -= 1
    return None


def find_mentions(path: str, start: int, end: int, field: str) -> list[Mention]:
    with open(path, encoding="utf-8") as f:
        all_lines = f.read().splitlines()
    # 1-indexed, inclusive range, same convention as `rg -n` / editor line numbers.
    start_idx = max(0, start - 1)
    end_idx = min(len(all_lines), end)
    needle = re.escape(field)
    pat = re.compile(needle)
    out = []
    for i in range(start_idx, end_idx):
        if pat.search(all_lines[i]):
            guard = branch_guard_for(all_lines, i, start_idx)
            out.append(Mention(lineno=i + 1, text=all_lines[i].rstrip(), branch_guard=guard))
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__)
        return 2
    path, start_s, end_s = argv[1], argv[2], argv[3]
    fields = argv[4:]
    start, end = int(start_s), int(end_s)

    print(f"# Field-mention recall list: {path}:{start}-{end}")
    print("# NOT a safety proof -- text recall only. See this script's module doc comment.\n")
    for field in fields:
        mentions = find_mentions(path, start, end, field)
        print(f"## `{field}` -- {len(mentions)} mention(s)")
        if not mentions:
            print("  (none found in range)")
        for m in mentions:
            guard = f"  [in: {m.branch_guard}]" if m.branch_guard else ""
            print(f"  {m.lineno}: {m.text.strip()}{guard}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
