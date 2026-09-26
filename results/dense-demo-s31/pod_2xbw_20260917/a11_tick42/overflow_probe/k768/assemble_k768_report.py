#!/usr/bin/env python3
"""K768_CPU_REPORT.src.md -> K768_CPU_REPORT.md: every `{{SECTION:<heading prefix>}}` marker is replaced by that section of
k768_tables.md (the text under the first heading that starts with the prefix, up to the next heading of the same or a
higher level; the heading line itself is dropped), and `{{FILE:<name>}}` by the file's content in a code fence. The tables
are generated from the JSONL records by analyze_k768.py, so the report's tables cannot drift from the records."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
tables = open(os.path.join(HERE, "k768_tables.md")).read().split("\n")


def section(prefix):
    for i, line in enumerate(tables):
        if line.startswith(prefix):
            level = len(line) - len(line.lstrip("#"))
            j = i + 1
            while j < len(tables):
                m = re.match(r"^(#+) ", tables[j])
                if m and len(m.group(1)) <= level: break
                j += 1
            return "\n".join(tables[i + 1:j]).strip("\n")
    return f"(section `{prefix}` not found in k768_tables.md)"


src = open(os.path.join(HERE, "K768_CPU_REPORT.src.md")).read()
src = re.sub(r"\{\{SECTION:(.*?)\}\}", lambda m: section(m.group(1)), src)
src = re.sub(r"\{\{FILE:(.*?)\}\}", lambda m: "```\n" + open(os.path.join(HERE, m.group(1))).read().rstrip("\n") + "\n```", src)
open(os.path.join(HERE, "K768_CPU_REPORT.md"), "w").write(src)
print("wrote K768_CPU_REPORT.md,", len(src.split("\n")), "lines")
