"""A stand-in for the `lean` command in tests. It understands just enough of a file:
a line containing ERR is an error, a theorem whose text contains `sorry` uses sorryAx,
one containing `cheat` uses the axiom `cheat`, and `#print axioms` reports accordingly."""

import re
import sys
import time

path = sys.argv[-1]
text = open(path, encoding="utf-8").read()
if "SLOW" in text:
    time.sleep(5)
code = 0
axioms: dict[str, list[str]] = {}
chunks = re.split(r"(?m)^(?=theorem )", text)
for chunk in chunks:
    m = re.match(r"theorem (\S+)", chunk)
    if not m:
        continue
    name = m.group(1)
    used = ["propext"]
    ref = re.search(r":= _root_\.CirclePackingConstants\.(\w+)", chunk)
    if ref:
        target = axioms.get(ref.group(1))
        if target is None:
            print(f"{path}:1:0: error: unknown constant 'CirclePackingConstants.{ref.group(1)}'")
            code = 1
            used.append("sorryAx")
        else:
            used = target
    else:
        if "sorry" in chunk:
            used.append("sorryAx")
        if "cheat" in chunk:
            used.append("cheat")
    axioms[name.split(".")[-1] if not name.startswith("Grader") else name] = used
for n, line in enumerate(text.splitlines(), 1):
    if "ERR" in line:
        print(f"{path}:{n}:2: error: unsolved goals")
        code = 1
    if "sorry" in line:
        print(f"{path}:{n}:2: warning: declaration uses 'sorry'")
    m = re.match(r"#print axioms (\S+)", line)
    if m:
        print(f"{path}:{n}:0: info: '{m.group(1)}' depends on axioms: [{', '.join(axioms.get(m.group(1), ['sorryAx']))}]")
sys.exit(code)
