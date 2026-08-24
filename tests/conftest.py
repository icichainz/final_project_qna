import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["PRELOAD"] = "0"          # never load the real 730 MB index in tests

# Tests pin the SHIPPED defaults, not the machine's deployment flags: the
# operator's .env flips switches in production, and a flipped switch must not
# change what the suite asserts (observed: 4 failures the day PLANNER/VERIFY
# went live). Tests that exercise a switch set it explicitly via monkeypatch.
# VERIFY_REPAIR is absent on purpose: eac4c94 removed the repair pathway, so
# nothing reads that variable any more and pinning it would pin a fiction.
for _flag, _default in (("PLANNER", "0"), ("VERIFY", "0"), ("VERIFY_LLM", "1"),
                        ("CONDUCTOR", "1"), ("RERANK", "0"),
                        ("SECTION_EXPAND", "0"), ("INDEX_NAME", "default")):
    os.environ[_flag] = _default
