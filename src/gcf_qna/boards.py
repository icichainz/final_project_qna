"""GCF board-meeting metadata shared across the app and registry tooling.

Years verified against 271 corpus page-1 dates (see the year-assist work);
gaps B.12/B.17 interpolated from adjacent meetings.
"""
from __future__ import annotations

import re

BOARD_YEARS = {11: 2015, 12: 2016, 13: 2016, 14: 2016, 15: 2016, 16: 2017,
               17: 2017, 18: 2017, 19: 2018, 20: 2018, 21: 2018, 22: 2019,
               23: 2019, 24: 2019, 25: 2020, 26: 2020, 27: 2020, 28: 2021,
               29: 2021, 30: 2021, 31: 2022, 32: 2022, 33: 2022, 34: 2022,
               35: 2023, 36: 2023, 37: 2023, 38: 2024, 39: 2024, 40: 2024,
               41: 2025, 42: 2025, 43: 2025}

BOARD_RE = re.compile(r"[-_]b\.?(\d+)[-_]", re.I)


def board_of(doc_id: str):
    m = BOARD_RE.search(doc_id)
    return int(m.group(1)) if m else None


def year_of(doc_id: str):
    b = board_of(doc_id)
    return BOARD_YEARS.get(b) if b else None
