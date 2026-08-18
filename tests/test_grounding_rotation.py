"""Grounding must survive page rotation, and must not trust single-token lines.

Two independent defects in the grounded viewer, both verified against the real
corpus before being fixed here.

ROTATION.  extraction/vlm.py records line and figure rects from the PDF's
*unrotated* page space (that is what page.get_text and cluster_drawings report),
but the cached JPEG is rendered with /Rotate applied. On the 205 rotated pages
in the corpus the two spaces disagree, so rects land transposed or clean off the
image. Real case, 214_gcf-b19-22-add03.pdf page 115 (rotation=90, JPEG
1609x1244): 21 of its 132 line rects fell outside the image and, of the 111 that
did land, 61 sat on blank paper. Table rects are the exception -- find_tables()
already reports display coordinates -- so they must be left alone.

The fix is applied when a sidecar is *loaded*, not when it is written, so all
41,500 cached pages are healed without re-extraction.

SINGLE-TOKEN LINES.  A line qualified on a coverage *fraction* alone, so any
one-token line ('5', a bullet, a page number) was a 100% match for every chunk
containing that token. Real case, page cache 0b9019a7e2be page 7: the chunk
'Annex 5' matched the standalone lines '5' and '5.' -- a footer page number and
a list marker -- at confidence 2/2 = 1.0, painting green boxes over a page
number. Legitimate short matches ('Annex 5' against a real 'Annex 5' heading)
must keep working.
"""
import json

import pytest

from gcf_qna.rag.ground import (
    MIN_CONFIDENCE,
    ground_chunk,
    load_page_assets,
    normalize_boxes,
    rotate_rect,
)

# --------------------------------------------------------------- helpers ----


def write_cache(tmp_path, boxes, page=7, doc_id="doc"):
    """Build a minimal page-cache directory holding one boxes sidecar."""
    name = f"page_{page:04d}.boxes.json"
    (tmp_path / name).write_text(json.dumps(boxes), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"pdf_name": f"{doc_id}.pdf",
                    "pages": [{"n": page, "w": boxes["w"], "h": boxes["h"],
                               "boxes": name}]}),
        encoding="utf-8")
    return tmp_path


def line_page(texts, w=1000, h=1000, rotation=0):
    """A boxes sidecar whose lines are stacked 40px apart down the page."""
    return {
        "w": w, "h": h, "zoom": 2.0, "rotation": rotation,
        "lines": [{"bbox": [50.0, 40.0 * i, 400.0, 40.0 * i + 30.0], "text": t}
                  for i, t in enumerate(texts, start=1)],
        "tables": [], "figures": [],
    }


def inside(bbox, w, h, tol=1.0):
    x0, y0, x1, y1 = bbox
    return (x0 >= -tol and y0 >= -tol and x1 <= w + tol and y1 <= h + tol
            and x1 >= x0 and y1 >= y0)


# ================================================================ rotation ==
# Geometry note for the hand-computed values below. A rotation of 90 or 270
# turns an unrotated page of Wu x Hu pixels into an image of Wi=Hu, Hi=Wu, so
# the image dims are the unrotated dims swapped. Every case below uses an image
# of 1200 x 800.


def test_rotation_zero_is_identity():
    assert rotate_rect([100, 157, 126, 212], 0, 1200, 800) == [100.0, 157.0, 126.0, 212.0]


def test_rotation_90_reviewer_repro():
    """The exact repro from review: unrotated (100,157,126,212) -> (988,100,1043,126).

    Cross-checked against pymupdf: Rect(b) * page.rotation_matrix agrees to 0.0 px.
    """
    assert rotate_rect([100, 157, 126, 212], 90, 1200, 800) == [988.0, 100.0, 1043.0, 126.0]


def test_rotation_90_hand_computed():
    # Unrotated page is 800 wide x 1200 tall. 90 maps (x,y) -> (1200 - y, x).
    # (10,20)->(1180,10) and (60,120)->(1080,60); reordered: (1080,10,1180,60).
    assert rotate_rect([10, 20, 60, 120], 90, 1200, 800) == [1080.0, 10.0, 1180.0, 60.0]


def test_rotation_180_hand_computed():
    # Dims are unchanged by 180: unrotated page is 1200 x 800.
    # (x,y) -> (1200 - x, 800 - y): (100,150)->(1100,650), (300,250)->(900,550).
    assert rotate_rect([100, 150, 300, 250], 180, 1200, 800) == [900.0, 550.0, 1100.0, 650.0]


def test_rotation_270_hand_computed():
    # Unrotated page is 800 wide x 1200 tall. 270 maps (x,y) -> (y, 800 - x).
    # (100,157)->(157,700) and (126,212)->(212,674); reordered: (157,674,212,700).
    assert rotate_rect([100, 157, 126, 212], 270, 1200, 800) == [157.0, 674.0, 212.0, 700.0]


@pytest.mark.parametrize("rotation, page_wh", [(90, (800, 1200)),
                                               (180, (1200, 800)),
                                               (270, (800, 1200))])
def test_full_page_rect_maps_to_full_image(rotation, page_wh):
    """The whole unrotated page must cover exactly the whole rendered image."""
    pw, ph = page_wh
    assert rotate_rect([0, 0, pw, ph], rotation, 1200, 800) == [0.0, 0.0, 1200.0, 800.0]


@pytest.mark.parametrize("rotation, corner, expected_quadrant", [
    (90, [0, 0, 10, 20], "top-right"),      # 90 sends the page's top-left ...
    (180, [0, 0, 10, 20], "bottom-right"),  # 180 sends it to the far corner
    (270, [0, 0, 10, 20], "bottom-left"),   # 270 is the mirror of 90
])
def test_page_top_left_lands_in_the_right_corner(rotation, corner, expected_quadrant):
    w, h = 1200, 800
    x0, y0, x1, y1 = rotate_rect(corner, rotation, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    quadrant = ("bottom-" if cy > h / 2 else "top-") + ("right" if cx > w / 2 else "left")
    assert quadrant == expected_quadrant


def test_rotation_90_is_invertible_by_270():
    """Applying 270 with the dims swapped back recovers the original rect."""
    original = [100.0, 157.0, 126.0, 212.0]
    rotated = rotate_rect(original, 90, 1200, 800)
    assert rotate_rect(rotated, 270, 800, 1200) == original


# ------------------------------------------------------- normalize_boxes ----


def test_normalize_rotated_page_puts_every_rect_in_bounds():
    """Synthetic rotation=90 page: nothing may be left outside the image."""
    w, h = 1200, 800                      # so the unrotated page is 800 x 1200
    boxes = {
        "w": w, "h": h, "zoom": 2.0, "rotation": 90,
        # y beyond 800 is legal in unrotated space (the page is 1200 tall) and is
        # exactly what used to escape the image.
        "lines": [{"bbox": [100, 157, 126, 212], "text": "a"},
                  {"bbox": [50, 900, 300, 1150], "text": "b"},
                  {"bbox": [0, 0, 800, 1200], "text": "whole page"}],
        "tables": [{"bbox": [10, 20, 900, 500], "rows": 3, "cols": 4}],
        "figures": [{"bbox": [20, 1000, 400, 1190], "kind": "image"}],
    }
    before = [i["bbox"] for i in boxes["lines"] + boxes["figures"]]
    assert sum(1 for b in before if not inside(b, w, h)) == 3   # 3 of 4 escape

    norm = normalize_boxes(boxes)
    for key in ("lines", "tables", "figures"):
        for item in norm[key]:
            assert inside(item["bbox"], w, h), (key, item["bbox"])

    assert norm["lines"][0]["bbox"] == [988.0, 100.0, 1043.0, 126.0]
    assert norm["lines"][2]["bbox"] == [0.0, 0.0, 1200.0, 800.0]
    assert norm["lines"][0]["text"] == "a"          # payload survives


def test_normalize_leaves_table_rects_alone():
    """find_tables() already reports display coords -- transforming them breaks them."""
    boxes = {"w": 1200, "h": 800, "zoom": 2.0, "rotation": 90,
             "lines": [], "figures": [],
             "tables": [{"bbox": [10, 20, 900, 500], "rows": 3, "cols": 4}]}
    norm = normalize_boxes(boxes)
    assert norm["tables"][0]["bbox"] == [10, 20, 900, 500]
    assert norm["tables"][0]["rows"] == 3


def test_normalize_is_identity_for_unrotated_pages():
    boxes = line_page(["alpha beta", "gamma delta"], rotation=0)
    assert normalize_boxes(boxes) is boxes


def test_normalize_does_not_mutate_the_input():
    boxes = {"w": 1200, "h": 800, "zoom": 2.0, "rotation": 90,
             "lines": [{"bbox": [100, 157, 126, 212], "text": "a"}],
             "tables": [], "figures": []}
    normalize_boxes(boxes)
    assert boxes["lines"][0]["bbox"] == [100, 157, 126, 212]
    assert boxes["rotation"] == 90


def test_normalize_is_idempotent():
    """Rotation is zeroed on the way out, so a second pass changes nothing."""
    boxes = {"w": 1200, "h": 800, "zoom": 2.0, "rotation": 270,
             "lines": [{"bbox": [100, 157, 126, 212], "text": "a"}],
             "tables": [], "figures": []}
    once = normalize_boxes(boxes)
    assert once["rotation"] == 0 and once["rotation_applied"] == 270
    assert normalize_boxes(once)["lines"] == once["lines"]


def test_normalize_tolerates_missing_dims_and_none():
    assert normalize_boxes(None) is None
    degenerate = {"rotation": 90, "lines": [{"bbox": [1, 2, 3, 4], "text": "x"}]}
    assert normalize_boxes(degenerate) is degenerate      # no dims -> can't place


def test_load_page_assets_normalizes_the_sidecar(tmp_path):
    """The loader is the choke point: callers never see unrotated coordinates."""
    w, h = 1200, 800
    boxes = {"w": w, "h": h, "zoom": 2.0, "rotation": 90,
             "lines": [{"bbox": [100, 157, 126, 212], "text": "alpha beta"},
                       {"bbox": [50, 900, 300, 1150], "text": "gamma delta"}],
             "tables": [], "figures": [{"bbox": [20, 1000, 400, 1190], "kind": "image"}]}
    write_cache(tmp_path, boxes)

    loaded = load_page_assets("doc", 7, cache_dir=tmp_path)["boxes"]
    assert loaded["rotation"] == 0
    for key in ("lines", "figures"):
        for item in loaded[key]:
            assert inside(item["bbox"], w, h), (key, item["bbox"])
    assert loaded["lines"][0]["bbox"] == [988.0, 100.0, 1043.0, 126.0]


def test_grounding_on_a_rotated_page_returns_in_bounds_rects(tmp_path):
    """End to end: a chunk grounded on a rotated page is drawable on the JPEG."""
    w, h = 1200, 800
    boxes = {"w": w, "h": h, "zoom": 2.0, "rotation": 90, "tables": [], "figures": [],
             "lines": [{"bbox": [100, 157, 400, 212], "text": "results monitoring and reporting"},
                       {"bbox": [100, 900, 400, 1150], "text": "beneficiaries per usd investment"}]}
    write_cache(tmp_path, boxes)

    g = ground_chunk({"doc_id": "doc", "page": 7,
                      "text": "results monitoring and reporting for beneficiaries "
                              "per usd investment across the programme"},
                     cache_dir=tmp_path)
    assert g.kind == "lines" and len(g.rects) == 2
    for r in g.rects:
        assert inside(r, w, h), r


# ==================================================== single-token matching ==


def test_solo_number_lines_are_not_matched(tmp_path):
    """The reported bug: 'Annex 5' must not highlight a page number and a bullet."""
    boxes = line_page(["5", "5.", "Annex 5 heading text"])
    write_cache(tmp_path, boxes)

    g = ground_chunk({"doc_id": "doc", "page": 7, "text": "Annex 5"}, cache_dir=tmp_path)

    solo_rects = [boxes["lines"][0]["bbox"], boxes["lines"][1]["bbox"]]
    assert all(r not in g.rects for r in solo_rects)
    # Nothing legitimate matched either, so this degrades to a page citation
    # rather than the old confidence 1.0 line highlight.
    assert g.kind == "page"
    assert g.rects == []
    assert g.confidence < MIN_CONFIDENCE


def test_real_two_token_heading_still_matches(tmp_path):
    """A 2-token line fully covered by the chunk is a legitimate match."""
    boxes = line_page(["5", "5.", "Annex 5", "Annex 7"])
    write_cache(tmp_path, boxes)

    g = ground_chunk({"doc_id": "doc", "page": 7, "text": "Annex 5"}, cache_dir=tmp_path)

    assert g.kind == "lines"
    assert g.rects == [boxes["lines"][2]["bbox"]]     # only the real heading
    assert g.confidence >= MIN_CONFIDENCE


def test_partially_covered_two_token_line_is_rejected(tmp_path):
    """'Annex 7' shares one token with 'Annex 5' -- one hit is not enough."""
    boxes = line_page(["Annex 7"])
    write_cache(tmp_path, boxes)
    g = ground_chunk({"doc_id": "doc", "page": 7, "text": "Annex 5"}, cache_dir=tmp_path)
    assert g.kind == "page" and g.rects == []


def test_short_chunk_without_an_exact_line_stays_page_level(tmp_path):
    """A short chunk needs one completely covered line, not just a good fraction."""
    boxes = line_page(["the quick brown fox jumps"])
    write_cache(tmp_path, boxes)

    g = ground_chunk({"doc_id": "doc", "page": 7, "text": "quick brown fox"},
                     cache_dir=tmp_path)
    # 3 of the line's 5 tokens hit = 0.6, which clears MIN_LINE_COVER, and raw
    # confidence would be 3/3 = 1.0 -- but the chunk is short and no line is
    # fully covered, so the citation degrades instead of highlighting.
    assert g.kind == "page" and g.rects == []


def test_long_chunk_keeps_partial_line_matches(tmp_path):
    """The short-chunk gate must not touch ordinary prose chunks."""
    boxes = line_page(["alpha beta gamma delta epsilon"])
    write_cache(tmp_path, boxes)

    g = ground_chunk(
        {"doc_id": "doc", "page": 7,
         "text": "alpha beta gamma zulu yankee xray whiskey victor tango"},
        cache_dir=tmp_path)
    # 3 of 5 line tokens hit (0.6) on a 9-token chunk -> confidence 0.333.
    assert g.kind == "lines"
    assert g.rects == [boxes["lines"][0]["bbox"]]
    assert g.confidence == pytest.approx(0.333, abs=0.001)


def test_single_token_lines_cannot_reach_min_confidence(tmp_path):
    """The stated constraint, taken to its worst case.

    A page made entirely of solo digits and a chunk full of those digits: under
    the old fraction-only rule every line matched and confidence pinned at 1.0.
    """
    boxes = line_page([str(n) for n in range(1, 12)])
    write_cache(tmp_path, boxes)

    g = ground_chunk({"doc_id": "doc", "page": 7,
                      "text": "1 2 3 4 5 6 7 8 9 10 11"}, cache_dir=tmp_path)
    assert g.rects == []
    assert g.confidence < MIN_CONFIDENCE


def test_ordinary_prose_grounding_is_unchanged(tmp_path):
    """Regression guard: the normal multi-token case still highlights."""
    boxes = line_page([
        "the board approved the funding proposal for the project",
        "unrelated administrative boilerplate about meeting logistics",
    ])
    write_cache(tmp_path, boxes)

    g = ground_chunk(
        {"doc_id": "doc", "page": 7,
         "text": "The Board approved the funding proposal for the project "
                 "at its nineteenth meeting."},
        cache_dir=tmp_path)
    assert g.kind == "lines"
    assert g.rects == [boxes["lines"][0]["bbox"]]
