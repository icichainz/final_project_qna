"""split_pages() must not lose or mis-number pages around empty page bodies.

The extraction pipeline merges pages as f"\\n\\n---\\n**Page {i+1}**\\n---\\n\\n{body}".
When one VLM completion comes back empty, that page contributes a marker with no
body, and the newlines around it collapse into a single run. A page-marker regex
with a greedy trailing "\\n+" eats that whole run -- including the blank line the
*next* marker needs to match -- so the next marker is skipped, its content is
indexed under the empty page's number, and every citation after it points at the
wrong page in the grounded viewer.

Empty pages themselves are intentionally dropped from the output (there is
nothing to index); what must survive is every *non-empty* page, with its own
number and a body free of raw marker text.
"""
from gcf_qna.rag.parse import split_pages

# Byte-for-byte the separator src/gcf_qna/extraction/vlm.py writes when merging.
PAGE_FMT = "\n\n---\n**Page {n}**\n---\n\n{body}"


def merge(bodies):
    """Rebuild a merged extraction file from per-page bodies (1-indexed)."""
    return "".join(
        PAGE_FMT.format(n=i, body=b) for i, b in enumerate(bodies, start=1)
    )


def test_empty_body_between_two_pages():
    """An empty page 2 must not swallow page 3's marker."""
    md = merge(["alpha", "", "gamma"])
    assert split_pages(md) == [(1, "alpha"), (3, "gamma")]


def test_empty_body_as_final_page():
    """A trailing empty page is dropped without disturbing the pages before it."""
    md = merge(["alpha", "beta", ""])
    assert split_pages(md) == [(1, "alpha"), (2, "beta")]


def test_consecutive_empty_bodies():
    """Several empty pages in a row still leave the survivors correctly numbered."""
    md = merge(["alpha", "", "", "", "epsilon"])
    assert split_pages(md) == [(1, "alpha"), (5, "epsilon")]


def test_first_page_empty():
    """An empty page 1 must not consume the page 2 marker."""
    md = merge(["", "beta", "gamma"])
    assert split_pages(md) == [(2, "beta"), (3, "gamma")]


def test_normal_multipage_unchanged():
    """The ordinary, no-empty-page case is untouched by the fix."""
    md = merge(["alpha", "beta", "gamma", "delta"])
    assert split_pages(md) == [
        (1, "alpha"),
        (2, "beta"),
        (3, "gamma"),
        (4, "delta"),
    ]


def test_marker_at_very_start_of_file():
    """Files are .strip()ed before splitting, so page 1's marker sits at index 0."""
    md = merge(["alpha", "beta"]).strip()
    assert md.startswith("---\n**Page 1**")
    assert split_pages(md) == [(1, "alpha"), (2, "beta")]


def test_multiline_bodies_and_blank_lines_preserved():
    """Blank lines *inside* a body survive; only the outer padding is stripped."""
    md = merge(["alpha\n\nstill alpha", "beta"])
    assert split_pages(md) == [(1, "alpha\n\nstill alpha"), (2, "beta")]


def test_no_markers_falls_back_to_page_zero():
    assert split_pages("plain text, no markers") == [(0, "plain text, no markers")]


def test_regression_reviewer_repro():
    """Exact repro from review: pages alpha/''/gamma/delta.

    Under the old greedy `---\\n+` tail this returned
        [(1, 'alpha'), (2, '---\\n**Page 3**\\n---\\n\\ngamma'), (4, 'delta')]
    -- page 3's text indexed as page 2 with the raw marker embedded, page 3 gone.
    """
    md = merge(["alpha", "", "gamma", "delta"])
    pages = split_pages(md)

    assert pages == [(1, "alpha"), (3, "gamma"), (4, "delta")]

    # No page number is ever emitted twice, and none is lost to a neighbour.
    numbers = [n for n, _ in pages]
    assert numbers == sorted(set(numbers))

    # No body carries leaked marker syntax from a marker the regex failed to eat.
    assert all("**Page" not in body for _, body in pages)
