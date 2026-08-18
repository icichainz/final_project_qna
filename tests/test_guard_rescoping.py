"""Guards on the decomposer's doc tags, and the blocking work at login.

The scenarios are the observed failures, not invented ones: a comparison
whose history half lost its scope and was answered from unrelated documents,
and a message naming two FPs answered entirely from a third.
"""
import asyncio
import inspect
import threading

from gcf_qna.app import accounts, chainlit_app
from gcf_qna.app.chainlit_app import _cited_docs, _rescope_items
from gcf_qna.rag.retrieve import _doc_match

FP214 = "61_gcf-b37-02-add05-funding-proposal-package-fp214"
FP274 = "02_gcf-b42-02-add16-funding-proposal-package-fp274"
FP203 = "72_GCF_B.35_02_Add.05_Funding_proposal_package_for_FP203"
FP242 = "34_b39-02-add17-funding-proposal-package-fp242"


def test_fabricated_tag_is_rewritten_not_dropped():
    """'02_fp214' matches no document; the FP number in it is salvageable."""
    items = _rescope_items([{"q": "total financing", "doc": "02_fp214"}],
                           "What is FP214's total financing?", [FP274])
    assert items[0]["doc"] == "fp214"
    assert items[0]["q"] == "total financing"      # still scoped, nothing to add
    assert _doc_match(FP214, items[0]["doc"])      # the plain token resolves
    assert not _doc_match(FP274, items[0]["doc"])


def test_comparison_keeps_both_sides_scoped():
    """Turn 2 of the verified failure: 'it' is FP274 from the previous turn.

    Blanket-nulling left two identical bare 'total financing' queries with no
    identifier for two-stage routing, so the FP274 side drew global noise.
    """
    items = _rescope_items(
        [{"q": "total financing", "doc": FP214},
         {"q": "total financing", "doc": FP274}],
        "How does FP214's financing compare to it?", [FP274])
    assert items[0]["doc"] == "fp214"              # named by the message
    assert items[1]["doc"] == FP274                # the 'it', cited in history
    assert [i["q"] for i in items] == ["total financing", "total financing"]


def test_history_doc_cannot_outvote_explicit_ids():
    """'FP214 ... FP265?' must not be answered from the previous turn's FP274:
    two ids, two sub-queries, no slot left for a history document."""
    items = _rescope_items(
        [{"q": "total financing", "doc": FP274},
         {"q": "climate rationale", "doc": FP274}],
        "Compare FP214 and FP265 please?", [FP274])
    assert [i["doc"] for i in items] == [None, None]
    assert items[0]["q"] == "total financing FP214 FP265"
    assert items[1]["q"] == "climate rationale FP214 FP265"


def test_unrelated_tag_stripped_and_query_rescoped():
    """A tag tied to neither the message nor history still goes — but the
    sub-query inherits the message's identifier instead of running bare."""
    items = _rescope_items(
        [{"q": "total financing", "doc": "13_gcf-b41-02-add03-fp250"}],
        "What is FP214's total financing?", [FP274])
    assert items[0]["doc"] is None
    assert items[0]["q"] == "total financing FP214"


def test_single_fp_message_still_routes():
    items = _rescope_items([{"q": "accredited entity of FP203", "doc": FP203}],
                           "Who is the accredited entity for FP203?", [])
    assert items[0]["doc"] == "fp203"
    assert _doc_match(FP203, items[0]["doc"])
    assert items[0]["q"] == "accredited entity of FP203"   # already carries it


def test_fanout_without_message_ids_keeps_history_tags():
    """'compare those two' — the case the whole fan-out exists for. Tags stay;
    a mangled one is repaired from the ids actually cited."""
    items = _rescope_items(
        [{"q": "total financing", "doc": "02_fp274"},
         {"q": "total financing", "doc": FP214}],
        "Compare those two on financing.", [FP274, FP214])
    assert [i["doc"] for i in items] == [FP274, FP214]


def test_lone_query_drops_a_tag_the_message_never_names():
    items = _rescope_items([{"q": "adaptation approach", "doc": FP274}],
                           "How do the proposals handle adaptation?", [FP274])
    assert items[0]["doc"] is None
    assert items[0]["q"] == "adaptation approach"   # no identifier to add


def test_untagged_queries_are_left_alone():
    items = _rescope_items([{"q": "adaptation finance in FP214", "doc": None},
                            {"q": "typical adaptation projects", "doc": None}],
                           "How does FP214 compare to typical projects?", [])
    assert [i["doc"] for i in items] == [None, None]
    assert items[1]["q"] == "typical adaptation projects"


def test_cited_docs_covers_every_corpus_id_shape():
    """The narrow r'\\d{1,3}_gcf-' pattern missed 3 of the 273 corpus ids."""
    hist = [{"role": "assistant",
             "content": f"see [{FP203}, p. 4], [{FP242}] and [{FP214}, p. 9]"},
            {"role": "user", "content": "and the others?"}]
    assert _cited_docs(hist) == [FP203, FP242, FP214]


# -- login must not block the event loop -------------------------------------

def test_auth_callback_offloads_the_scrypt_verify(monkeypatch):
    seen = {}

    def fake_check(username, password):
        seen["worker"] = threading.current_thread()
        return True

    monkeypatch.setattr(accounts, "parse_env_users", lambda: {})
    monkeypatch.setattr(accounts, "check_login", fake_check)

    async def go():
        seen["loop"] = threading.current_thread()
        return await chainlit_app.auth("offload-probe", "hunter2hunter2")

    assert inspect.iscoroutinefunction(chainlit_app.auth)
    user = asyncio.run(go())
    assert user is not None and user.identifier == "offload-probe"
    assert seen["worker"] is not seen["loop"]     # ran off the event loop


def test_login_throttle_matches_the_signup_shape():
    name = "throttle-probe"
    assert all(accounts.login_allowed(name) for _ in range(accounts.LOGIN_LIMIT))
    assert not accounts.login_allowed(name)
    assert accounts.login_allowed("someone-else")
    assert accounts.signup_allowed("1.2.3.4")     # separate bucket, untouched
