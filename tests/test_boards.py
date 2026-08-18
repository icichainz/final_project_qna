from gcf_qna.boards import board_of, year_of


def test_standard_dashed_id():
    assert board_of("61_gcf-b37-02-add05-funding-proposal-package-fp214") == 37
    assert year_of("61_gcf-b37-02-add05-x") == 2023


def test_underscore_dotted_id():          # review finding #4
    assert board_of("72_GCF_B.35_02_Add.05_Funding_proposal_package_for_FP205") == 35
    assert year_of("72_GCF_B.35_02_Add.05_x") == 2023


def test_uppercase_and_mixed():
    assert board_of("24_GCF-B40-02-ADD14-x") == 40
    assert board_of("34_b39-02-x") is None or True   # leading-underscore variant below
    assert board_of("34_b39-02-x") == 39


def test_no_board():
    assert board_of("some_random_doc") is None
    assert year_of("some_random_doc") is None
