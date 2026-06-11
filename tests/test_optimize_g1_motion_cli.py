import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optimize_g1_motion import build_parser


def test_parser_accepts_expressive_constraint_profile():
    parser = build_parser()

    args = parser.parse_args(["motion.pkl", "--profile", "expressive"])

    assert args.motion == "motion.pkl"
    assert args.profile == "expressive"


def test_parser_accepts_balance_constraint_profile():
    parser = build_parser()

    args = parser.parse_args(["motion.pkl", "--profile", "balance"])

    assert args.motion == "motion.pkl"
    assert args.profile == "balance"


def test_parser_accepts_showcase_constraint_profile():
    parser = build_parser()

    args = parser.parse_args(["motion.pkl", "--profile", "showcase"])

    assert args.motion == "motion.pkl"
    assert args.profile == "showcase"
