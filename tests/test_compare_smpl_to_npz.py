import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from compare_smpl_to_npz import output_path_for_comparison


def test_output_path_for_comparison_uses_motion_and_reference_stems():
    root = os.path.dirname(os.path.dirname(__file__))
    expected = os.path.join(root, "videos", "dance_vs_dance_direct_compare.mp4")

    assert output_path_for_comparison("aist/dance.pkl", "optimized/dance_direct.npz") == expected
