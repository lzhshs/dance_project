import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from g1_dance.export_direct_mapping import save_direct_reference


def test_save_direct_reference_writes_expected_npz_fields(tmp_path):
    qpos = np.arange(12, dtype=np.float64).reshape(2, 6)
    out = tmp_path / "direct.npz"

    save_direct_reference(out, qpos, fps=60.0, source="motion.pkl")

    data = np.load(out, allow_pickle=True)
    np.testing.assert_allclose(data["qpos_ref"], qpos.astype(np.float32))
    np.testing.assert_allclose(data["fps"], np.array([60.0], dtype=np.float32))
    assert data["source"][0] == "motion.pkl"
    assert data["mapping"][0] == "direct_dof"
