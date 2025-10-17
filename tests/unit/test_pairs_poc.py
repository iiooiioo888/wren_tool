from scripts.pairs_poc import run_pairs_poc
import os


def test_pairs_poc_runs(tmp_path):
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_pair.csv'))
    out_file = tmp_path / 'pairs_out.json'
    res = run_pairs_poc(csv_path, str(out_file))
    assert 'mtm' in res
    assert out_file.exists()
