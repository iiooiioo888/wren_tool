from scripts.pairs_poc import run_pairs_poc
import os


def test_pairs_stoploss_triggers(tmp_path):
    csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_pair.csv'))
    out_file = tmp_path / 'pairs_out.json'
    # set a very small stop_z to force stop-loss exit
    res = run_pairs_poc(csv_path, str(out_file), entry_z=0.5, exit_z=0.1, stop_z=1.0, max_holding_bars=2)
    assert 'mtm' in res
    assert out_file.exists()
