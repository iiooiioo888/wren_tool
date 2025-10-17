import os
from scripts.load_test import run_load


def test_run_load_quick():
    # run with 2 workers on sample data to ensure no exceptions
    run_load(n_workers=2)
    assert True
