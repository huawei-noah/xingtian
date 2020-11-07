import os
from zeus.common.util.evaluate_xt import fetch_train_event
from zeus.common.util.evaluate_xt import read_train_records
from zeus.common.util.evaluate_xt import DEFAULT_ARCHIVE_DIR


def test_fetch_train_path():
    archive = os.path.join(
        os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
    )
    _id = "xt_cartpole_0204"
    r = fetch_train_event(archive, _id)
    print("get", r)


def test_fetch_train_path_single():
    archive = os.path.join(
        os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
    )
    _id = "xt_cartpole_0204"
    r = fetch_train_event(archive, _id, True)
    print("get", r)


def test_read_train_record():
    bm_args = {
        "archive_root": os.path.join(
            os.path.expanduser("~"), "projects/rl/{}".format(DEFAULT_ARCHIVE_DIR)
        ),
        "bm_id": "xt_cartpole_0204",
    }
    # import numpy as np
    d = read_train_records(bm_args)
    # d = np.array(d)
    print("\n", d)
