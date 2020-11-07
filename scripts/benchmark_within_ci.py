#!/usr/bin/env python
import os
import sys
import yaml
import argparse
import subprocess
import shutil
from datetime import datetime


def get_bm_fix_path(bm_info, key_seq, last_path=None):
    """Get model path of benchmark yaml."""
    _bm_path_seq = [bm_info[_key] for _key in key_seq]
    if last_path:
        _bm_path_seq += [last_path]
    # model_path = os.path.join(bm_info["archive_root"], bm_info["id"], "models")
    target_path = os.path.join(*_bm_path_seq)
    return target_path


def assemble_config_file(config_info, total_steps):
    """Add timestamp into benchmark id."""
    target_info = config_info.copy()
    _bm = config_info["benchmark"]
    target_info["benchmark"].update({"id": "+".join([
        _bm["id"], datetime.now().strftime("%Y%m%d%H%M%S")])})

    if "agent_config" not in target_info["agent_para"]:
        target_info["agent_para"]["agent_config"] = dict()
    target_info["agent_para"]["agent_config"].update({
        'complete_step': int(total_steps),
    })

    return target_info


def main():
    parser = argparse.ArgumentParser(description="Benchmark tools.")

    parser.add_argument(
        "-yp",
        "--yaml_path",
        default="examples/benchmark_cases",
        help="""config file with yaml""",
    )
    parser.add_argument(
        "-c", "--case", default="all", help="task choice to run benchmark.",
    )
    parser.add_argument(
        "-s", "--steps", default=10000000, type=int,
        help="total steps to run benchmark.",
    )

    args, _ = parser.parse_known_args()

    if not os.path.isdir(args.yaml_path):
        print("config dir: {} is not exist".format(args.yaml_path))
        sys.exit(1)

    # files = [fi for fi in files if fi.startswith("bm_")]
    if args.case in ("ALL", "all",):
        files_candi = [
            os.path.join(args.yaml_path, fi)
            for fi in os.listdir(args.yaml_path)
            if fi[0] != "."
        ]
    elif isinstance(args.case, str):
        if "," in args.case:
            case_list = args.case.strip().split(",")
            files_candi = [
                os.path.join(args.yaml_path, "{}.yaml".format(_yaml))
                for _yaml in case_list
            ]
        else:
            files_candi = [os.path.join(args.yaml_path, "{}.yaml".format(args.case))]
    else:
        print("invalid args: {}".format(args))
        sys.exit(1)

    # real files to benchmark
    bm_cases = [_file for _file in files_candi if os.path.exists(_file)]
    print(
        "There are < {} > cases will be benchmark:\n{}".format(len(bm_cases), bm_cases)
    )
    sys.stdout.flush()

    for config_file in bm_cases:
        print("start benchmark case: {}".format(config_file))
        with open(config_file, "r") as yaml_file:
            config_info = yaml.safe_load(yaml_file)

        target_info = assemble_config_file(config_info, args.steps)
        tar_bm_info = target_info["benchmark"]

        _model_path = get_bm_fix_path(tar_bm_info, ["archive_root", "id"], "models")
        print("model_path: {}".format(_model_path))
        if not os.path.exists(_model_path):
            os.makedirs(_model_path)

        to_yaml = os.path.join(
            os.path.dirname(_model_path),
            "target_{}".format(os.path.basename(config_file)))
        with open(to_yaml, "a+") as to_file:
            yaml.dump(target_info, to_file)
        print("run with yaml: {}".format(to_yaml))
        sys.stdout.flush()

        subprocess.call(
            ["xt_main", "-f", "{}".format(to_yaml)],
            # stdout=PIPE,
        )

        os.system("ls -al {} | grep actor* | wc -l".format(_model_path))
        shutil.rmtree(_model_path)
        print(" 'ls -al' after remove model files")
        subprocess.call(["ls", "-al", "{}".format(os.path.dirname(_model_path))])

        print("end benchmark case: {}".format(config_file))

    print("BENCHMARK TEST FINISHED")


if __name__ == "__main__":
    main()
