#!/usr/bin/env python
import glob
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Process
import yaml
from xt.benchmark.tools.get_config import check_if_patch_local_node

MODULE_PATH = os.path.abspath(os.path.join("."))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from xt.benchmark.tools.evaluate_xt import get_bm_args_from_config, read_train_event_id

CI_WORKSPACE = "scripts/ci_tmp_yaml"


def rm_files(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)


def write_conf_file(config_folder, config):
    with open(config_folder, "w") as file:
        yaml.dump(config, file)


def check_sys_argv(argv):
    """fetch ci parameters."""
    if len(argv) != 2:
        print("input argv err")
        exit(1)

    config_file = argv[1]
    with open(config_file) as conf:
        config_list = yaml.load(conf)

    end_tag = config_list.get("end_flag")
    ci_task = config_list.get("task")
    save_steps = config_list.get("model_save_step", 100)
    config_dir = config_list.get("config_dir", "examples/default_cases")
    single_flag = config_list.get("single_case", None)

    print("##################################")
    print("TEST CONFIG FOLDER IS ", config_dir)
    print("STEP FOR EACH TEST IS ", save_steps)
    print("##################################")

    if ci_task == "train":
        node_array = config_list.get("node_config")

    elif ci_task == "eval":
        node_array = config_list.get("test_node_config")

    else:
        node_array = None
        print("invalid test type: {}".format(ci_task))
        exit(1)

    return node_array, end_tag, ci_task, save_steps, config_dir, single_flag


def assemble_ci_config(target_yaml, ci_task, node_list, save_steps):
    with open(target_yaml) as config_file:
        config = yaml.load(config_file)

    alg_config = config["alg_para"].get("alg_config")
    if alg_config is None:
        alg_save_steps = {"alg_config": {"save_model_step": save_steps}}
        config["alg_para"].update(alg_save_steps)
    else:
        config["alg_para"]["alg_config"].setdefault("save_model_step", save_steps)

    if ci_task == "train":
        # check key
        if "node_config" not in config:
            config.update({"node_config": list()})

        for k in config.get("node_config"):
            config["node_config"].pop()
        for i in range(len(node_list)):
            config["node_config"].append(node_list[i])
    elif ci_task == "evaluate":
        # fixme:
        config["test_node_config"].pop()
        config["test_node_config"].append(node_list[0])

    return config


def run_test(tmp_conf, ci_task):
    process = subprocess.Popen(
        ["setsid", "python3", "xt/main.py", "--config_file", tmp_conf, "--task", ci_task],
        # stdout=subprocess.PIPE,
    )
    return process


def check_test(flag, ci_task, model_path, tmp_file):
    if os.path.isdir(model_path) is False:
        previous_length = 0
    else:
        files_model = os.listdir(model_path)
        previous_length = len(files_model)
    start = time.time()

    test_process = run_test(tmp_file, ci_task)
    normal_return_code = (0, -9, -15)

    while True:
        returncode = test_process.poll()
        # print("returncode:", returncode)
        if returncode is not None and returncode not in normal_return_code:
            print("get a err on test", tmp_file)
            if flag:
                exit(1)
            else:
                break

        if ci_task == "train":
            time.sleep(2)
            try:
                file_module = os.listdir(model_path)
                files_num = len(file_module)
            except Exception:
                files_num = 0

            print(files_num, previous_length, tmp_file, model_path)
            if previous_length < files_num:
                if returncode is None:
                    close_test(test_process)
                elif returncode in normal_return_code:
                    rm_files(model_path)
                    break

        elif ci_task == "evaluate":
            end = time.time() - start
            if end > 20:
                if returncode is None:
                    close_test(test_process)
                elif returncode == 0:
                    break
            else:
                print("test failed")
                exit(1)


def close_test(process):
    process.send_signal(signal.SIGINT)
    # process.kill()
    # process.terminate()
    print("sent close signal to work process")
    time.sleep(1)


def parallel_case_check(processes):
    while True:
        exitcodes = []
        for process in processes:
            exitcodes.append(process.exitcode)
            if process.exitcode is not None and process.exitcode != 0:
                return 1

        exitcode_state = True
        for exitcode in exitcodes:
            if exitcode is None:
                exitcode_state = False

        if exitcode_state:
            return 0

        time.sleep(0.1)


def main():
    node_list, end_flag, ci_task, save_steps, conf_dir, sgf = check_sys_argv(sys.argv)

    if not os.path.isdir(CI_WORKSPACE):
        os.makedirs(CI_WORKSPACE)

    _candidates = glob.glob("{}/*.yaml".format(conf_dir))
    target_yaml = [item for item in _candidates if item[0] != "."]
    print("CI start parse yaml: \n", target_yaml)

    if len(target_yaml) < 1:
        print("exit with config folder is empty")
        exit(1)

    # go through all the config files
    for one_yaml in target_yaml:
        # print(end_flag)
        if sgf and one_yaml != sgf:
            continue
        print("processing: {}".format(one_yaml))
        config_tmp = assemble_ci_config(one_yaml, ci_task, node_list, save_steps)
        processes_parallel = []
        # go through all the node in node_config
        for node_n in range(len(node_list)):
            tmp_name = (
                os.path.split(one_yaml)[-1]
                + "_node_"
                + str(len(config_tmp.get("node_config")))
            )
            if node_n != 0:
                config_tmp["node_config"].pop()

            # try environment number in 1 and 2
            for env_n in range(2):
                config_tmp["env_num"] = env_n + 1
                tmp_name += "_e-" + str(config_tmp.get("env_num"))

                # ---------
                bm_id = config_tmp.get("benchmark", dict()).get("id")
                if not bm_id:
                    _str_list = list()
                    _str_list.append(config_tmp.get("agent_para").get("agent_name"))
                    _str_list.append(config_tmp.get("env_para").get("env_name"))
                    _str_list.append(config_tmp.get("env_para").get("env_info").get("name"))
                    bm_id = "+".join(_str_list)
                bm_id = "{}+e{}".format(bm_id, env_n)

                if not config_tmp.get("benchmark"):
                    config_tmp.update({"benchmark": {"id": bm_id}})
                else:
                    config_tmp["benchmark"].update({"id": bm_id})

                tmp_yaml_name = os.path.join(CI_WORKSPACE, tmp_name)

                write_conf_file(tmp_yaml_name, config_tmp)
                from xt.benchmark.tools.evaluate_xt import (
                    get_train_model_path_from_config,
                )

                model_path = get_train_model_path_from_config(config_tmp)
                print("model save path: ", model_path)

                p = Process(
                    target=check_test,
                    args=(end_flag, ci_task, model_path, tmp_yaml_name),
                )
                p.start()
                processes_parallel.append(p)
                time.sleep(0.4)

        end_check = parallel_case_check(processes_parallel)

        time.sleep(1)

        if end_check == 1:
            print("test failed")
            exit(1)

        rm_files(CI_WORKSPACE)

    print("Normal train passed")


if __name__ == "__main__":
    main()
