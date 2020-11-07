import sys

from zeus.common.util.get_xt_config import *


def test_main():
    """
    test parse config file
    """
    config_file = sys.argv[1]
    ret_para = parse_xt_multi_case_paras(config_file)
    for para in ret_para:
        print(para)


def test_find_items():
    """
    test find key in dictionary
    """
    config_file = sys.argv[1]
    with open(config_file) as file_hander:
        yaml_obj = yaml.safe_load(file_hander)

    ret_obj = finditem(yaml_obj, "agent_config")
    print(ret_obj)


if __name__ == "__main__":
    test_main()
    # test_find_items()
