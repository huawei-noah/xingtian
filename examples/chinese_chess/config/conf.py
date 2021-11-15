import os

currentpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_BASEDIR = os.path.join(currentpath)
YUNDAO_DIR_PATH = ""


class ResourceConfig:
    python_executor = 'python'
    # the number of games in each block
    block_min_games = 5000
    # the max block number
    train_max_block = 100
    # the min-number of block for training
    train_min_block = 3
    num_process = 10
    restore_path = None
    distributed_datadir = os.path.join(PROJECT_BASEDIR, 'data/distributed')
    nash_battle_local_dir = os.path.join(PROJECT_BASEDIR, 'data/nash_battle')
    history_selfplay_dir = os.path.join(PROJECT_BASEDIR, 'data/history_selfplays')
    model_dir = os.path.join(PROJECT_BASEDIR, 'data/models')
    validate_dir = os.path.join(PROJECT_BASEDIR, 'data/validate')
    tensorboard_dir = os.path.join(PROJECT_BASEDIR, 'data/tensorboard')
    # yundao
    new_data_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'self_play')
    validate_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'validate')
    pool_weights_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'pool_weights')
    nash_weights_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'nash_weights')
    nash_battle_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'nash_battle')
    nash_battle_yundao_share_dir = os.path.join(YUNDAO_DIR_PATH, 'nash_battle_share')
    tensorboard_yundao_dir = os.path.join(YUNDAO_DIR_PATH, 'tensorboard')
    nash_battle_list_json = 'nash_battle_list.json'
    nash_res_json = 'nash_res.json'
    nash_res_bot_json = 'nash_res_bot.json'
    model_pool_list_json = 'model_pool_list.json'
    chosen_model_json = 'chosen_model.json'
    sub_validate_json = 'sub_validate_res.json'
    eliminated_model_json = 'eliminated_model.json'


class TrainingConfig:
    network_filters = 192
    network_layers = 10
    batch_size = 2048
    sample_games = 500
    c_puct = 1.5
    # Saver model after every 1k steps
    saver_step = 400
    # lr = [
    #     (0, 0.03),
    #     (10000, 0.01),
    #     (20000, 0.003),
    #     (30000, 0.001),
    #     (40000, 0.0003),
    #     (50000, 0.0001),
    # ]
    lr = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    max_model_num = 10
    train_with_bot = True


class SelfPlayConfig:
    non_cap_draw_round = 120
    train_playout = 800
    train_temp_round = 3000
    gpu_num = 8
    num_proc_each_gpu = 4
    self_play_download_weight_dt = 30
    self_play_upload_data_dt = 60
    self_play_games_one_time = 1
    cpu_proc_num = 8
    yigou_cpu_proc_num = 16
    resign_score = -0.95
    game_num_to_restart = 100
    py_env = False


class EvaluateConfig:
    cchess_playout = 400
    val_playout = 800
    val_temp_round = 12
    model_pool_size = 20
    nash_each_battle_num = 64
    bot_name = 'bot'
    icy_player_name = 'icy'
    nash_eva_waiting_dt = 60
    gpu_num = 8
    num_proc_each_gpu = 4
    nash_nodes_num = 2
