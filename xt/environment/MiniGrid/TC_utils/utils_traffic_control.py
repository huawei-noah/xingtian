# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np
import random
import copy

def init_agent_two_id_onehot(n_agents):
    total_agent_two_id_onehot_route_list = []
    total_two_id_onehot_route_list = []
    base_agent_id_onehot_route = np.zeros(n_agents)
    base_two_id_onehot_route = np.zeros(2)
    for agent_i in range(n_agents):
        for two_route_i in range(2):
            agent_id_onehot_route = base_agent_id_onehot_route.copy()
            two_id_onehot_route = base_two_id_onehot_route.copy()
            agent_id_onehot_route[agent_i] = 1
            two_id_onehot_route[two_route_i] = 1
            final_agent_two_id_onehot_route = np.concatenate((agent_id_onehot_route, two_id_onehot_route), axis=0)
            total_agent_two_id_onehot_route_list.append(final_agent_two_id_onehot_route)
            total_two_id_onehot_route_list.append(two_id_onehot_route)
    return total_agent_two_id_onehot_route_list, total_two_id_onehot_route_list

def init_easy(straight_area, world_size):

    world_numpy = np.arange(world_size*world_size).reshape(world_size, world_size)

    xy_id_route_0_A = world_numpy[:(straight_area+1+1), straight_area]
    xy_id_route_0_B = world_numpy[straight_area+1, (world_size-straight_area-1):]
    xy_id_route_0_CD = world_numpy[:, straight_area]

    xy_id_route_1_A = world_numpy[straight_area+1, :(straight_area+1+1)]
    xy_id_route_1_B = world_numpy[:(straight_area+1), straight_area+1][::-1]
    xy_id_route_1_CD = world_numpy[straight_area+1, :]

    xy_id_route_2_A = world_numpy[(world_size-straight_area-1-1):, straight_area+1][::-1]
    xy_id_route_2_B = world_numpy[straight_area, :(straight_area+1)][::-1]
    xy_id_route_2_CD = world_numpy[:, straight_area+1][::-1]

    xy_id_route_3_A = world_numpy[straight_area, (world_size-straight_area-1-1):][::-1]
    xy_id_route_3_B = world_numpy[(world_size-straight_area-1):, straight_area]
    xy_id_route_3_CD = world_numpy[straight_area, :][::-1]

    xy_id_route_0_AB = np.concatenate((xy_id_route_0_A, xy_id_route_0_B), axis=-1)
    xy_id_route_1_AB = np.concatenate((xy_id_route_1_A, xy_id_route_1_B), axis=-1)
    xy_id_route_2_AB = np.concatenate((xy_id_route_2_A, xy_id_route_2_B), axis=-1)
    xy_id_route_3_AB = np.concatenate((xy_id_route_3_A, xy_id_route_3_B), axis=-1)

    total_xy_id_route_list = [xy_id_route_0_AB, xy_id_route_0_CD, xy_id_route_1_AB, xy_id_route_1_CD,
                              xy_id_route_2_AB, xy_id_route_2_CD, xy_id_route_3_AB, xy_id_route_3_CD]
    total_len_route_list = [len(xy_id_route_0_AB), len(xy_id_route_0_CD), len(xy_id_route_1_AB), len(xy_id_route_1_CD),
                            len(xy_id_route_2_AB), len(xy_id_route_2_CD), len(xy_id_route_3_AB), len(xy_id_route_3_CD)]
    
    return total_xy_id_route_list, total_len_route_list

def init_moderate(straight_area, world_size):

    world_numpy = np.arange(world_size*world_size).reshape(world_size, world_size)

    xy_id_route_00_A = world_numpy[:(straight_area+2+2), straight_area+1]
    xy_id_route_00_B = world_numpy[straight_area+2+1, (world_size-straight_area-2):]
    xy_id_route_00_CD = world_numpy[:, straight_area+1]
    xy_id_route_01_A = world_numpy[:(straight_area+2+1), straight_area]
    xy_id_route_01_B = world_numpy[straight_area+2, (world_size-straight_area-2-1):]
    xy_id_route_01_CD = world_numpy[:, straight_area]

    xy_id_route_10_A = world_numpy[straight_area+2, :(straight_area+2+2)]
    xy_id_route_10_B = world_numpy[:(straight_area+2), straight_area+2+1][::-1]
    xy_id_route_10_CD = world_numpy[straight_area+2, :]
    xy_id_route_11_A = world_numpy[straight_area+2+1, :(straight_area+2+1)]
    xy_id_route_11_B = world_numpy[:(straight_area+2+1), straight_area+2][::-1]
    xy_id_route_11_CD = world_numpy[straight_area+2+1, :]

    xy_id_route_20_A = world_numpy[(world_size-straight_area-2-2):, straight_area+2][::-1]
    xy_id_route_20_B = world_numpy[straight_area, :(straight_area+2)][::-1]
    xy_id_route_20_CD = world_numpy[:, straight_area+2][::-1]
    xy_id_route_21_A = world_numpy[(world_size-straight_area-2-1):, straight_area+2+1][::-1]
    xy_id_route_21_B = world_numpy[straight_area+1, :(straight_area+2+1)][::-1]
    xy_id_route_21_CD = world_numpy[:, straight_area+2+1][::-1]

    xy_id_route_30_A = world_numpy[straight_area+1, (world_size-straight_area-2-2):][::-1]
    xy_id_route_30_B = world_numpy[(world_size-straight_area-2):, straight_area]
    xy_id_route_30_CD = world_numpy[straight_area+1, :][::-1]
    xy_id_route_31_A = world_numpy[straight_area, (world_size-straight_area-2-1):][::-1]
    xy_id_route_31_B = world_numpy[(world_size-straight_area-2-1):, straight_area+1]
    xy_id_route_31_CD = world_numpy[straight_area, :][::-1]

    xy_id_route_00_AB = np.concatenate((xy_id_route_00_A, xy_id_route_00_B), axis=-1)
    xy_id_route_01_AB = np.concatenate((xy_id_route_01_A, xy_id_route_01_B), axis=-1)
    xy_id_route_10_AB = np.concatenate((xy_id_route_10_A, xy_id_route_10_B), axis=-1)
    xy_id_route_11_AB = np.concatenate((xy_id_route_11_A, xy_id_route_11_B), axis=-1)
    xy_id_route_20_AB = np.concatenate((xy_id_route_20_A, xy_id_route_20_B), axis=-1)
    xy_id_route_21_AB = np.concatenate((xy_id_route_21_A, xy_id_route_21_B), axis=-1)
    xy_id_route_30_AB = np.concatenate((xy_id_route_30_A, xy_id_route_30_B), axis=-1)
    xy_id_route_31_AB = np.concatenate((xy_id_route_31_A, xy_id_route_31_B), axis=-1)

    total_xy_id_route_list = [xy_id_route_00_AB, xy_id_route_00_CD, xy_id_route_01_AB, xy_id_route_01_CD,
                              xy_id_route_10_AB, xy_id_route_10_CD, xy_id_route_11_AB, xy_id_route_11_CD,
                              xy_id_route_20_AB, xy_id_route_20_CD, xy_id_route_21_AB, xy_id_route_21_CD,
                              xy_id_route_30_AB, xy_id_route_30_CD, xy_id_route_31_AB, xy_id_route_31_CD]
    total_len_route_list = [len(xy_id_route_00_AB), len(xy_id_route_00_CD), len(xy_id_route_01_AB), len(xy_id_route_01_CD),
                            len(xy_id_route_10_AB), len(xy_id_route_10_CD), len(xy_id_route_11_AB), len(xy_id_route_11_CD),
                            len(xy_id_route_20_AB), len(xy_id_route_20_CD), len(xy_id_route_21_AB), len(xy_id_route_21_CD),
                            len(xy_id_route_30_AB), len(xy_id_route_30_CD), len(xy_id_route_31_AB), len(xy_id_route_31_CD)]

    return total_xy_id_route_list, total_len_route_list

def init_complex(straight_area, world_size):

    world_numpy = np.arange(world_size*world_size).reshape(world_size, world_size)

    xy_id_route_00_A = world_numpy[:(straight_area+2+2), straight_area+1]
    xy_id_route_00_B = world_numpy[straight_area+2+1, (world_size-straight_area-2):]
    xy_id_route_00_CD = world_numpy[:, straight_area+1]
    xy_id_route_01_A = world_numpy[:(straight_area+2+1), straight_area]
    xy_id_route_01_B = world_numpy[straight_area+2, (world_size-straight_area-2-1):]
    xy_id_route_01_CD = world_numpy[:, straight_area]

    xy_id_route_10_A = world_numpy[straight_area+2, :(straight_area+2+2)]
    xy_id_route_10_B = world_numpy[:(straight_area+2), straight_area+2+1][::-1]
    xy_id_route_10_CD = world_numpy[straight_area+2, :]
    xy_id_route_11_A = world_numpy[straight_area+2+1, :(straight_area+2+1)]
    xy_id_route_11_B = world_numpy[:(straight_area+2+1), straight_area+2][::-1]
    xy_id_route_11_CD = world_numpy[straight_area+2+1, :]

    xy_id_route_20_A = world_numpy[(world_size-straight_area-2-2):, straight_area+2][::-1]
    xy_id_route_20_B = world_numpy[straight_area, :(straight_area+2)][::-1]
    xy_id_route_20_CD = world_numpy[:, straight_area+2][::-1]
    xy_id_route_21_A = world_numpy[(world_size-straight_area-2-1):, straight_area+2+1][::-1]
    xy_id_route_21_B = world_numpy[straight_area+1, :(straight_area+2+1)][::-1]
    xy_id_route_21_CD = world_numpy[:, straight_area+2+1][::-1]

    xy_id_route_30_A = world_numpy[straight_area+1, (world_size-straight_area-2-2):][::-1]
    xy_id_route_30_B = world_numpy[(world_size-straight_area-2):, straight_area]
    xy_id_route_30_CD = world_numpy[straight_area+1, :][::-1]
    xy_id_route_31_A = world_numpy[straight_area, (world_size-straight_area-2-1):][::-1]
    xy_id_route_31_B = world_numpy[(world_size-straight_area-2-1):, straight_area+1]
    xy_id_route_31_CD = world_numpy[straight_area, :][::-1]

    xy_id_route_00_AB = np.concatenate((xy_id_route_00_A, xy_id_route_00_B), axis=-1)
    xy_id_route_01_AB = np.concatenate((xy_id_route_01_A, xy_id_route_01_B), axis=-1)
    xy_id_route_10_AB = np.concatenate((xy_id_route_10_A, xy_id_route_10_B), axis=-1)
    xy_id_route_11_AB = np.concatenate((xy_id_route_11_A, xy_id_route_11_B), axis=-1)
    xy_id_route_20_AB = np.concatenate((xy_id_route_20_A, xy_id_route_20_B), axis=-1)
    xy_id_route_21_AB = np.concatenate((xy_id_route_21_A, xy_id_route_21_B), axis=-1)
    xy_id_route_30_AB = np.concatenate((xy_id_route_30_A, xy_id_route_30_B), axis=-1)
    xy_id_route_31_AB = np.concatenate((xy_id_route_31_A, xy_id_route_31_B), axis=-1)

    total_xy_id_route_list = [xy_id_route_00_AB, xy_id_route_00_CD, xy_id_route_01_AB, xy_id_route_01_CD, 
                              xy_id_route_10_AB, xy_id_route_10_CD, xy_id_route_11_AB, xy_id_route_11_CD,
                              xy_id_route_20_AB, xy_id_route_20_CD, xy_id_route_21_AB, xy_id_route_21_CD,
                              xy_id_route_30_AB, xy_id_route_30_CD, xy_id_route_31_AB, xy_id_route_31_CD]
    total_len_route_list = [len(xy_id_route_00_AB), len(xy_id_route_00_CD), len(xy_id_route_01_AB), len(xy_id_route_01_CD), 
                            len(xy_id_route_10_AB), len(xy_id_route_10_CD), len(xy_id_route_11_AB), len(xy_id_route_11_CD),
                            len(xy_id_route_20_AB), len(xy_id_route_20_CD), len(xy_id_route_21_AB), len(xy_id_route_21_CD),
                            len(xy_id_route_30_AB), len(xy_id_route_30_CD), len(xy_id_route_31_AB), len(xy_id_route_31_CD)]

    return total_xy_id_route_list, total_len_route_list

def get_various_list(total_xy_id_route_list, world_size):

    # [x, y] => [row ,col]
    total_xy_real_route_list = []
    for xy_id_route_i in total_xy_id_route_list:
        xy_real_route_list = []
        for xy_id_route_i_j in xy_id_route_i:
            x_real = xy_id_route_i_j//world_size
            y_real = xy_id_route_i_j%world_size
            xy_real_route_list.append([x_real, y_real])
        total_xy_real_route_list.append(xy_real_route_list)
    
    total_xy_scalar_route_list = []
    for xy_real_route_i in total_xy_real_route_list:
        xy_scalar_route_list = []
        for xy_real_route_i_j in xy_real_route_i:
            xy_scalar_route_list.append([xy_real_route_i_j[0]/(world_size-1), xy_real_route_i_j[1]/(world_size-1)])
        total_xy_scalar_route_list.append(xy_scalar_route_list)

    total_xy_onehot_route_list = []
    for xy_real_route_i in total_xy_real_route_list:
        xy_onehot_route_list = []
        for xy_real_route_i_j in xy_real_route_i:
            x_onehot = np.zeros(world_size)
            y_onehot = np.zeros(world_size)
            x_onehot[xy_real_route_i_j[0]] = 1
            y_onehot[xy_real_route_i_j[1]] = 1
            xy_onehot = np.concatenate((x_onehot, y_onehot), axis=0)
            xy_onehot_route_list.append(xy_onehot)
        total_xy_onehot_route_list.append(xy_onehot_route_list)

    return total_xy_real_route_list, total_xy_scalar_route_list, total_xy_onehot_route_list

def init_pos(map, init_range, n_agents, actual_total_xy_real_route_list, world, straight_area):

    total_cur_id_list = [-1]*n_agents

    assert init_range[1] <= straight_area

    if map == 'easy' or map == 'moderate':
        possible_pos = set(range(init_range[0], init_range[1]))
        for agent_i in range(n_agents):
            init_pos_id = random.sample(possible_pos, 1)[0]
            xy_real = actual_total_xy_real_route_list[agent_i][init_pos_id]
            world[agent_i, xy_real[0], xy_real[1]] = 1
            total_cur_id_list[agent_i] = init_pos_id
    elif map == 'complex':
        even = (np.arange(n_agents/2).astype(int)*2)
        odd = even+1
        even = list(even)
        odd = list(odd)
        possible_pos = [set(range(init_range[0], init_range[1])) for _ in range(int(n_agents/2))]
        for possible_pos_i, agent_i in enumerate(even):
            init_pos_id = random.sample(possible_pos[possible_pos_i], 1)
            possible_pos[possible_pos_i] -= set(init_pos_id)
            init_pos_id = init_pos_id[0]
            xy_real = actual_total_xy_real_route_list[agent_i][init_pos_id]
            world[agent_i, xy_real[0], xy_real[1]] = 1
            total_cur_id_list[agent_i] = init_pos_id
        
        for possible_pos_i, agent_i in enumerate(odd):
            init_pos_id = random.sample(possible_pos[possible_pos_i], 1)[0]
            xy_real = actual_total_xy_real_route_list[agent_i][init_pos_id]
            world[agent_i, xy_real[0], xy_real[1]] = 1
            total_cur_id_list[agent_i] = init_pos_id

    return total_cur_id_list, world

def get_world_plot_wall(world_size, straight_area, map):
    world_plot_wall = np.ones((world_size, world_size))
    if map == 'easy':
        world_plot_wall[straight_area, :] = 0
        world_plot_wall[straight_area+1, :] = 0
        world_plot_wall[:, straight_area] = 0
        world_plot_wall[:, straight_area+1] = 0
    elif map == 'moderate' or map == 'complex':
        world_plot_wall[straight_area, :] = 0
        world_plot_wall[straight_area+1, :] = 0
        world_plot_wall[straight_area+2, :] = 0
        world_plot_wall[straight_area+3, :] = 0

        world_plot_wall[:, straight_area] = 0
        world_plot_wall[:, straight_area+1] = 0
        world_plot_wall[:, straight_area+2] = 0
        world_plot_wall[:, straight_area+3] = 0

    return world_plot_wall
