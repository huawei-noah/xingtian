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
import random
import math
import os
import gym
import copy
import numpy as np
from xt.environment.MiniGrid.TC_utils.utils_traffic_control import init_easy, init_moderate, init_complex,  \
                                                init_pos, get_world_plot_wall, init_agent_two_id_onehot, \
                                                get_various_list
from PIL import Image, ImageDraw, ImageFont

class TrafficControlEnv(gym.Env):
    def __init__(self, map, save_frame=False, seed=2021):
        if map == 'easy':
            self.straight_area = 2
            self.init_range = [0,2]
            self.max_steps = 20
            self.agent_color = ['green', 'blue', 'purple', 'darkorange']
        elif map == 'moderate':
            self.straight_area = 2
            self.init_range = [0,2]
            self.max_steps = 40
            self.agent_color = ['green', 'green', 'blue', 'blue', 'purple', 'purple', 'darkorange', 'darkorange']
        elif map == 'complex':
            self.straight_area = 3
            self.init_range = [0,3]
            self.max_steps = 60
            self.agent_color = ['green', 'green', 'green', 'green', 'blue', 'blue', 'blue', 'blue', \
                                'purple', 'purple', 'purple', 'purple', 'darkorange', 'darkorange', 'darkorange', 'darkorange']

        self.map = map
        self.timestep_penalty = -0.02
        self.collision_penalty = -2
        self.exit_bouns = 0.1
        self.font_indicator = ImageFont.truetype('./envs/TC_utils/cmb10.ttf', 25)
        self.n_actions = 2
        
        if self.map == 'easy':
            self.world_size = (self.straight_area + 1) * 2
        elif self.map == 'moderate' or self.map == 'complex':
            self.world_size = (self.straight_area + 2) * 2

        if self.map == 'easy':
            self.n_agents = 4
            self.total_xy_id_route_list, self.total_len_route_list = init_easy(self.straight_area, self.world_size)
        elif self.map == 'moderate':
            self.n_agents = 8
            self.total_xy_id_route_list, self.total_len_route_list = init_moderate(self.straight_area, self.world_size)
        elif self.map == 'complex':
            self.n_agents = 16
            self.total_xy_id_route_list, self.total_len_route_list = init_complex(self.straight_area, self.world_size)
        else:
            raise ValueError('No such map !')

        self.total_xy_real_route_list, self.total_xy_scalar_route_list, self.total_xy_onehot_route_list = get_various_list(self.total_xy_id_route_list, self.world_size)
        self.total_agent_two_id_onehot_route_list, self.total_two_id_onehot_route_list = init_agent_two_id_onehot(self.n_agents)

        self.viewer = None
        self.world_plot_wall = get_world_plot_wall(self.world_size, self.straight_area, self.map)
        self.draw_base_img = self.draw_grid()

        self.obs_shape=self.world_size*2
        self.state_dim=self.world_size*2*self.n_agents
        
        self.obs_shape = self.obs_shape+self.n_agents+2
        self.state_dim = self.state_dim+2*self.n_agents
        self.seed(seed)
        self.possible_two_route = [0, 1]
        self.episode_count = 0
        self.mission = 'cross traffic junctions without collision'
        self.save_frame = save_frame
        if self.save_frame:
            if not os.path.exists('./saved_frames/'):
                os.makedirs('./saved_frames/')

    def save_img(self, obs):
        i = Image.fromarray(obs)
        i.save(os.path.join('./saved_frames/', 'epi'+str(self.episode_count)+'step'+str(self.step_count)+'.png'))

    def reset(self):
        self.episode_count += 1
        self.step_count = 0

        self.two_route_id_list = np.random.choice(self.possible_two_route, size=self.n_agents, replace=True)
        if self.map == 'easy' or self.map == 'moderate':
            self.actual_route_id_list = [two_route_id_i*2+two_route_id for two_route_id_i, two_route_id in enumerate(self.two_route_id_list)]
        elif self.map == 'complex':
            self.actual_route_id_list = []
            base_route_id_list = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]
            for base_route_id, two_route_id in zip(base_route_id_list, self.two_route_id_list):
                self.actual_route_id_list.append(base_route_id+two_route_id)

        self.actual_total_xy_real_route_list = []
        self.actual_total_xy_scalar_route_list = []
        self.actual_total_xy_onehot_route_list = []
        self.actual_total_len_route_list = []
        self.actual_total_agent_two_id_onehot_route_list = []
        self.actual_total_two_id_onehot_route_list = []

        for i, actual_route_id in enumerate(self.actual_route_id_list):
            self.actual_total_xy_real_route_list.append(self.total_xy_real_route_list[actual_route_id])
            self.actual_total_xy_scalar_route_list.append(self.total_xy_scalar_route_list[actual_route_id])
            self.actual_total_xy_onehot_route_list.append(self.total_xy_onehot_route_list[actual_route_id])
            self.actual_total_len_route_list.append(self.total_len_route_list[actual_route_id])
            if self.map == 'complex':
                self.actual_total_agent_two_id_onehot_route_list.append(self.total_agent_two_id_onehot_route_list[actual_route_id])
                self.actual_total_two_id_onehot_route_list.append(self.total_two_id_onehot_route_list[actual_route_id])
            else:
                self.actual_total_agent_two_id_onehot_route_list.append(self.total_agent_two_id_onehot_route_list[actual_route_id])
                self.actual_total_two_id_onehot_route_list.append(self.total_two_id_onehot_route_list[actual_route_id])

        self.world = np.zeros((self.n_agents, self.world_size, self.world_size))
        self.actual_total_cur_id_list, self.world = init_pos(map=self.map, 
                                                             init_range=self.init_range, n_agents=self.n_agents, 
                                                             actual_total_xy_real_route_list=self.actual_total_xy_real_route_list, 
                                                             world=self.world, straight_area=self.straight_area)
        self.alive_masks = np.ones(self.n_agents)
        self.total_rewards = np.zeros(self.n_agents)
        self.world_plot_is_collision = np.zeros(self.n_agents)
        self.success = 1
        self.done = 0

        obs_list, state = self.get_state_obs()
        alive_masks = self.get_alive_mask()

        if self.save_frame:
            img = self.render()
            self.save_img(img)
        return obs_list, state, alive_masks

    def step(self, actions_list):
        assert len(actions_list) == self.n_agents
        self.step_count += 1
        self.world_plot_is_collision = np.zeros(self.n_agents)
        self.total_rewards = np.zeros(self.n_agents)
        self.total_collision_rewards = np.zeros(self.n_agents)
        self.total_exit_rewards = np.zeros(self.n_agents)

        for agent_i, action in enumerate(actions_list):
            if self.alive_masks[agent_i] == 1:
                if action == 1:
                    orig_xy_real = self.actual_total_xy_real_route_list[agent_i][self.actual_total_cur_id_list[agent_i]]
                    self.world[agent_i, orig_xy_real[0], orig_xy_real[1]] = 0
                    self.actual_total_cur_id_list[agent_i] += 1
                    if self.actual_total_cur_id_list[agent_i] == self.actual_total_len_route_list[agent_i]:
                        self.alive_masks[agent_i] = 0
                        self.actual_total_cur_id_list[agent_i] = -1
                        self.total_exit_rewards[agent_i] = 1
                        continue
                    update_xy_real = self.actual_total_xy_real_route_list[agent_i][self.actual_total_cur_id_list[agent_i]]
                    self.world[agent_i, update_xy_real[0], update_xy_real[1]] = 1
        
        for agent_i in range(self.n_agents):
            if self.alive_masks[agent_i] == 1:
                xy_real = self.actual_total_xy_real_route_list[agent_i][self.actual_total_cur_id_list[agent_i]]
                if self.world[:, xy_real[0], xy_real[1]].sum(0) > 1:
                    self.total_collision_rewards[agent_i] = 1
                    self.world_plot_is_collision[agent_i] = 1
                    self.success = 0
        
        if self.alive_masks.sum() == 0 or self.step_count >= self.max_steps:
            done = 1
        else:
            done = 0

        self.total_timestep_rewards = self.step_count*self.timestep_penalty
        self.total_rewards = self.total_collision_rewards*self.collision_penalty + self.total_exit_rewards*self.exit_bouns
        self.total_rewards = self.total_rewards.sum()
        self.total_rewards = self.total_rewards + self.total_timestep_rewards
        self.done = done

        obs_list, state = self.get_state_obs()
        alive_masks = self.get_alive_mask()
        info = {}

        if self.save_frame:
            img = self.render()
            self.save_img(img)
        return obs_list, state, alive_masks, self.total_rewards, done, info

    def get_state_obs(self):
        obs_list = []
        state_list=[]
        for agent_i in range(self.n_agents):
            if self.alive_masks[agent_i] == 1:
                cur_obs = self.actual_total_xy_onehot_route_list[agent_i][self.actual_total_cur_id_list[agent_i]]
                state_list.append(cur_obs)
                state_list.append(self.actual_total_two_id_onehot_route_list[agent_i])
                final_obs = cur_obs
            else:
                cur_obs = np.zeros_like(self.actual_total_xy_onehot_route_list[agent_i][0])
                state_list.append(cur_obs)
                state_list.append(np.zeros_like(self.actual_total_two_id_onehot_route_list[agent_i]))
                final_obs = cur_obs
            final_obs = np.concatenate((final_obs, self.actual_total_agent_two_id_onehot_route_list[agent_i]), axis=0)
            obs_list.append(final_obs)
        state = np.concatenate(state_list, axis=0)
        return obs_list, state

    def get_alive_mask(self):
        return np.copy(self.alive_masks)
    
    def draw_grid(self, cell_size=100, fill='black', line_color='gray'):
        if self.map == 'easy':
            max_rows = self.world_size
            max_cols = self.world_size
        elif self.map == 'moderate':
            max_rows = self.world_size
            max_cols = self.world_size
        elif self.map == 'complex':
            max_rows = self.world_size
            max_cols = self.world_size

        height = max_rows * cell_size
        width = max_cols * cell_size

        image = Image.new(mode='RGB', size=(width, height), color=fill)
        draw = ImageDraw.Draw(image)

        y_start_main_vertical = 0
        y_end_main_vertical = cell_size * self.world_size
        x_start_main_vertical = 0
        x_end_main_vertical = cell_size * self.world_size
        for y in range(y_start_main_vertical, y_end_main_vertical, cell_size):
            line = ((y, x_start_main_vertical), (y, x_end_main_vertical))
            draw.line(line, fill=line_color)
        y_end_main_vertical_final = y_end_main_vertical - 1
        line = ((y_end_main_vertical_final, x_start_main_vertical), (y_end_main_vertical_final, x_end_main_vertical))
        draw.line(line, fill=line_color)

        y_start_main_horizontal = 0
        y_end_main_horizontal = cell_size * self.world_size
        x_start_main_horizontal = 0
        x_end_main_horizontal = cell_size * self.world_size
        for x in range(x_start_main_horizontal, x_end_main_horizontal, cell_size):
            line = ((y_start_main_horizontal, x), (y_end_main_horizontal, x))
            draw.line(line, fill=line_color)
        x_end_main_horizontal_final = x_end_main_horizontal - 1
        line = ((y_start_main_horizontal, x_end_main_horizontal_final), (y_end_main_horizontal, x_end_main_horizontal_final))
        draw.line(line, fill=line_color)

        for row in range(self.world_plot_wall.shape[0]):
            for col in range(self.world_plot_wall.shape[1]):
                if self.world_plot_wall[row, col] == 1:
                    self.draw_wall(image, row, col, margin=0)
        del draw
        return image

    def draw_wall(self, image, row, col, cell_size=100, fill='gray', margin=0):
        assert cell_size is not None and 0 <= margin <= 1
        row, col = row * cell_size, col * cell_size
        margin_x, margin_y = margin * cell_size, margin * cell_size
        x, y, x_dash, y_dash = row + margin_x, col + margin_y, row + cell_size - margin_x, col + cell_size - margin_y
        ImageDraw.Draw(image).rectangle([(y, x), (y_dash, x_dash)], fill=fill)

    def draw_agent_ellipse(self, image, agent_i, row, col, cell_size=100, agent_fill='white', collide_fill='red', 
                         radius=0.15, margin=0.1, collide=False):
        if collide == False:
            row, col = row * cell_size, col * cell_size
            gap_x, gap_y = cell_size * radius, cell_size * radius
            x, y = row + gap_x, col + gap_y
            x_dash, y_dash = row + cell_size - gap_x, col + cell_size - gap_y
            ImageDraw.Draw(image).ellipse([(y, x), (y_dash, x_dash)], outline=self.agent_color[agent_i], fill=self.agent_color[agent_i])
        if collide == True:
            row, col = row * cell_size, col * cell_size
            gap_x, gap_y = cell_size * radius, cell_size * radius
            x, y = row + gap_x, col + gap_y
            x_dash, y_dash = row + cell_size - gap_x, col + cell_size - gap_y
            ImageDraw.Draw(image).ellipse([(y, x), (y_dash, x_dash)], outline=collide_fill, fill=collide_fill)

    def draw_agent_num(self, image, agent_i, row, col, cell_size=100, agent_fill='white', collide_fill='red', 
                         radius=0.3, margin=0.1):
        row_id, col_id = row * cell_size, col * cell_size
        if agent_i < 10:
            margin_x, margin_y = margin * cell_size, (margin+0.05) * cell_size
        else:
            margin_x, margin_y = margin * cell_size, margin * cell_size
        x_id, y_id = row_id + margin_x, col_id + margin_y
        ImageDraw.Draw(image).text((y_id, x_id), text=str(agent_i), fill=agent_fill, font=self.font_indicator)

    def render(self, mode='human', close=None, highlight=None, tile_size=None):
        img = copy.copy(self.draw_base_img)
        for agent_i in range(self.n_agents):
            if self.alive_masks[agent_i] == 1:
                xy_real = self.actual_total_xy_real_route_list[agent_i][self.actual_total_cur_id_list[agent_i]]
                if self.world_plot_is_collision[agent_i] == 1:
                    self.draw_agent_ellipse(img, agent_i, xy_real[0], xy_real[1], collide=True)
                else:
                    self.draw_agent_ellipse(img, agent_i, xy_real[0], xy_real[1], collide=False, margin=0.38)
                    self.draw_agent_num(img, agent_i, xy_real[0], xy_real[1], margin=0.38)
        return img
