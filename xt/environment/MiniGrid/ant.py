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
from .extended_minigrid import *
from operator import add
from PIL import Image
import os
import scipy.signal as sig

class Ant(ExtendedWorldObj):
    def __init__(self, idx, init_energy=10, energy_threshold=1, color_default='grey', color_carrying='green'):
        super(Ant, self).__init__('ant', color_default)
        self.energy = init_energy
        self.idx = idx
        self.carrying = None
        self.energy_threshold = energy_threshold
        self.color_default = color_default
        self.color_carrying = color_carrying

    def can_attack(self):
        return True

    def render(self, img):
        if self.carrying is None:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color_default])
        else:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color_carrying])


class Queen(ExtendedWorldObj):
    def __init__(self, idx, init_energy=10, breed_energy_threshold=30, breed_energy_cost=15, queen_rate=0.1, act_point=1, color='white'):
        super(Queen, self).__init__('queen', color)
        self.energy = init_energy
        self.idx = idx
        self.breed_energy_threshold = breed_energy_threshold
        self.breed_energy_cost = breed_energy_cost
        self.queen_rate = queen_rate
        self.act_point = 1

    def can_attack(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Food(ExtendedWorldObj):
    def __init__(self, idx, energy=40):
        super().__init__('food', 'yellow')
        self.energy = energy
        self.idx = idx

    def can_attack(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class AntEnv(MiniGridEnv):
    """
    Env with ants, queen and foods
    """

    class AgentActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        attack = 3
        
        # Pick up an object
        pickup = 4
        # Drop an object
        drop = 5
        # Toggle/activate an object
        toggle = 6

    def __init__(
            self,
            size=20,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            agent_energy = 20,
            num_init_queen = 1,
            num_init_ant = 4,
            num_init_food = 3,
            food_gen_rate = 0.3,
            step_penalty=False,
            max_steps = 200,
            save_frame = False
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_init_queen = num_init_queen
        self.num_init_ant = num_init_ant
        self.num_init_food = num_init_food
        self.food_gen_rate = food_gen_rate
        self.step_penalty = step_penalty
        self.agent_init_energy = agent_energy
        self.agent_energy = agent_energy
        self.save_frame = save_frame
        self.n_agents = 1

        if self.save_frame:
            if not os.path.exists('./saved_frames/'):
                os.makedirs('./saved_frames/')
        self.episode_count = 0

        super().__init__(grid_size=size,
                         max_steps=max_steps,
                         see_through_walls=True)
        # Allow only 4 actions permitted: left, right, forward, attack

        self.actions = AntEnv.AgentActions
        self.action_space = spaces.Discrete(self.actions.attack + 1)
        self.reward_range = (-100, 1)

    @staticmethod
    def del_from_list(list, idx):
        del (list[idx])
        for i in range(idx, len(list)):
            list[i].idx -= 1

    def save_img(self, obs):
        i = Image.fromarray(obs['img'])
        i.save(os.path.join('./saved_frames/', 'epi'+str(self.episode_count)+'step'+str(self.step_count)+'.png'))

    def reset(self):
        super().reset()
        self.episode_count += 1
        self.step_count = 0

        obs = self.get_obs()
        self.agent_energy = self.agent_init_energy
        if self.save_frame:
            self.save_img(obs)

        return obs

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = ExtGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place ants
        self.ants = []
        for i in range(self.num_init_ant):
            self.ants.append(Ant(idx=i))
            self.place_obj(self.ants[i], max_tries=100)

        # Place queens
        self.queens = []
        for i in range(self.num_init_queen):
            self.queens.append(Queen(idx=i))
            self.place_obj(self.queens[i], max_tries=100)

        # Place foods
        self.foods = []
        for i in range(self.num_init_food):
            self.foods.append(Food(idx=i))
            self.place_obj(self.foods[i], max_tries=100)

        self.mission = "kill all ants and queens"

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # original step ###############################################
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Attack ###########################################
        elif action == self.actions.attack:
            if self.agent_energy > 0:
                self.agent_energy -= 1
                if fwd_cell and fwd_cell.can_attack():
                    if fwd_cell.type == 'ant':
                        self.del_from_list(self.ants, fwd_cell.idx)
                        reward += 1
                    elif fwd_cell.type == 'queen':
                        self.del_from_list(self.queens, fwd_cell.idx)
                        reward += 1
                    elif fwd_cell.type == 'food':
                        self.del_from_list(self.foods, fwd_cell.idx)
                    self.grid.set(*fwd_pos, None)
        # ##################################################
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True
            reward -= (len(self.ants) + len(self.queens))

        # ################################################################
        if self.step_penalty:
            reward -= -1
        else:
            reward -= 0
        for ant in self.ants:
            if not ant.carrying:
                # if ant's energy < threshold, and without food, he need wait for energy regen.
                if ant.energy < ant.energy_threshold:
                    ant.energy += 1
                else:
                    # FIXME: if there has been an object on the next pos, its enery will still decay.
                    target_pos, target_idx = ant.get_nearest_target_pos(self.foods)
                    if target_pos is None:
                        # if there is no food, the ant will stand still.
                        ant.energy -= 1
                        continue
                    # if the ant has arrived at the food, it will pickup it.
                    if ant.dis(ant.cur_pos, target_pos) <= math.sqrt(2.01):
                        ant.carrying = self.grid.get(*target_pos)
                        self.del_from_list(self.foods, target_idx)
                        self.grid.set(*target_pos, None)
                    else:
                        self.ant_move_towards(ant.idx, target_pos)
                    ant.energy -= 1
            else:
                # if ant's energy < threshold, eat food
                if ant.energy < ant.energy_threshold:
                    ant.energy += ant.carrying.energy
                    ant.carrying = None
                else:
                    # FIXME: if there has been an object on the next pos, its enery will still decay.
                    target_pos, target_idx = ant.get_nearest_target_pos(self.queens)
                    if target_pos is None:
                        # if there is no queen, the ant will stand still.
                        ant.energy -= 1
                        continue
                    if ant.dis(ant.cur_pos, target_pos) <= math.sqrt(2.01):
                        self.queens[target_idx].energy += ant.carrying.energy
                        ant.carrying = None
                    else:
                        self.ant_move_towards(ant.idx, target_pos)
                    ant.energy -= 1

        for queen in self.queens:
            if queen.act_point == 0:
                queen.act_point += 1
                continue
            else:
                queen.act_point -= 1
            if queen.energy > queen.breed_energy_threshold:
                if self._rand_float(0, 1) < queen.queen_rate:
                    new_queen = Queen(idx=len(self.queens))
                    result = self.place_obj_ext(new_queen, top=(queen.cur_pos[0]-1,queen.cur_pos[1]-1),
                                            size=(3, 3), max_tries=100)
                    if result is not None:
                        self.queens.append(new_queen)
                        queen.energy -= queen.breed_energy_cost
                else:
                    new_ant = Ant(idx=len(self.ants))
                    result = self.place_obj_ext(new_ant, top=(queen.cur_pos[0] - 1, queen.cur_pos[1] - 1),
                                                size=(3, 3), max_tries=100)
                    if result is not None:
                        self.ants.append(new_ant)
                        queen.energy -= queen.breed_energy_cost
            else:
                self.queen_move_randomly(queen.idx)
                queen.energy -= 1

        if self._rand_float(0, 1) < self.food_gen_rate:
            new_food = Food(idx=len(self.foods))
            result = self.place_obj_ext(new_food, max_tries=100)
            if result is not None:
                self.foods.append(new_food)
        obs = self.get_obs()
        # obs = {'img':img}
        if self.save_frame:
            self.save_img(obs)
        return obs, reward, done, {}

    @staticmethod
    def sign(x):
        if x != 0:
            return int(x/abs(x))
        else:
            return 0

    @staticmethod
    def angle(v1, v2):
        # calculate angle of two vectors
        dx1 = v1[0]
        dy1 = v1[1]
        dx2 = v2[0]
        dy2 = v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)

        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)

        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def place_obj_ext(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                return None # Different from parent class's place obj function. This is to avoid raise error.
            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break
        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def ant_move_towards(self, i_ant, target_obst_pos):
        v = 1 # velocity
        cur_pos = self.ants[i_ant].cur_pos
        pos_diff = (target_obst_pos[0] - cur_pos[0], target_obst_pos[1] - cur_pos[1])
        if abs(pos_diff[0]) > 1 or abs(pos_diff[1]) > 1:
            x_dis = pos_diff[0]
            y_dis = pos_diff[1]
            if abs(x_dis) > abs(y_dis):
                next_pos = (cur_pos[0] + self.sign(x_dis) * v, cur_pos[1])
            elif abs(x_dis) < abs(y_dis):
                next_pos = (cur_pos[0], cur_pos[1] + self.sign(y_dis) * v)
            else:
                next_pos = (cur_pos[0] + self.sign(x_dis) * v,  cur_pos[1] + self.sign(y_dis) * v)
            result = self.place_obj_ext(self.ants[i_ant], top=next_pos, size=(1,1), max_tries=100)
            if result is not None:
                self.grid.set(*cur_pos, None)

    def queen_move_randomly(self, i_queen):
        old_pos = self.queens[i_queen].cur_pos
        top = tuple(map(add, old_pos, (-1, -1)))
        result = self.place_obj_ext(self.queens[i_queen], top=top, size=(3, 3), max_tries=100)
        if result is not None:
            self.grid.set(*old_pos, None)

    def get_rgb_obs(self):
        
        img = self.grid.render(32,
                               self.agent_pos,
                               self.agent_dir,
                               highlight_mask= None)
        
        return img # rgb image, (640, 640, 3)

    def get_simple_obs(self):
        simple_img = self.grid.encode(vis_mask=None)
        agent_pos = self.agent_pos

        simple_img[agent_pos[1]][agent_pos[0]] = DIR_TO_IDX[self.agent_dir]
        return simple_img

    def get_obs(self):
        img = self.get_rgb_obs()

        simple_img = self.get_simple_obs()

        obs = {'img': img,
               'simple': simple_img}

        img = self.downsample3d(img, 7)
        img = img[4:88, 4:88, :]

        return img

    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        # print('TILE_PIXELS = ', TILE_PIXELS)

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def downsample2d(self, inputArray, kernelSize):
        """This function downsamples a 2d numpy array by convolving with a flat
        kernel and then sub-sampling the resulting array.
        A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
        a resulting downsampling of 2-fold.
        :param: inputArray: 2d numpy array
        :param: kernelSize: integer
        """
        average_kernel = np.ones((kernelSize,kernelSize))

        blurred_array = sig.convolve2d(inputArray, average_kernel, mode='same')
        downsampled_array = blurred_array[::kernelSize,::kernelSize]
        return downsampled_array

    def downsample3d(self, inputArray, kernelSize):
        """This function downsamples a 3d numpy array (an image stack)
        by convolving each frame with a flat kernel and then sub-sampling the resulting array,
        re-building a smaller 3d numpy array.
        A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
        a resulting downsampling of 2-fold.
        The array will be downsampled in the first 2 dimensions, as shown below.
        import numpy as np
        >>> A = np.random.random((100,100,20))
        >>> B = downsample3d(A, 2)
        >>> A.shape
        (100, 100, 20)
        >>> B.shape
        (50, 50, 20)
        :param: inputArray: 2d numpy array
        :param: kernelSize: integer
        """
        first_smaller = self.downsample2d(inputArray[:,:,0], kernelSize)
        smaller = np.zeros((first_smaller.shape[0], first_smaller.shape[1], inputArray.shape[2]))
        smaller[:,:,0] = first_smaller

        for i in range(1, inputArray.shape[2]):
            smaller[:,:,i] = self.downsample2d(inputArray[:,:,i], kernelSize)
        return smaller
