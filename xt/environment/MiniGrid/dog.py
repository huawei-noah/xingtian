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
from PIL import Image
import os
import scipy.signal as sig

class DogGrid(ExtGrid):
    @classmethod
    def render_tile(cls,
                    obj,
                    agent_dir=None,
                    highlight=False,
                    tile_size=TILE_PIXELS,
                    subdivs=3):
        """
        Render a tile and cache the result
        """

        key = (agent_dir, highlight, tile_size)

        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

class Lamb(ExtendedWorldObj):
    def __init__(self, idx, rand_rate=0.3, safe_dist=5):
        super().__init__('lamb', 'grey')
        self.idx = idx
        self.rand_rate = rand_rate
        self.safe_dist = safe_dist

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Ewe(ExtendedWorldObj):
    def __init__(self, idx, safe_dist = 5):
        super().__init__('ewe', 'white')
        self.idx = idx
        self.safe_dist = safe_dist

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Sheepfold(ExtendedWorldObj):
    def __init__(self):
        super().__init__('sheepfold', 'green')

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class DogEnv(MiniGridEnv):
    """
    Env with sheeps and dogs
    """
    class AgentActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        down = 3

    def __init__(self,
                 size=20,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 num_lamb = 4,
                 step_penalty=False,
                 max_steps = 400,
                 save_frame = False):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_lamb = num_lamb            
        self.step_penalty = step_penalty     
        self.save_frame = save_frame       
        self.n_agents = 1

        if self.save_frame:
            if not os.path.exists('./saved_frames/'):
                os.makedirs('./saved_frames/')

        self.episode_count = 0

        super().__init__(grid_size=size,
                         max_steps=max_steps,
                         see_through_walls=True)

        self.actions = DogEnv.AgentActions

        self.action_space = spaces.Discrete(self.actions.down + 1)

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

        if self.save_frame:
            self.save_img(obs)

        return obs

    def place_agent(self,
                    top=None,
                    size=None,
                    rand_dir=True,
                    max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None

        pos = self.place_obj(None, top, size, max_tries=max_tries)

        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    def _gen_grid(self, width, height):

        self.grid = DogGrid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.sheepfold = Sheepfold()
        self.place_obj(self.sheepfold, top=(width-2, height-2), size=(1, 1))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            # self.agent_pos = (1, 1)
            self.agent_dir = self.agent_start_dir
            # self.agent_dir = 0
        else:
            self.place_agent()

        # Place lambs
        self.lambs = []
        # self.num_lamb = 4
        for i in range(self.num_lamb):
            self.lambs.append(Lamb(idx=i))
            self.place_obj(self.lambs[i], max_tries=100)

        # Place the ewe
        self.ewe = [Ewe(idx=0)]
        self.place_obj(self.ewe[0], max_tries=100)

        self.mission = "push all sheep into the sheepfold"

    @property
    def up_pos(self):
        return self.agent_pos + np.array((0, -1))

    @property
    def down_pos(self):
        return self.agent_pos + np.array((0, 1))

    @property
    def left_pos(self):
        return self.agent_pos + np.array((-1, 0))

    @property
    def right_pos(self):
        return self.agent_pos + np.array((1, 0))

    def step(self, action):

        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # original step
        self.step_count += 1

        reward = 0
        done = False

        # Rotate left
        if action == self.actions.left:
            lft_pos = self.left_pos
            lft_cell = self.grid.get(*lft_pos)
            if lft_cell == None or lft_cell.can_overlap():
                self.agent_pos = lft_pos
        # Rotate right
        elif action == self.actions.right:
            rgt_pos = self.right_pos
            rgt_cell = self.grid.get(*rgt_pos)
            if rgt_cell == None or rgt_cell.can_overlap():
                self.agent_pos = rgt_pos
        # Move forward
        elif action == self.actions.forward:
            up_pos = self.up_pos
            up_cell = self.grid.get(*up_pos)
            if up_cell == None or up_cell.can_overlap():
                self.agent_pos = up_pos
        elif action == self.actions.down:
            dwn_pos = self.down_pos
            dwn_cell = self.grid.get(*dwn_pos)
            if dwn_cell == None or dwn_cell.can_overlap():
                self.agent_pos = dwn_pos
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True
            reward -= (len(self.lambs) + len(self.ewe))

        if self.step_penalty:
            reward -= -1

        for ewe in self.ewe:
            if ewe.dis(ewe.cur_pos, self.agent_pos) < ewe.safe_dist:
                reward += self.sheep_move_away(ewe)
            else:
                reward += self.sheep_move_randomly(ewe)

        for lamb in self.lambs:
            if lamb.dis(lamb.cur_pos, self.agent_pos) < lamb.safe_dist:
                reward += self.sheep_move_away(lamb)
            else:
                if self._rand_float(0, 1) < lamb.rand_rate:
                    reward += self.sheep_move_randomly(lamb)
                else:
                    reward += self.sheep_move_towards(lamb)

        if len(self.ewe) == 0 and len(self.lambs) == 0:
            done = True

        obs = self.get_obs()

        if self.save_frame:
            self.save_img(obs)

        return obs, reward, done, {}

    @staticmethod
    def sign(x):
        if x != 0:
            return int(x/abs(x))
        else:
            return 0

    def place_obj_ext(self,
                      obj,
                      top=None,
                      size=None,
                      reject_fn=None,
                      max_tries=math.inf):
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

    def sheep_move_away(self, sheep):
        v = 1
        dog_pos = self.agent_pos
        cur_pos = sheep.cur_pos
        pos_diff = (dog_pos[0] - cur_pos[0], dog_pos[1] - cur_pos[1])

        x_dis = pos_diff[0]
        y_dis = pos_diff[1]
        if (cur_pos[0] == 1 and cur_pos[1] == 1) or (cur_pos[0] == 1 and cur_pos[1] == self.grid.height - 2) \
                or (cur_pos[0] == self.grid.width - 2 and cur_pos[1] == self.grid.height - 2) or (
                cur_pos[0] == self.grid.width - 2 and cur_pos[1] == 1):
            if abs(x_dis) < abs(y_dis):
                next_pos = (cur_pos[0] + self.sign(x_dis) * v, cur_pos[1])
            else:
                next_pos = (cur_pos[0], cur_pos[1] + self.sign(y_dis) * v)
        elif cur_pos[0] == 1 or cur_pos[0] == self.grid.width - 2:
            sign = self.sign(y_dis) if self.sign(y_dis) != 0 else self.sign(cur_pos[1] - (self.grid.height - 1) / 2)
            next_pos = (cur_pos[0], cur_pos[1] - sign * v)
        elif cur_pos[1] == 1 or cur_pos[1] == self.grid.height - 2:
            sign = self.sign(x_dis) if self.sign(x_dis) != 0 else self.sign(cur_pos[0] - (self.grid.width - 1) / 2)
            next_pos = (cur_pos[0] - sign * v, cur_pos[1])
        else:
            next_pos = (cur_pos[0] - self.sign(x_dis) * v, cur_pos[1] - self.sign(y_dis) * v)
        next_cell = self.grid.get(*next_pos)

        if next_cell is not None and next_cell.type == 'sheepfold':
            reward = 1
            self.del_from_list(self.ewe if sheep.type == 'ewe' else self.lambs, sheep.idx)
            self.grid.set(*cur_pos, None)
        else:
            reward = 0
            result = self.place_obj_ext(sheep, top=next_pos, size=(1, 1), max_tries=100)
            if result is not None:
                self.grid.set(*cur_pos, None)

        return reward

    def sheep_move_towards(self, sheep):
        v = 1 # velocity
        cur_pos = sheep.cur_pos
        if len(self.ewe) > 0:
            target_pos = self.ewe[0].cur_pos
        else:
            target_pos = self.sheepfold.cur_pos
        pos_diff = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
        if abs(pos_diff[0]) > 0 or abs(pos_diff[1]) > 0:
            x_dis = pos_diff[0]
            y_dis = pos_diff[1]
            if abs(x_dis) > abs(y_dis):
                next_pos = (cur_pos[0] + self.sign(x_dis) * v, cur_pos[1])
            elif abs(x_dis) < abs(y_dis):
                next_pos = (cur_pos[0], cur_pos[1] + self.sign(y_dis) * v)
            else:
                next_pos = (cur_pos[0] + self.sign(x_dis) * v,  cur_pos[1] + self.sign(y_dis) * v)
            next_cell = self.grid.get(*next_pos)
            if next_cell is not None and next_cell.type == 'sheepfold':
                reward = 1
                self.del_from_list(self.ewe if sheep.type == 'ewe' else self.lambs, sheep.idx)
                self.grid.set(*cur_pos, None)
            else:
                reward = 0
                result = self.place_obj_ext(sheep, top=next_pos, size=(1,1), max_tries=100)
                if result is not None:
                    self.grid.set(*cur_pos, None)
            return reward

    def sheep_move_randomly(self, sheep):
        cur_pos = sheep.cur_pos
        next_pos = np.array((
            self._rand_int(sheep.cur_pos[0] - 1, min(sheep.cur_pos[0] + 2, self.grid.width)),
            self._rand_int(sheep.cur_pos[1] - 1, min(sheep.cur_pos[1] + 2, self.grid.height))
        ))
        next_cell = self.grid.get(*next_pos)
        if next_cell is not None and next_cell.type == 'sheepfold':
            reward = 1
            self.del_from_list(self.ewe if sheep.type == 'ewe' else self.lambs, sheep.idx)
            self.grid.set(*cur_pos, None)
        else:
            reward = 0
            result = self.place_obj_ext(sheep, top=next_pos, size=(1, 1), max_tries=100)
            if result is not None:
                self.grid.set(*cur_pos, None)
        return reward

    def get_rgb_obs(self):

        img = self.grid.render(32,
                               self.agent_pos,
                               self.agent_dir,
                               highlight_mask=None)

        return img

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
        img = self.grid.render(tile_size,
                               self.agent_pos,
                               self.agent_dir,
                               highlight_mask=highlight_mask if highlight else None)

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
