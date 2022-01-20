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
from gym_minigrid.minigrid import *

COLORS = {
    'white' : np.array([255, 255, 255]),
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

COLOR_TO_IDX.update({
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'white' : 6
})

OBJECT_TO_IDX.update({
    'empty':0,
    'ant':1,
    'queen':2,
    'food':3,
    'lamb':1,
    'ewe':2,
    'sheepfold':3,
    'wall':9
})

DIR_TO_IDX = [4, 5, 6, 7]

class ExtGrid(Grid):

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

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)

        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))
        # else:
        #     fill_coords(img, point_in_circle(0.5, 0.5, 0.31), (255, 0, 0))
        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self,
               tile_size,
               agent_pos=None,
               agent_dir=None,
               highlight_mask=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # highlight_mask = None
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))

                tile_img = self.render_tile(cell,
                                            agent_dir=agent_dir if agent_here else None,
                                            highlight=highlight_mask[i, j],
                                            tile_size=tile_size)

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def wall_rect(self, x, y, w, h):

        self.horz_wall(x, y, w, obj_type=ExtWall)   
        self.horz_wall(x, y+h-1, w, obj_type=ExtWall)
        self.vert_wall(x, y, h, obj_type=ExtWall)   
        self.vert_wall(x+w-1, y, h, obj_type=ExtWall) 

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[j, i] = OBJECT_TO_IDX['empty']
                    else:
                        array[j, i] = v.encode()[0]

        return array


class ExtWall(Wall):
    def __init__(self, color='grey'):
        super().__init__(color)

    def can_attack(self):
        return False


class ExtendedWorldObj(WorldObj):
    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None
    @staticmethod
    def dis(pos1, pos2):
        return math.sqrt(pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2))

    def get_nearest_target_pos(self, target_list):
        nearest_target_idx = None
        nearest_target_pos = None
        nearest_target_dist = 99999999
        for i in range(len(target_list)):
            dist = self.dis(self.cur_pos, target_list[i].cur_pos)
            if dist < nearest_target_dist:
                nearest_target_dist = dist
                nearest_target_pos = target_list[i].cur_pos
                nearest_target_idx = i
        return nearest_target_pos, nearest_target_idx
