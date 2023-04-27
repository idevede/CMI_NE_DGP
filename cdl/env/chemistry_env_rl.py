"""Gym environment for changing colors of shapes."""
import sys

# CHANGES: replace the MLP transition with a customized functional map; add ring structure

import numpy as np
import torch
import torch.nn as nn
import re

import gym
from collections import OrderedDict
from dataclasses import dataclass
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import skimage

from cswm import utils
import random

import copy

graphs = {
    'chain3': '0->1->2->0',
    'fork3': '0->{1-2}',
    'collider3': '{0-1}->2',
    'collider4': '{0-2}->3',
    'collider5': '{0-3}->4',
    'collider6': '{0-4}->5',
    'collider7': '{0-5}->6',
    'collider8': '{0-6}->7',
    'collider9': '{0-7}->8',
    'collider10': '{0-8}->9',
    'collider11': '{0-9}->10',
    'collider12': '{0-10}->11',
    'collider13': '{0-11}->12',
    'collider14': '{0-12}->13',
    'collider15': '{0-13}->14',
    'confounder3': '{0-2}->{0-2}',
    'chain4': '0->1->2->3->0',
    'chain5': '0->1->2->3->4->0',
    'chain6': '0->1->2->3->4->5',
    'chain7': '0->1->2->3->4->5->6',
    'chain8': '0->1->2->3->4->5->6->7',
    'chain9': '0->1->2->3->4->5->6->7->8',
    'chain10': '0->1->2->3->4->5->6->7->8->9->0',
    'chain11': '0->1->2->3->4->5->6->7->8->9->10',
    'chain12': '0->1->2->3->4->5->6->7->8->9->10->11',
    'chain13': '0->1->2->3->4->5->6->7->8->9->10->11->12',
    'chain14': '0->1->2->3->4->5->6->7->8->9->10->11->12->13',
    'chain15': '0->1->2->3->4->5->6->7->8->9->10->11->12->13->14',
    'full3': '{0-2}->{0-2}',
    'full4': '{0-3}->{0-3}',
    'full5': '{0-4}->{0-4}',
    'full6': '{0-5}->{0-5}',
    'full7': '{0-6}->{0-6}',
    'full8': '{0-7}->{0-7}',
    'full9': '{0-8}->{0-8}',
    'full10': '{0-9}->{0-9}',
    'full11': '{0-10}->{0-10}',
    'full12': '{0-11}->{0-11}',
    'full13': '{0-12}->{0-12}',
    'full14': '{0-13}->{0-13}',
    'full15': '{0-14}->{0-14}',
    'tree9': '0->1->3->7,0->2->6,1->4,3->8,2->5',
    'tree10': '0->1->3->7,0->2->6,1->4->9,3->8,2->5',
    'tree11': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5',
    'tree12': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11',
    'tree13': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12',
    'tree14': '0->1->3->7,0->2->6,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'tree15': '0->1->3->7,0->2->6->14,1->4->10,3->8,4->9,2->5->11,5->12,6->13',
    'jungle3': '0->{1-2}',
    'jungle4': '0->1->3,0->2,0->3',
    'jungle5': '0->1->3,1->4,0->2,0->3,0->4',
    'jungle6': '0->1->3,1->4,0->2->5,0->3,0->4,0->5',
    'jungle7': '0->1->3,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6',
    'jungle8': '0->1->3->7,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7',
    'jungle9': '0->1->3->7,3->8,1->4,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8',
    'jungle10': '0->1->3->7,3->8,1->4->9,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9',
    'jungle11': '0->1->3->7,3->8,1->4->9,4->10,0->2->5,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10',
    'jungle12': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11',
    'jungle13': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12',
    'jungle14': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13',
    'jungle15': '0->1->3->7,3->8,1->4->9,4->10,0->2->5->11,5->12,2->6->13,6->14,0->3,0->4,0->5,0->6,1->7,1->8,1->9,1->10,2->11,2->12,2->13,2->14',
    'bidiag3': '{0-2}->{0-2}',
    'bidiag4': '{0-1}->{1-2}->{2-3}',
    'bidiag5': '{0-1}->{1-2}->{2-3}->{3-4}',
    'bidiag6': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}',
    'bidiag7': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}',
    'bidiag8': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}',
    'bidiag9': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}',
    'bidiag10': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}',
    'bidiag11': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}',
    'bidiag12': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}',
    'bidiag13': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}',
    'bidiag14': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}',
    'bidiag15': '{0-1}->{1-2}->{2-3}->{3-4}->{4-5}->{5-6}->{6-7}->{7-8}->{8-9}->{9-10}->{10-11}->{11-12}->{12-13}->{13-14}',
}


def parse_skeleton(graph, M=None):
    """
    Parse the skeleton of a causal graph in the mini-language of --graph.
    
    The mini-language is:
        
        GRAPH      = ""
                     CHAIN{, CHAIN}*
        CHAIN      = INT_OR_SET {-> INT_OR_SET}
        INT_OR_SET = INT | SET
        INT        = [0-9]*
        SET        = \{ SET_ELEM {, SET_ELEM}* \}
        SET_ELEM   = INT | INT_RANGE
        INT_RANGE  = INT - INT
    """

    regex = re.compile(r'''
        \s*                                      # Skip preceding whitespace
        (                                        # The set of tokens we may capture, including
          [,]                                  | # Commas
          (?:\d+)                              | # Integers
          (?:                                    # Integer set:
            \{                                   #   Opening brace...
              \s*                                #   Whitespace...
              \d+\s*(?:-\s*\d+\s*)?              #   First integer (range) in set...
              (?:,\s*\d+\s*(?:-\s*\d+\s*)?\s*)*  #   Subsequent integers (ranges)
            \}                                   #   Closing brace...
          )                                    | # End of integer set.
          (?:->)                                 # Arrows
        )
    ''', re.A | re.X)

    # Utilities
    def parse_int(s):
        try:
            return int(s.strip())
        except:
            return None

    def parse_intrange(s):
        try:
            sa, sb = map(str.strip, s.strip().split("-", 1))
            sa, sb = int(sa), int(sb)
            sa, sb = min(sa, sb), max(sa, sb) + 1
            return range(sa, sb)
        except:
            return None

    def parse_intset(s):
        try:
            i = set()
            for s in map(str.strip, s.strip()[1:-1].split(",")):
                if parse_int(s) is not None:
                    i.add(parse_int(s))
                else:
                    i.update(set(parse_intrange(s)))
            return sorted(i)
        except:
            return None

    def parse_either(s):
        asint = parse_int(s)
        if asint is not None: return asint
        asset = parse_intset(s)
        if asset is not None: return asset
        raise ValueError

    def find_max(chains):
        m = 0
        for chain in chains:
            for link in chain:
                link = max(link) if isinstance(link, list) else link
                m = max(link, m)
        return m

    # Crack the string into a list of lists of (ints | lists of ints)
    graph = [graph] if isinstance(graph, str) else graph
    chains = []
    for gstr in graph:
        for chain in re.findall("((?:[^,{]+|\{.*?\})+)+", gstr, re.A):
            links = list(map(str.strip, regex.findall(chain)))
            assert (len(links) & 1)

            chain = [parse_either(links.pop(0))]
            while links:
                assert links.pop(0) == "->"
                chain.append(parse_either(links.pop(0)))
            chains.append(chain)

    # Find the maximum integer referenced within the skeleton
    uM = find_max(chains) + 1
    if M is None:
        M = uM
    else:
        assert (M >= uM)
        M = max(M, uM)

    # Allocate adjacency matrix.
    gamma = np.zeros((M, M), dtype=np.float32)

    # Interpret the skeleton
    for chain in chains:
        for prevlink, nextlink in zip(chain[:-1], chain[1:]):
            # print('PRELINK,NEXTLINK')
            # print(prevlink, nextlink)
            if isinstance(prevlink, list) and isinstance(nextlink, list):
                for i in nextlink:
                    for j in prevlink:
                        if i > j:
                            gamma[i, j] = 1
            elif isinstance(prevlink, list) and isinstance(nextlink, int):
                for j in prevlink:
                    if nextlink > j:
                        gamma[nextlink, j] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, list):
                minn = min(nextlink)
                if minn == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif minn < prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to ancestor " +
                                     str(minn) + " !")
                else:
                    for i in nextlink:
                        gamma[i, prevlink] = 1
            elif isinstance(prevlink, int) and isinstance(nextlink, int):
                if nextlink == prevlink:
                    raise ValueError("Edges are not allowed from " +
                                     str(prevlink) + " to oneself!")
                elif nextlink < prevlink:
                    # raise ValueError("Edges are not allowed from " +
                    #                  str(prevlink) + " to ancestor " +
                    #                  str(nextlink) + " !")
                    gamma[nextlink, prevlink] = 1
                else:
                    gamma[nextlink, prevlink] = 1

    # print('GAMMA')
    # print(gamma)
    # sys.exit()

    # Return adjacency matrix.
    return gamma


mpl.use('Agg')


def random_dag(M, N, rng, g=None):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
    if g is None:
        expParents = 5
        idx = np.arange(M).astype(np.float32)[:, np.newaxis]
        idx_maxed = np.minimum(idx * 0.5, expParents)
        p = np.broadcast_to(idx_maxed / (idx + 1), (M, M))
        B = rng.binomial(1, p)
        B = np.tril(B, -1)
        return B
    else:
        gammagt = parse_skeleton(g, M=M)
        return gammagt


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(objects, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in objects.items():
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im / 255.


class MLP(nn.Module):
    def __init__(self, num_objs, num_colors):
        super().__init__()
        self.num_objs = num_objs
        self.num_colors = num_colors

    def forward(self, x, mask):
        x_masked = x * mask
        id = torch.argmax(x_masked)
        obj_id = id // self.num_colors
        color_id = id % self.num_colors
        obj_child_id = (obj_id + 1) % self.num_objs  # only for the ring structure

        color_child = x[0, obj_child_id * self.num_colors: (obj_child_id + 1) * self.num_colors]
        color_child_id = torch.argmax(color_child)
        # state transition of a variable when the action node of its parent variable is intervened
        color_child_id = (color_child_id + 1) % self.num_colors

        color_child_new = torch.zeros(self.num_colors)
        color_child_new[color_child_id] = 1

        return color_child_new


@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x,
                     self.y + other.y)


@dataclass
class Object:
    pos: Coord
    color: int


class ColorChangingRL(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=5, height=5, render_type='shapes',
                 *, num_objects=5,
                 num_colors=None, movement='Static', graph='chain5', max_steps=10, seed=None):
        # self.np_random.seed(0)
        # torch.manual_seed(0)
        self.width = width
        self.height = height
        self.render_type = render_type

        self.num_objects = num_objects
        self.movement = movement

        if num_colors is None:
            num_colors = num_objects
        self.num_colors = num_colors
        # self.num_actions = self.num_objects * self.num_colors  # Chao commented
        # self.num_actions = self.num_objects + 1  # Chao added; adding no-action node
        self.num_actions = self.num_objects  # Chao added
        self.num_target_interventions = max_steps
        self.max_steps = max_steps

        self.mlps = []
        self.mask = None

        colors = ['blue', 'green', 'yellow', 'white', 'red']

        self.colors, _ = utils.get_colors_and_weights(cmap='Set1',
                                                      num_colors=self.num_colors)  # [mpl.colors.to_rgba(colors[i]) for i in range(self.num_colors)]
        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = None
        # self.object_to_color_init = None  # Chao added to test

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()

        self.adjacency_matrix = None

        self.mlps = []
        for i in range(self.num_objects):
            self.mlps.append(
                MLP(self.num_objects, self.num_colors))  # All objects have the same MLP structure and paramters but with different inputs.

        self.graph = graph
        num_nodes = self.num_objects
        num_edges = self.np_random.integers(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        # if graph is None:
        self.adjacency_matrix = random_dag(num_nodes, num_edges, self.np_random)
        # else:
        #    self.adjacency_matrix = random_dag(num_nodes, num_nodes, self.np_random, g = graph)

        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()
        # self.set_graph(self.graph)

        # Generate masks so that each variable only receives input from its parents.
        self.generate_masks()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True
        self.actions_to_target = []

        self.objects = OrderedDict()
        # Randomize object position.
        fixed_object_to_position_mapping = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2), (1, 1), (1, 3), (3, 1), (3, 3),
                                            (0, 2)]
        while len(self.objects) < self.num_objects:
            idx = len(self.objects)
            # Re-sample to ensure objects don't fall on same spot.
            # while not (idx in self.objects and self.valid_pos(self.objects[idx].pos, idx)):  # Chao commented
            while not (idx in self.objects):
                self.objects[idx] = Object(
                    pos=Coord(
                        # x=fixed_object_to_position_mapping[idx][0],  # Chao commented
                        # y=fixed_object_to_position_mapping[idx][1],
                        x=0,
                        y=0,
                    ),
                    color=torch.argmax(self.object_to_color[idx]))

        self.action_space = spaces.Discrete(self.num_actions)
        ### Chao commented start
        # self.observation_space = spaces.Box(
        #     low=0, high=1,
        #     shape=(3, 50, 50),
        #     dtype=np.float32
        # )
        ### Chao commented end

        ### Chao added start
        # Concatenate the current state and target state as a current observation
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(2 * self.num_objects * self.num_colors,),  # current and target obs
            dtype=int
        )
        ### Chao added end

        self.reward_baseline = 0
        self.cur_step = 0
        self.seed(seed)
        # self.reset()  # Chao commented
        # self.generate_target()  # Chao added

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load_save_information(self, save):
        self.adjacency_matrix = save['graph']
        for i in range(self.num_objects):
            self.mlps[i].load_state_dict(save['mlp' + str(i)])
        self.generate_masks()
        self.reset()

    def set_graph(self, g):
        if g in graphs.keys():
            # if int(g[-1]) != self.num_objects:
            #    print('ERROR:Env created for ' + str(self.num_objects) + ' objects while graph specified for ' + g[-1] + ' objects')
            #    exit()
            print('INFO: Loading predefined graph for configuration ' + str(g))
            g = graphs[g]
        num_nodes = self.num_objects
        # num_edges = self.np_random.integers(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        num_edges = self.np_random.integers(num_nodes, (((num_nodes) * (num_nodes - 1)) // 2) + 1)
        self.adjacency_matrix = random_dag(num_nodes, num_edges, self.np_random, g=g)
        self.adjacency_matrix = torch.from_numpy(self.adjacency_matrix).float()
        print(self.adjacency_matrix)
        self.generate_masks()
        self.reset()

    def get_save_information(self):
        save = {}
        save['graph'] = self.adjacency_matrix
        for i in range(self.num_objects):
            save['mlp' + str(i)] = self.mlps[i].state_dict()
        return save

    def render(self):
        return self.get_state()[0]  # Chao added
        # return np.concatenate((dict(
        #     grid=self.render_grid,
        #     circles=self.render_circles,
        #     shapes=self.render_shapes,
        #     cubes=self.render_cubes,
        # )[self.render_type](), dict(
        #     grid=self.render_grid_target,
        #     circles=self.render_circles_target,
        #     shapes=self.render_shapes_target,
        #     cubes=self.render_cubes,
        # )[self.render_type]()), axis=0)  # Chao commented

    def get_state(self):
        # print('OBJECT_TO_COLOR_TARGET IN GET_STATE METHOD')
        # print(self.object_to_color_target)

        # im = np.zeros(
        #      (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)  # Chao commented
        state_vec = np.zeros(self.num_objects * self.num_colors, dtype=np.int64)  # Chao added
        # im_target = np.zeros(
        #     (self.num_objects * self.num_colors, self.width, self.height), dtype=np.int32)  # Chao commented
        state_vec_target = np.zeros(self.num_colors * self.num_objects, dtype=np.int64)  # Chao added

        for idx, obj in self.objects.items():
            # im[idx * self.num_colors + obj.color, obj.pos.x, obj.pos.y] =  1  # Chao commented
            state_vec[idx * self.num_colors + obj.color] = 1  # Chao added
            # im_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item(), obj.pos.x, obj.pos.y] =  1  # Chao commented
            state_vec_target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item()] = 1  # Chao added

        # return im, im_target  # Chao commented
        return state_vec, state_vec_target  # Chao added

    def generate_masks(self):
        mask = self.adjacency_matrix.unsqueeze(-1)
        mask = mask.repeat(1, 1, self.num_colors)
        self.mask = mask.view(self.adjacency_matrix.size(0),
                              -1)  # For each node, mask the colors (num_colors) of its parent nodes
        # print('MASK')
        # print(self.mask)
        # # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # #         [1., 1., 1., 0., 0., 0., 0., 0., 0.],
        # #         [0., 0., 0., 1., 1., 1., 0., 0., 0.]]) for chain3; num_objs * (num_objs*num_colors)

    def generate_target(self, num_steps=1):
        self.actions_to_target = []
        for i in range(num_steps):
            intervention_id = self.np_random.integers(0, self.num_objects - 1)  # Intervened object id
            # to_color = self.np_random.integers(0, self.num_colors - 1)  # Color id assigned to the intervened object
            # self.actions_to_target.append(intervention_id * self.num_colors + to_color)  # [0, num_objs*num_colors)
            self.actions_to_target.append(intervention_id)  # [0, num_objs*num_colors)
            ########################################################
            # print(f"ACTIONS_TO_TARGET AT STEP {i}")
            # print(self.actions_to_target[-1])
            ########################################################
            # while to_color == torch.argmax(self.object_to_color[intervention_id]):
            #    to_color = self.np_random.integers(0, self.num_colors - 1)

            ### Chao commented the following two lines to keep the color of intervened object unchanged
            # self.object_to_color_target[intervention_id] = torch.zeros(self.num_colors)
            # self.object_to_color_target[intervention_id][to_color] = 1

            self.sample_variables_target(intervention_id)
            ########################################################
            # print(f"OBJECT_TO_COLOR_TARGET AT STEP {i}")
            # print(self.object_to_color_target, '\n')
            ########################################################

        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches += 1
        self.reward_baseline = matches / self.num_objects

        # print('OBJECT_TO_COLOR_TARGET IN GENERATE_TARGET METHOD')
        # print(self.object_to_color_target)  # [tensor([1., 0., 0., 0., 0.]), tensor([0., 1., 0., 0., 0.]),
        # # tensor([1., 0., 0., 0., 0.]), tensor([0., 0., 1., 0., 0.]), tensor([1., 0., 0., 0., 0.])]

    def check_softmax(self):
        s_ = []
        print('OBJECT_TO_COLOR IN CHECK_SOFTMAX')
        print(self.object_to_color)
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color, dim=0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            # _, s = self.mlps[i](x, mask, return_softmax=True)
            s = self.mlps[i](x, mask)
            s_.append(s.detach().cpu().numpy().tolist())
        print('CHECK_MLP')
        print(s_, '\n')
        return s_

    def check_softmax_target(self):
        s_ = []
        print('OBJECT_TO_COLOR_TARGET IN CHECK_SOFTMAX_TARGET')
        print(self.object_to_color_target)
        for i in range(1, len(self.objects)):
            x = torch.cat(self.object_to_color_target, dim=0).unsqueeze(0)
            mask = self.mask[i].unsqueeze(0)
            # _, s = self.mlps[i](x, mask, return_softmax=True)
            s = self.mlps[i](x, mask)
            s_.append(s.detach().cpu().numpy().tolist())
        print('CHECK_MLP_TARGET')
        print(s_)
        return s_

    def reset(self, num_steps=1, graph=None):
        self.cur_step = 0
        num_steps = self.max_steps  # Chao added

        self.object_to_color = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]
        self.object_to_color_target = [torch.zeros(self.num_colors) for _ in range(self.num_objects)]

        ### Chao commented starts
        # # Sample color for root node randomly
        # root_color = self.np_random.integers(0, self.num_colors)
        # self.object_to_color[0][root_color] = 1
        #
        # # Sample color for other nodes using MLPs
        # self.sample_variables(0, do_everything=True)
        # if self.movement == 'Dynamic':
        #     self.objects = OrderedDict()
        #     # Randomize object position.
        #     while len(self.objects) < self.num_objects:
        #         idx = len(self.objects)
        #         # Re-sample to ensure objects don't fall on same spot.
        #         # while not (idx in self.objects and self.valid_pos(self.objects[idx].pos, idx)):  # Chao commented
        #         while not (idx in self.objects):
        #             self.objects[idx] = Object(
        #                 pos=Coord(
        #                     # x=self.np_random.choice(np.arange(self.width)),  # Chao commented
        #                     # y=self.np_random.choice(np.arange(self.height)),
        #                     x=0,
        #                     y=0,
        #                 ),
        #                 color=torch.argmax(self.object_to_color[idx]))
        ### Chao commented ends

        # Sample color for all nodes randomly, Chao added
        for i in range(self.num_objects):
            random_color = self.np_random.integers(0, self.num_colors)
            self.object_to_color[i][random_color] = 1

        for idx, obj in self.objects.items():
            obj.color = torch.argmax(self.object_to_color[
                                         idx])  # torch.argmax(): convert one-hot color to index that is assigned to self.objects[idx].color
        # print('SELF.OBJECTS')
        # print(self.objects, '\n')  # OrderedDict([(0, Object(pos=Coord(x=0, y=0), color=tensor(0))), (1, Object(pos=Coord(x=0, y=4), color=tensor(1))),
        # # (2, Object(pos=Coord(x=4, y=0), color=tensor(0))), (3, Object(pos=Coord(x=4, y=4), color=tensor(1))), (4, Object(pos=Coord(x=2, y=2), color=tensor(4)))])

        # self.sample_variables_target(0, do_everything = True)
        for i in range(len(self.object_to_color)):
            self.object_to_color_target[i][
                torch.argmax(self.object_to_color[i])] = 1  # set self.object_to_color_target = self.object_to_color

        # print('OBJECT_TO_COLOR INITIAL')
        # print(self.object_to_color)  # [tensor([0., 0., 0., 0., 1.]), tensor([0., 1., 0., 0., 0.]),
        # # tensor([0., 0., 0., 1., 0.]), tensor([0., 0., 0., 0., 1.]), tensor([0., 0., 0., 0., 1.])]
        ########################################################
        # print('OBJECT_TO_COLOR_TARGET INITIAL')
        # print(self.object_to_color_target)  # [tensor([0., 0., 0., 0., 1.]), tensor([0., 1., 0., 0., 0.]),
        ########################################################
        # tensor([0., 0., 0., 1., 0.]), tensor([0., 0., 0., 0., 1.]), tensor([0., 0., 0., 0., 1.])]

        self.generate_target(num_steps)
        # self.check_softmax()
        # self.check_softmax_target()

        # self.object_to_color_init = copy.deepcopy(self.object_to_color)  # Chao added to test

        # print("ACTIONS_TO_TARGET FINAL")
        # print(self.object_to_color_target)

        # observations = self.render()  # Chao commented
        # observation_in, observations_target = observations[:3, :, :], observations[3:, :, :]  # Chao commented
        state_in, state_target = self.get_state()  # state_in <- self.objects[idx].color; state_target <- self.object_to_color_target
        # print('STATE_TARGET')
        # print(state_target)  # [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0]
        # print('STATE_IN')
        # print(state_in)  # [0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0]
        # print("INITIAL OBS ENV")
        # print(np.concatenate((state_in, state_target)))

        # state_in = np.zeros(state_in.shape)  # Chao added; uncomment to hide all objs

        info = {}
        # return (state_in, observation_in), (state_target, observations_target)  # Chao commented
        return np.concatenate((state_in, state_target)), info  # Chao added
        # return state_in  # Chao added. Initial state of each episode

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos.x not in range(0, self.width):
            return False
        if pos.y not in range(0, self.height):
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def is_reachable(self, idx, reached):
        """
        whether idx is the child of reached
        """
        for r in reached:
            if self.adjacency_matrix[idx, r] == 1:
                return True
        return False

    def sample_variables(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        do_everything: run MLPs of all subsequent objects (idx+1 to num_objs) if true

        self.object_to_color: a list of n_objs elements, each of which is a 1D n_colors tensor
        self.mask: 2D tensor: num_objs*(num_objs*num_colors). Masking its parents for each object
        """
        reached = [idx]  # At each step only one state variable is intervened
        # for v in range(idx + 1, self.num_objects):  # Chao commented
        for v in range(self.num_objects):  # for ring structure
            if do_everything or self.is_reachable(v, reached):
                # reached.append(v)  # affect all descendants; if commented, only affect direct children
                inp = torch.cat(self.object_to_color, dim=0).unsqueeze(0)  # 2D tensor: 1*(num_objs*num_colors). Colors of all objects
                mask = self.mask[v].unsqueeze(0)  # mask the colors of v's parent nodes

                out = self.mlps[v](inp, mask)  # output the color for object v
                self.object_to_color[v] = out.squeeze(0)  # overwrite v's color stored in object_to_color

    def sample_variables_target(self, idx, do_everything=False):
        """
        idx: variable at which intervention is performed
        """
        reached = [idx]
        # for v in range(idx + 1, self.num_objects):  # Chao commented
        for v in range(self.num_objects):
            if do_everything or self.is_reachable(v, reached):
                # reached.append(v)  # Chao commented
                inp = torch.cat(self.object_to_color_target, dim=0).unsqueeze(0)
                mask = self.mask[v].unsqueeze(0)

                out = self.mlps[v](inp, mask)
                self.object_to_color_target[v] = None
                self.object_to_color_target[v] = out.squeeze(0)

    # def transition(self, obj_id, color_id):  # Chao commented
    def transition(self, obj_id):  # Chao added

        ### Chao commented the following three lines to keep the color of intervened object unchanged
        # color_ = torch.zeros(self.num_colors)
        # color_[color_id] = 1
        # self.object_to_color[obj_id] = color_

        self.sample_variables(obj_id)  # one-step transition in DAG
        for idx, obj in self.objects.items():
            obj.color = torch.argmax(self.object_to_color[idx])

    def step(self, action: int):
        """
        action: [0, num_objs)
        """
        # action = self.actions_to_target[self.cur_step]
        # obj_id = action // self.num_colors  # Intervened object id
        obj_id = action  # Intervened object id
        # color_id = action % self.num_colors  # Color id assigned to the intervened object

        # self.transition(obj_id, color_id)
        if obj_id in range(self.num_objects):
            self.transition(obj_id)

        matches = 0
        for c1, c2 in zip(self.object_to_color, self.object_to_color_target):
            if torch.argmax(c1).item() == torch.argmax(c2).item():
                matches += 1
        reward = 0
        # reward = 0
        # if matches == self.num_objects:
        #    reward = 1

        ### Chao commented start
        # state_obs = self.render()
        # state_obs = state_obs[:3, :, :]
        # state = self.get_state()[0]
        # state_obs = (state, state_obs)
        ### Chao commented end

        state, target = self.get_state()  # Chao added commented for test
        # state = np.zeros(state.shape)

        reward = matches / self.num_objects - self.reward_baseline
        self.reward_baseline = matches / self.num_objects

        # # Chao added for test
        # self.object_to_color = copy.deepcopy(self.object_to_color_init)
        # state = np.zeros(self.num_objects * self.num_colors, dtype=np.int32)
        # target = np.zeros(self.num_colors * self.num_objects, dtype=np.int32)
        # for idx, obj in self.objects.items():
        #     # state[idx * self.num_colors + torch.argmax(self.object_to_color[idx]).item()] = 1  # comment to hide all objs
        #     target[idx * self.num_colors + torch.argmax(self.object_to_color_target[idx]).item()] = 1
        ########################################################
        # print("ACTION ENV")
        # print(action)
        # print("NEXT OBS ENV")
        # print(self.object_to_color)
        # # print(np.concatenate((state, target)))
        # print("REWARD ENV")
        # print(reward)
        # print("\n")
        ########################################################
        self.cur_step += 1
        terminated = False
        truncated = False
        info = {}
        return np.concatenate((state, target)), reward, terminated, truncated, info  # Chao added
        # return state, reward, done, {}  # Chao added


