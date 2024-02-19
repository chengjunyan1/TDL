import os
import torch
from collections import Counter
from operator import itemgetter
import random
from random import sample
import numpy as np
import json
from collections import OrderedDict
import pickle
import matplotlib.pylab as plt
from functools import partial
import pprint
import hashlib
import copy
import sys
import math
from copy import deepcopy
import hashlib
import uuid 
import ast
# Baby-ARC related imports
#try:
from .constants import *
from .utils import *
from .objects import *
from .canvas import *
from .operator import *

from itertools import product
import re

from collections import namedtuple
# These specs are for complicated operators
OperatorMoveSpec = namedtuple("OperatorMoveSpec", "autonomous direction distance hit_type linkage_move linkage_move_distance_ratio")

class BabyARCDataset(object):
    """
    The main class of the BabyARC Dataset。
    TODO:
    1. Support different background color.
    """
    def __init__(
        self, pretrained_obj_cache, 
        data_dir=None, save_directory="./BabyARCDataset/", 
        object_limit=None,
        # below are basic statistics of the dataset
        dataset_background_color=0.0,
        relation_vocab = ["SameAll", "SameShape", "SameColor", 
                          "SameRow", "SameCol", 
                          "IsInside", "IsTouch"],
        num_pool = [2,3,4],
        sparse_prob = 0.3,
        noise_level = 1, 
        canvas_size = None,
        debug = False,
        skip_load_pretrain_obj=False,
    ): # if canvas size is provided, square shape canvas is sampled.
        if data_dir == None:
            if skip_load_pretrain_obj:
                self.ObE = ObjectEngine(background_c=dataset_background_color)
            else:
                self.training_objs = torch.load(pretrained_obj_cache)
                if debug:
                    logger.info("Creating new BabyARC dataset by loading in pretrained objects.")
                    logger.info(f"Loading the object engine and canvas engine with "
                                f"a limit of object number {object_limit}, "
                                f"background_color={int(dataset_background_color)}.")
                if object_limit == 0:
                    if debug:
                        logger.info("WARNING: 0 object requested from pretrained objects file. Overwrite to 1 for safety.")
                    object_limit = 1 # overwrite to 1 for simplicity
                if object_limit:
                    self.ObE = ObjectEngine(self.training_objs[:object_limit], 
                                            background_c=dataset_background_color)
                else:
                    self.ObE = ObjectEngine(self.training_objs, 
                                            background_c=dataset_background_color)
            
            self.CanvasE = CanvasEngine(background_color=dataset_background_color)
            self.relation_vocab = relation_vocab
            self.num_pool = num_pool
            self.sparse_prob = sparse_prob
            self.noise_level = noise_level
        self.data_dir = data_dir # if it is None, we need to generate new dataset
        self.save_directory = save_directory
        if canvas_size:
            if debug:
                logger.info(f"Create BabyARC canvas with fixed width and height = {canvas_size}.")
        self.canvas_size = canvas_size
    
    def sample_single_core_edges(self):
        relation_num = random.choice(self.num_pool)
        edges = OrderedDict({ })
        relation_sampled = []
        for i in range(relation_num):
            relation_sampled += [random.choice(self.relation_vocab)]
            sorted(relation_sampled)
        current_obj_count = 0
        for rel in relation_sampled:
            if rel == "IsInside":
                # we need to propose two new objects
                edges[(f'obj_{current_obj_count}', f'obj_{current_obj_count+1}')] = rel
                current_obj_count += 2
            else:
                # if others we can decide whether to connect
                if current_obj_count == 0 or random.uniform(0, 1) < self.sparse_prob:
                    # sparse
                    edges[(f'obj_{current_obj_count}', f'obj_{current_obj_count+1}')] = rel
                    current_obj_count += 2
                else:
                    # select a old obj
                    ref_obj = random.randint(0,current_obj_count-1)
                    edges[(f'obj_{current_obj_count}', f'obj_{ref_obj}')] = rel
                    current_obj_count += 1
        return edges
    
    def sample_single_canvas_by_core_edges(
        self, edges, is_plot=True, 
        min_length=20, max_length=30, 
        allow_connect=False,
        rainbow_prob=0.2, 
        concept_collection=["line", "Lshape", "rectangle", "rectangleSolid", 
                            "randomShape", "arcShape", "Tshape", "Eshape", 
                            "Hshape", "Cshape", "Ashape", "Fshape"],
        parsing_check=False,
        color_avail=None,
        save_file=None,
        axis_off=False,
        large_shape=False,
        concept_limits={},
    ):
        relation_num = len(edges)
        nodes = OrderedDict({ })
        
        composite_obj_offset = 0
        has_composite_obj = False
        
        if self.canvas_size:
            test_canvas = CanvasEngine().sameple_canvas_by_size(min_length=self.canvas_size, 
                                                                max_length=self.canvas_size)[0]
        else:
            if relation_num >= 3:
                test_canvas = CanvasEngine().sameple_canvas_by_size(min_length=min_length, max_length=max_length)[0]
            else:
                test_canvas = CanvasEngine().sameple_canvas()[0]
        placement_result = -1 # default to non-placement
        placed_objs = set([])
        current_id = 0
        for edge, rel in edges.items():
            node_left = edge[0]
            node_right = edge[1]
            rel_n = rel

            # ok, we now support you specify some node attribute
            # ('obj_0', 'color_[1]', 'Attr')
            # ('obj_0', 'pos_[0,1]', 'Attr')
            if rel_n.startswith("Attr"):
                if node_left not in placed_objs:
                    if node_right.startswith("color"):
                        new_c = node_right.split("_")[-1]
                        new_c = ast.literal_eval(new_c)
                        new_c = int(new_c[0]) # get the color
                        obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        obj_refer = self.ObE.fix_color(obj_refer, new_color=new_c)
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("pos"):
                        new_p = node_right.split("_")[-1]
                        new_p = ast.literal_eval(new_p)
                        if len(new_p) != 2:
                            placement_result = -1
                            break
                        obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        placement_result = test_canvas.placement_position_fixed(obj_refer, new_p[0], new_p[1])
                        if placement_result == -1:
                            break
                    elif node_right.startswith("pixel"):
                        obj_refer = self.ObE.sample_objs_with_pixel(n=1)[0]
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("line"):
                        line_spec = node_right.split("_")[-1]
                        line_spec = ast.literal_eval(line_spec)
                        if line_spec[0] != -1:
                            len_lims=[line_spec[0], line_spec[0]]
                        else:
                            len_lims=[2, min(test_canvas.init_canvas.shape[0], test_canvas.init_canvas.shape[1])]
                        if line_spec[1] != -1:
                            thickness = line_spec[1]
                        else:
                            thickness = 1
                        if line_spec[2] == -1:
                            directions=["v", "h"]
                            direction = random.sample(directions, k=1)[0]
                        elif line_spec[2] == 0:
                            direction = "v"
                        elif line_spec[2] == 1:
                            direction = "h"
                        obj_refer = self.ObE.sample_objs_with_line(n=1, len_lims=len_lims, 
                                                                  thickness=thickness, 
                                                                  direction=direction,
                                                                  rainbow_prob=rainbow_prob)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("rectangle") and not node_right.startswith("rectangleSolid"):
                        rect_spec = node_right.split("_")[-1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]    
                        obj_refer = self.ObE.sample_objs_with_rectangle(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                         thickness=1, rainbow_prob=rainbow_prob)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("arcShape"):
                        rect_spec = node_right.split("_")[-1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]    
                        obj_refer = self.ObE.sample_objs_by_bound_area(
                            n=1, w_lim=rect_spec[0], h_lim=rect_spec[1], 
                            rainbow_prob=rainbow_prob, concept_collection=["arcShape"]
                        )[0]
                        if obj_refer == None:
                            placement_result = -1
                            break
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("randomShape"):
                        rect_spec = node_right.split("_")[-1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]    
                        obj_refer = self.ObE.sample_objs_with_random_shape(
                            n=1, w_lims=w_lims, h_lims=h_lims, 
                            rainbow_prob=rainbow_prob
                        )[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("rectangleSolid"):
                        rect_spec = node_right.split("_")[-1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]
                        obj_refer = self.ObE.sample_objs_with_rectangle_solid(
                            n=1, w_lims=w_lims, h_lims=h_lims,
                            rainbow_prob=rainbow_prob
                        )[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break   
                    elif node_right.startswith("Lshape"):
                        lshape_spec = node_right.split("_")[-1]
                        lshape_spec = ast.literal_eval(lshape_spec)
                        if lshape_spec[0] != -1:
                            w_lims = [lshape_spec[0], lshape_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if lshape_spec[1] != -1:
                            h_lims = [lshape_spec[1], lshape_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]   
                        if lshape_spec[2] == -1:
                            directions=[0, 1, 2, 3]
                            direction = random.sample(directions, k=1)[0]
                        elif lshape_spec[2] == 0 or lshape_spec[2] == 1 or lshape_spec[2] == 2 or lshape_spec[2] == 3:
                            direction = lshape_spec[2]
                        else:
                            raise NotImplementedError("Invalid direction for Lshape")
                        obj_refer = self.ObE.sample_objs_with_l_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                      thickness=1, rainbow_prob=rainbow_prob, 
                                                                      direction=direction)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("Tshape") or \
                        node_right.startswith("Eshape") or \
                        node_right.startswith("Hshape") or \
                        node_right.startswith("Cshape") or \
                        node_right.startswith("Ashape") or \
                        node_right.startswith("Fshape"):
                        lshape_spec = node_right.split("_")[-1]
                        lshape_spec = ast.literal_eval(lshape_spec)
                        if lshape_spec[0] != -1:
                            w_lims = [lshape_spec[0], lshape_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if lshape_spec[1] != -1:
                            h_lims = [lshape_spec[1], lshape_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]  
                        if node_right.startswith("Tshape"):
                            obj_refer = self.ObE.sample_objs_with_t_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        elif node_right.startswith("Eshape"):
                            obj_refer = self.ObE.sample_objs_with_e_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        elif node_right.startswith("Hshape"):
                            obj_refer = self.ObE.sample_objs_with_h_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        elif node_right.startswith("Cshape"):
                            obj_refer = self.ObE.sample_objs_with_c_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        elif node_right.startswith("Ashape"):
                            obj_refer = self.ObE.sample_objs_with_a_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        elif node_right.startswith("Fshape"):
                            obj_refer = self.ObE.sample_objs_with_f_shape(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                          rainbow_prob=rainbow_prob)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.split("_")[0] in {"RectE1a", "RectE1b", "RectE1c", 
                       "RectE2a", "RectE2b", "RectE2c",
                       "RectE3a", "RectE3b", "RectE3c", 
                       "RectF1a", "RectF1b", "RectF1c", 
                       "RectF2a", "RectF2b", "RectF2c",
                       "RectF3a", "RectF3b", "RectF3c",}:
                        composite_obj_offset += 2
                        has_composite_obj = True
                        lshape_spec = node_right.split("_")[-1]
                        lshape_spec = ast.literal_eval(lshape_spec)
                        if lshape_spec[0] != -1:
                            w_lims = [lshape_spec[0], lshape_spec[0]]
                        else:
                            w_lims = [test_canvas.init_canvas.shape[1], test_canvas.init_canvas.shape[1]]
                        if lshape_spec[1] != -1:
                            h_lims = [lshape_spec[1], lshape_spec[1]]
                        else:
                            h_lims = [test_canvas.init_canvas.shape[1], test_canvas.init_canvas.shape[1]]
                        obj_refer = self.ObE.sample_objs_with_composite_shape(
                            n=1, w_lims=w_lims, h_lims=h_lims, 
                            rainbow_prob=rainbow_prob, 
                            chosen_concept=node_right.split("_")[0],
                            n_retry=30,
                            allow_connect=allow_connect,
                            parsing_check=parsing_check,
                            color_avail=color_avail,
                        )
                        if obj_refer[0] == None:
                            placement_result = -1
                            break
                        placement_result = test_canvas.placement(
                            obj_refer[0],
                        ) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("symmetry"):
                        rect_spec = node_right.split("_")[1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            w_lims = [2, test_canvas.init_canvas.shape[1]-1]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            h_lims = [2, test_canvas.init_canvas.shape[0]-1]
                        axis_spec = node_right.split("_")[-1]
                        axis_spec = ast.literal_eval(axis_spec)
                        obj_refer = self.ObE.sample_objs_with_symmetry_shape(
                            n=1, w_lims=w_lims, h_lims=h_lims,
                            rainbow_prob=rainbow_prob, axis_list=axis_spec)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False, 
                                                                 connect_allow=allow_connect) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("enclosure"):
                        rect_spec = node_right.split("_")[-1]
                        rect_spec = ast.literal_eval(rect_spec)
                        if rect_spec[0] != -1:
                            w_lims = [rect_spec[1], rect_spec[1]]
                        else:
                            w_lims = [4, test_canvas.init_canvas.shape[1]]
                        if rect_spec[1] != -1:
                            h_lims = [rect_spec[0], rect_spec[0]]
                        else:
                            h_lims = [4, test_canvas.init_canvas.shape[0]]
                        if rect_spec[-1] == -1:
                            gravities = [True, False]
                            gravity = random.sample(gravities, k=1)[0]
                        elif rect_spec[-1] == 0:
                            gravity = False
                        else: 
                            gravity = True
                        obj_refer = self.ObE.sample_objs_with_enclosure(n=1, w_lims=w_lims, h_lims=h_lims, 
                                                                   thickness=1, rainbow_prob=rainbow_prob, 
                                                                   gravity=gravity)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(obj_refer, consider_tag=False) # place old obj with free pos
                        if placement_result == -1:
                            break
                    elif node_right.startswith("boundary_attachment"):
                        # sample the object and move the desired boundary
                        boundary_spec = node_right.split("_")[-1]
                        boundary_spec = ast.literal_eval(boundary_spec)
                        if boundary_spec[0] == -1:
                            boundary = random.randint(0,3)
                        else:
                            boundary = boundary_spec[0]
                        obj_refer = self.ObE.sample_objs(n=1, is_plot=False)[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        canvas_w = test_canvas.init_canvas.shape[0]
                        canvas_h = test_canvas.init_canvas.shape[1]
                        if boundary == 0:
                            random_pos = [i for i in range(0, canvas_w)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.placement_position_fixed(obj_refer, 0, 
                                                                         i)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 1:
                            random_pos = [i for i in range(0, canvas_h)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.placement_position_fixed(obj_refer, 
                                                                         i, 
                                                                         canvas_w-obj_refer.image_t.shape[1])
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 2:
                            random_pos = [i for i in range(0, canvas_w)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.placement_position_fixed(obj_refer, 
                                                                         canvas_h-obj_refer.image_t.shape[0],
                                                                         i)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 3:
                            random_pos = [i for i in range(0, canvas_h)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.placement_position_fixed(obj_refer, 
                                                                         i, 
                                                                         0)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        else:
                            placement_result = -1
                            break
                    else:
                        placement_result = -1
                        break # everything else we are not supported!

                    nodes[node_left] = current_id
                    current_id += 1
                    placed_objs.add(node_left)
                else:
                    if node_right.startswith("color"):
                        # just need to override the color i think
                        new_c = node_right.split("_")[-1]
                        new_c = ast.literal_eval(new_c)
                        new_c = int(new_c[0]) # get the color
                        placement_result = test_canvas.change_obj_color(nodes[node_left], new_c)
                        if placement_result == -1:
                            break
                    elif node_right.startswith("pos"):
                        # just need to override the position i think
                        new_p = node_right.split("_")[-1]
                        new_p = ast.literal_eval(new_p)
                        if len(new_p) != 2:
                            placement_result = -1
                            break
                        placement_result = test_canvas.change_obj_pos(nodes[node_left], new_p[0], new_p[1])
                        if placement_result == -1:
                            break
                    elif node_right.startswith("boundary_attachment"):
                        obj_refer = test_canvas.get_obj(nodes[node_left])
                        # move and attach to the desired boundary
                        boundary_spec = node_right.split("_")[-1]
                        boundary_spec = ast.literal_eval(boundary_spec)
                        if boundary_spec[0] == -1:
                            boundary = random.randint(0,3)
                        else:
                            boundary = boundary_spec[0]
                        canvas_h = test_canvas.init_canvas.shape[0]
                        canvas_w = test_canvas.init_canvas.shape[1]
                        if boundary == 0:
                            random_pos = [i for i in range(0, canvas_w)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.change_obj_pos(nodes[node_left], 0, 
                                                               i)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 1:
                            random_pos = [i for i in range(0, canvas_h)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.change_obj_pos(nodes[node_left], 
                                                               i, 
                                                               canvas_w-obj_refer.image_t.shape[1])
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 2:
                            random_pos = [i for i in range(0, canvas_w)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.change_obj_pos(nodes[node_left], 
                                                               canvas_h-obj_refer.image_t.shape[0],
                                                               i)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        elif boundary == 3:
                            random_pos = [i for i in range(0, canvas_h)]
                            random.shuffle(random_pos)
                            for i in random_pos:
                                placement_result = \
                                    test_canvas.change_obj_pos(nodes[node_left], 
                                                               i, 
                                                               0)
                                if placement_result != -1:
                                    break
                            if placement_result == -1:
                                break
                        else:
                            placement_result = -1
                            break
                    else:
                        placement_result == -1 # we currently dont support this DSL
                        break
                continue

            # check number of new nodes in the proposal
            # some relations require two new nodes come with it
            # some onlt require one

            new_node_count = 0
            if node_left not in placed_objs:
                new_node_count += 1
            if node_right not in placed_objs:
                new_node_count += 1
            
            if new_node_count == 0:
                """
                Limitations:
                We only support non-cycle dependencies.
                A->B->C-> ...
                A->B; A->C
                Above patterns are fine.
                A->B->A-> ...
                A->B; B->A
                Above patterns are NOT fine.
                """
                
                # WARNING: this new are pretentious. We are essentially adjusting
                # features of the node_new in this scope.
                node_new = node_right
                node_old = node_left
                obj_new = test_canvas.get_obj(nodes[node_new])
                obj_old = test_canvas.get_obj(nodes[node_old])
                if rel_n == "IsNonOverlapXY":
                    assert False
                elif rel_n == "IsOutside":
                    placement_result = test_canvas.placement(
                        obj_new, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule="IsInside", 
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "IsInside":
                    # the new obj is the outside obj
                    # this is to place the object inside referring to the outside object
                    out_obj = obj_new
                    placement_result = test_canvas.placement(
                        out_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule="IsOutside", 
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "SameAll":
                    left_map = obj_new.image_t
                    right_map = obj_old.image_t
                    if torch.equal(left_map, right_map):
                        pass
                    else:
                        placement_result = -1
                        break
                elif rel_n == "SameRow":
                    in_obj = obj_new
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n, 
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "SameCol":
                    in_obj = obj_new
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n,
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "IsTouch":
                    in_obj = obj_new
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n,
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "SameShape":
                    left_map = obj_new.image_t.bool()
                    right_map = obj_old.image_t.bool()
                    if torch.equal(left_map, right_map):
                        pass
                    else:
                        placement_result = -1
                        break
                elif rel_n == "SameColor":
                    new_c = test_canvas.unify_color(nodes[node_old])
                    obj_new = self.ObE.fix_color(obj_new, new_color=new_c)
                    placement_result = test_canvas.placement(
                        obj_new, 
                        placement_rule="SameColor", 
                        consider_tag=False,
                        connect_allow=allow_connect,
                        # in_place placement
                        in_place=True, 
                        to_placement_obj_id=nodes[node_new]
                    )
                    if placement_result == -1:
                        break
                    
            elif new_node_count == 2:
                if rel_n == "IsEnclosed":
                    # UPDATE STATUS: DONE
                    # sample a outside reactangle
                    if large_shape:
                        w_lims = [4, test_canvas.init_canvas.shape[1]]
                        h_lims = [4, test_canvas.init_canvas.shape[0]]
                    else:
                        w_lims = [4, max(4,test_canvas.init_canvas.shape[1]//2)]
                        h_lims = [4, max(4,test_canvas.init_canvas.shape[0]//2)]
                    # this is to place the object inside referring to the outside object
                    out_obj = self.ObE.sample_objs_with_rectangle(
                        n=1, thickness=1, rainbow_prob=rainbow_prob,
                        w_lims=w_lims, h_lims=h_lims,
                        concept_limits=concept_limits,
                    )[0] 
                    if color_avail:
                        # We can sample color now based on color collection.
                        out_obj = self.ObE.fix_color(out_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(out_obj)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    in_obj = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=out_obj.image_t.shape[1]-2, h_lim=out_obj.image_t.shape[0]-2,
                        concept_collection=concept_collection,
                        concept_limits=concept_limits,
                    )
                    if in_obj == None or len(in_obj) < 1 or in_obj[0] == None:
                        placement_result = -1
                        break
                    in_obj = in_obj[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        in_obj, to_relate_objs=[nodes[node_left]], 
                        placement_rule="IsInside", connect_allow=allow_connect
                    )
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "IsNonOverlapXY":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection,
                        concept_limits=concept_limits,
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        obj_refer = self.ObE.sample_objs_by_bound_area(
                            n=1, rainbow_prob=rainbow_prob, 
                            w_lim=w_lim, h_lim=h_lim,
                            concept_collection=concept_collection,
                            concept_limits=concept_limits,
                        )
                        if obj_refer == None or len(obj_refer) < 1 or obj_refer[0] == None:
                            placement_result = -1
                            break
                        obj_refer = obj_refer[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            obj_refer, to_relate_objs=[nodes[node_left]], 
                            placement_rule=rel_n, 
                            connect_allow=allow_connect
                        )
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "IsInside":
                    # UPDATE STATUS: DONE
                    # sample a outside reactangle
                    if large_shape:
                        w_lims = [4, test_canvas.init_canvas.shape[1]]
                        h_lims = [4, test_canvas.init_canvas.shape[0]]
                    else:
                        w_lims = [4, max(4,test_canvas.init_canvas.shape[1]//2)]
                        h_lims = [4, max(4,test_canvas.init_canvas.shape[0]//2)]
                    # this is to place the object inside referring to the outside object
                    out_obj = self.ObE.sample_objs_with_rectangle(
                        n=1, thickness=1, rainbow_prob=rainbow_prob,
                        w_lims=w_lims, h_lims=h_lims,
                        concept_limits=concept_limits,
                    )[0] 
                    if color_avail:
                        # We can sample color now based on color collection.
                        out_obj = self.ObE.fix_color(out_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(out_obj)
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                    in_obj = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=out_obj.image_t.shape[1]-2, h_lim=out_obj.image_t.shape[0]-2,
                        concept_collection=concept_collection,
                        concept_limits=concept_limits,
                    )
                    if in_obj == None or len(in_obj) < 1 or in_obj[0] == None:
                        placement_result = -1
                        break
                    in_obj = in_obj[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        in_obj, to_relate_objs=[nodes[node_right]], 
                        placement_rule="IsInside", connect_allow=allow_connect
                    )
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                elif rel_n == "SameAll":
                    # UPDATE STATUS: DONE
                    # obj_new = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    # let us restrict this a little bit, so there are space for other objects
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_new = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_new == None or len(obj_new) < 1 or obj_new[0] == None:
                        placement_result = -1
                        break
                    obj_new = obj_new[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_new = self.ObE.fix_color(obj_new, random.choice(color_avail))
                    obj_new_copy = copy.deepcopy(obj_new)
                    placement_result = test_canvas.placement(obj_new)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    placement_result = test_canvas.placement(obj_new_copy, connect_allow=allow_connect)
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "SameRow":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        obj_refer = self.ObE.sample_objs_by_fixed_height(
                            n=1, rainbow_prob=rainbow_prob, 
                            height=obj_anchor.image_t.shape[0], w_lim=w_lim, 
                            concept_collection=concept_collection
                        )
                        if obj_refer == None or len(obj_refer) < 1 or obj_refer[0] == None:
                            placement_result = -1
                            break
                        obj_refer = obj_refer[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            obj_refer, to_relate_objs=[nodes[node_left]], 
                            placement_rule=rel_n, 
                            connect_allow=allow_connect
                        )
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1

                elif rel_n == "SameCol":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        obj_refer = self.ObE.sample_objs_by_fixed_width(
                            n=1, rainbow_prob=rainbow_prob, 
                            width=obj_anchor.image_t.shape[1], h_lim=h_lim, 
                            concept_collection=concept_collection
                        )
                        if obj_refer == None or len(obj_refer) < 1 or obj_refer[0] == None:
                            placement_result = -1
                            break
                        obj_refer = obj_refer[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            obj_refer, to_relate_objs=[nodes[node_left]], 
                            placement_rule=rel_n, 
                            connect_allow=allow_connect
                        )
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "IsTouch":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        obj_refer = self.ObE.sample_objs_by_bound_area(
                            n=1, rainbow_prob=rainbow_prob, 
                            w_lim=w_lim, h_lim=h_lim,
                            concept_collection=concept_collection
                        )
                        if obj_refer == None or len(obj_refer) < 1 or obj_refer[0] == None:
                            placement_result = -1
                            break
                        obj_refer = obj_refer[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            obj_refer, to_relate_objs=[nodes[node_left]], 
                            placement_rule=rel_n, 
                            connect_allow=allow_connect
                        )
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "SameShape":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        obj_refer = self.ObE.random_color(
                            obj_anchor, rainbow_prob=rainbow_prob
                        )
                        if obj_refer == None:
                            placement_result = -1
                            break
                        if color_avail:
                            # We can sample color now based on color collection.
                            obj_refer = self.ObE.fix_color(obj_refer, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            obj_refer, placement_rule="SameShape", 
                            consider_tag=False, 
                            connect_allow=allow_connect

                        ) # place old obj with free pos
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                elif rel_n == "SameColor":
                    # UPDATE STATUS: DONE
                    # obj_anchor = self.ObE.sample_objs(n=1, is_plot=False)[0]
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)

                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_anchor = obj_anchor[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_anchor = self.ObE.fix_color(obj_anchor, random.choice(color_avail))
                    placement_result = test_canvas.placement(obj_anchor)
                    if placement_result == -1:
                        break
                    nodes[node_left] = current_id
                    current_id += 1
                    # this may fail, we wamt tp retry
                    amortize_retry = 5
                    for _ in range(amortize_retry):
                        new_c = test_canvas.unify_color(nodes[node_left])
                        # random get an object
                        obj_refer = self.ObE.sample_objs_by_bound_area(
                            n=1, rainbow_prob=rainbow_prob, 
                            w_lim=w_lim, h_lim=h_lim,
                            concept_collection=concept_collection
                        )
                        if obj_refer == None or len(obj_refer) < 1 or obj_refer[0] == None:
                            placement_result = -1
                            break
                        obj_refer = obj_refer[0]
                        obj_refer = self.ObE.fix_color(obj_refer, new_color=new_c)
                        placement_result = test_canvas.placement(
                            obj_refer, placement_rule="SameColor", 
                            consider_tag=False, 
                            connect_allow=allow_connect
                        ) # place old obj with free pos
                        if placement_result != -1:
                            break
                    if placement_result == -1:
                        break
                    nodes[node_right] = current_id
                    current_id += 1
                placed_objs.add(node_left)
                placed_objs.add(node_right)
            elif new_node_count == 1:
                # UPDATE STATUS: NOTSTART for this whole branch, it is probably outdated!
                # we only need to place the new obj
                node_new = node_left if node_left not in placed_objs else node_right
                node_old = node_left if node_new == node_right else node_right
                obj_old = test_canvas.get_obj(nodes[node_old])
                if rel_n == "IsNonOverlapXY":
                    assert False
                elif rel_n == "IsInside":
                    if node_new == node_left:
                        # this is unlikely to succeed as the outer obj is randomly sampled
                        # from our object pool.
                        # the new obj is the inside obj
                        in_obj = self.ObE.sample_objs_by_bound_area(
                            n=1, rainbow_prob=rainbow_prob, 
                            w_lim=3, h_lim=3,
                            concept_collection=concept_collection,
                        )
                        if in_obj == None or len(in_obj) < 1 or in_obj[0] == None:
                            placement_result = -1
                            break
                        in_obj = in_obj[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            in_obj, 
                            to_relate_objs=[nodes[node_old]], 
                            placement_rule="IsInside", 
                            connect_allow=allow_connect
                        )
                    else:
                        # the new obj is the outside obj
                        # this is to place the object inside referring to the outside object
                        out_obj = self.ObE.sample_objs_with_rectangle(
                            n=1, thickness=1, rainbow_prob=rainbow_prob, 
                            w_lims=[3,8], h_lims=[3,8]
                        )
                        if out_obj == None or len(out_obj) < 1 or out_obj[0] == None:
                            placement_result = -1
                            break
                        out_obj = out_obj[0]
                        if color_avail:
                            # We can sample color now based on color collection.
                            out_obj = self.ObE.fix_color(out_obj, random.choice(color_avail))
                        placement_result = test_canvas.placement(
                            out_obj, 
                            to_relate_objs=[nodes[node_old]], 
                            placement_rule="IsOutside", 
                            connect_allow=allow_connect
                        )

                    if placement_result == -1:
                        break
                elif rel_n == "SameAll":
                    placement_result = test_canvas.placement(
                        obj_old, 
                        placement_rule="SameAll", 
                        consider_tag=False, 
                        connect_allow=allow_connect
                    ) # place old obj with free pos
                    if placement_result == -1:
                        break
                elif rel_n == "SameRow":
                    # we need to actually make sure height is the same
                    height_old = obj_old.image_t.shape[0]
                    in_obj = self.ObE.sample_objs_by_fixed_height(
                        n=1, rainbow_prob=rainbow_prob, 
                        height=height_old, w_lim=5,
                        concept_collection=concept_collection,
                    )
                    if in_obj == None or len(in_obj) < 1 or in_obj[0] == None:
                        placement_result = -1
                        break
                    in_obj = in_obj[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n, 
                        connect_allow=allow_connect
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "SameCol":
                    # we need to actually make sure width
                    width_old = obj_old.image_t.shape[1]
                    in_obj = self.ObE.sample_objs_by_fixed_width(
                        n=1, rainbow_prob=rainbow_prob, width=width_old, h_lim=5,
                        concept_collection=concept_collection,
                    )
                    if in_obj == None or len(in_obj) < 1 or in_obj[0] == None:
                        placement_result = -1
                        break
                    in_obj = in_obj[0]
                    if color_avail:
                        # We can sample color now based on color collection.
                        in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n,
                        connect_allow=allow_connect
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "IsTouch":
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    in_obj = obj_anchor[0]                    

                    if color_avail:
                        # We can sample color now based on color collection.
                        in_obj = self.ObE.fix_color(in_obj, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        in_obj, 
                        to_relate_objs=[nodes[node_old]], 
                        placement_rule=rel_n,
                        connect_allow=allow_connect
                    )
                    if placement_result == -1:
                        break
                elif rel_n == "SameShape":
                    obj_new = self.ObE.random_color(
                        obj_old, rainbow_prob=rainbow_prob
                    )
                    if color_avail:
                        # We can sample color now based on color collection.
                        obj_new = self.ObE.fix_color(obj_new, random.choice(color_avail))
                    placement_result = test_canvas.placement(
                        obj_new, 
                        placement_rule="SameShape", 
                        consider_tag=False,
                        connect_allow=allow_connect
                    ) # place old obj with free pos
                    if placement_result == -1:
                        break
                elif rel_n == "SameColor":
                    new_c = test_canvas.unify_color(nodes[node_old])
                    # random get an object
                    amortize_ratio = [1,2]
                    ratio = np.random.choice(amortize_ratio)
                    w_lim = int((test_canvas.init_canvas.shape[1]-1)/ratio)
                    if ratio == 1:
                        ratio = np.random.choice([2])
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    else:
                        h_lim = int((test_canvas.init_canvas.shape[0]-1)/ratio)
                    obj_anchor = self.ObE.sample_objs_by_bound_area(
                        n=1, rainbow_prob=rainbow_prob, 
                        w_lim=w_lim, h_lim=h_lim,
                        concept_collection=concept_collection
                    )
                    if obj_anchor == None or len(obj_anchor) < 1 or obj_anchor[0] == None:
                        placement_result = -1
                        break
                    obj_new = obj_anchor[0]   

                    obj_new = self.ObE.fix_color(obj_new, new_color=new_c)
                    placement_result = test_canvas.placement(
                        obj_new, 
                        placement_rule="SameColor", 
                        consider_tag=False,
                        connect_allow=allow_connect
                    ) # place old obj with free pos
                    if placement_result == -1:
                        break
                nodes[node_new] = current_id
                current_id += 1
                placed_objs.add(node_new)

        if placement_result == -1:
            return -1
                    
        for i in range(self.noise_level):
            # try to add one noise
            noise_obj = self.ObE.sample_objs(n=1, is_plot=False)[0]
            test_canvas.placement(noise_obj, consider_tag=False)
        
        if parsing_check:
            image_t, _, _ = test_canvas.render(is_plot=False)
            objs = find_connected_components_colordiff(image_t, is_diag=True, color=True)
            if has_composite_obj:
                if len(objs) != (len(placed_objs)+composite_obj_offset):
                    return -1
            else:
                if len(objs) != len(placed_objs):
                    return -1
        
        # check if all relations complied.
        ret_dict = test_canvas.repr_as_dict(nodes, edges)

        for edge, rel in edges.items():
            node_left = edge[0]
            node_right = edge[1]
            rel_n = rel
            
            # this is a special case, we don't have that.
            if not rel_n == "IsOutside":
                if edge not in ret_dict['partial_relation_edges'].keys():
                    return -1
                if not rel_n in ret_dict['partial_relation_edges'][edge]:
                    return -1

        if is_plot:
            test_canvas.render(save_file=save_file, axis_off=axis_off)
            
        return ret_dict

    def sample_single_task_canvas(self, retry=5, is_plot=True):
        edges = self.sample_single_core_edges()
        print(self.sample_single_canvas_by_core_edges(edges, retry=retry, is_plot=is_plot))
        
    def sample_task_canvas_from_arc(self, image_t, is_diag=True, color=True, is_plot=True, allow_modify=False, background_color=0):
        if not color:
            objs = find_connected_components(image_t, is_diag=is_diag)
        else:
            objs = find_connected_components_colordiff(image_t, is_diag=is_diag, color=color)
        # generate canvas
        test_canvas = CanvasEngine().sample_task_canvas_from_arc(image_t, objs)
        if is_plot:
            test_canvas.render()

        return test_canvas.repr_as_dict()
        