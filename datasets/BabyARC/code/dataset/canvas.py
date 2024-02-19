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
from .constants import *
from .utils import *
from .objects import *

class Canvas:
    """
    BabyARC uses canvas to place different relational
    objects. A single canvas can be seen as an instance
    of a single input in a task. It contains many information
    like object numbers, object shapes, etc.. It can manage
    object relations.
    """
    def __init__(self, init_canvas=None, repre_dict=None, arc_image_t=None, arc_parsed_objs=None):
        if arc_parsed_objs is not None:
            # we need to preload from parsed obj list
            self.init_canvas = torch.zeros(arc_image_t.shape[0], 
                                           arc_image_t.shape[1])
            self.oid_map = OrderedDict()
            self.opos_map = OrderedDict() # upper left corner position
            self.node_id_map = OrderedDict()
            self.id_node_map = OrderedDict()
            self.background_color = 0
            obj_idx = 0
            for obj in arc_parsed_objs:
                obj_image_t = obj[0]
                obj_pos = obj[1]
                self.oid_map[obj_idx] = Object(obj_image_t, position_tags=[])
                self.opos_map[obj_idx] = torch.tensor([obj_pos[0], obj_pos[1]])
                self.node_id_map[f"obj_{obj_idx}"] = obj_idx
                self.id_node_map[obj_idx] = f"obj_{obj_idx}"
                obj_idx += 1
            # extra fields we might need
            self.partial_relation_edges = self.parse_relations()
            self.image_t = arc_image_t
        elif repre_dict == None:
            self.init_canvas = init_canvas
            self.oid_map = OrderedDict()
            self.opos_map = OrderedDict() # upper left corner position
            self.background_color = 0

            # extra fields we might need
            self.partial_relation_edges = OrderedDict()
            self.node_id_map = OrderedDict()
            self.id_node_map = OrderedDict()
            self.image_t = None
        else:
            self.id_node_map = OrderedDict()
            self._load_as_dict(repre_dict)
        self.color_dict = {0: [0, 0, 0],
                          1: [0, 0, 1],
                          2: [1, 0, 0],
                          3: [0, 1, 0],
                          4: [1, 1, 0],
                          5: [.5, .5, .5],
                          6: [.5, 0, .5],
                          7: [1, .64, 0],
                          8: [0, 1, 1],
                          9: [.64, .16, .16],
                         }   
    
    def reconcile(self):
        updated_canvas, r_diff, c_diff = self.render(is_plot=False, minimum_cover=False)
        self.image_t = updated_canvas
        self.partial_relation_edges = self.parse_relations()
    
    def _load_as_dict(self, repre_dict_in):
        """
        loading a canvas from a dictionary file
        """
        repre_dict = copy.deepcopy(repre_dict_in)
        self.init_canvas = torch.zeros(repre_dict['image_t'].shape[0], 
                                       repre_dict['image_t'].shape[1])
        self.oid_map = repre_dict['id_object_map']
        # turing them into objects
        for k, v in self.oid_map.items():
            self.oid_map[k] = Object(v, position_tags=[])
        self.opos_map = repre_dict['id_position_map']
        # we need to turn this into multiedge cases
        self.partial_relation_edges = OrderedDict()
        for edge, relation in repre_dict['partial_relation_edges'].items():
            if edge not in self.partial_relation_edges:
                self.partial_relation_edges[edge] = [relation]
            else:
                self.partial_relation_edges[edge].append(relation)
        self.node_id_map = repre_dict['node_id_map']
        for k, v in self.node_id_map.items():
            self.id_node_map[v] = k
        self.image_t = repre_dict['image_t']
        self.background_color = repre_dict['background_color']
    
    def _get_obj_color(self, obj):
        return obj.image_t.unique().tolist()
    
    def unify_color(self, oid):
        # unify the color of an object, this is useful in creating SameColor relations
        current_obj = self.oid_map[oid]
        unique = current_obj.image_t.unique().tolist()
        
        if len(unique) == 2:
            if unique[0] == 0:
                return int(unique[1])
            elif unique[1] == 0:
                return int(unique[0])
            else:
                pass
        elif len(unique) == 1:
            return int(unique[0])

        new_c = randint_exclude(0,9,[self.background_color])
        for i in range(current_obj.image_t.shape[0]):
            for j in range(current_obj.image_t.shape[1]):
                if current_obj.image_t[i,j] != self.background_color:
                    current_obj.image_t[i,j] = new_c
        return new_c

    def get_obj(self, oid):
        return self.oid_map[oid]
    
    def _check_exclusive(self, p_r, p_c, connect_allow=False, to_placement_obj=None):
        if p_r == -1 or p_c == -1:
            return False
        
        # check oob
        if p_r+to_placement_obj.image_t.shape[0] > self.init_canvas.shape[0] or \
            p_c+to_placement_obj.image_t.shape[1] > self.init_canvas.shape[1]:
            return False
        
        to_render_img = self.generate_render()
        to_placement_mask = \
            self.generate_mask_inplace([[(p_r, p_c), to_placement_obj]], connect_allow=connect_allow)
        # print(to_render_img)
        # print(to_placement_mask)
        
        canvas_r = self.init_canvas.shape[0]
        canvas_c = self.init_canvas.shape[1]
        # connect_allow and (to_render_img[i,j].tolist() in self._get_obj_color(to_placement_obj) 
        for i in range(canvas_r):
            for j in range(canvas_c):
                if to_placement_mask[i,j] != 0 and \
                    to_render_img[i,j] != self.background_color:
                    return False
        
        return True
    
    def _propose_position_simple(self, o_r, o_c, p_r, p_c, 
                                 connect_allow=False, to_placement_obj=None, 
                                 early_stop=True):
        """
        this is an important helper to randomly locate an object
        with size (o_r, o_c) on this canvas without collide with
        existing objects on the canvas
        """
        # TODO: to support allow connected component placement
        potential_pool = []
        canvas_r = self.init_canvas.shape[0]
        canvas_c = self.init_canvas.shape[1]
        
        if p_r == -1 and p_c == -1:
            random_pos_iter = []
            for i in range((canvas_r-o_r)+1):
                for j in range((canvas_c-o_c)+1):
                    random_pos_iter.append((i,j))
            random.shuffle(random_pos_iter)
            for pos in random_pos_iter:
                i = pos[0]
                j = pos[1]
                if self._check_exclusive(i, j, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                    potential_pool.append((i, j))
                    return i, j
        else:
            if p_r == -1:
                random_pos_iter = []
                for i in range(canvas_r-o_r):
                        random_pos_iter.append(i)
                random.shuffle(random_pos_iter)
                for i in random_pos_iter:
                    if self._check_exclusive(i, p_c, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                        return i, p_c
            elif p_c == -1:
                random_pos_iter = []
                for i in range(canvas_c-o_c):
                        random_pos_iter.append(i)
                random.shuffle(random_pos_iter)
                for i in random_pos_iter:
                    if self._check_exclusive(p_r, i, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                        return p_r, i
        return -1, -1
    
    def _propose_position_same_row(self, o_r, o_c, p_r, p_c, rel_r, rel_c, 
                                   connect_allow=False, to_placement_obj=None, 
                                   early_stop=True):
        # sanity check
        if p_r != -1:
            if p_r != rel_r:
                return -1, -1

        potential_pool = []
        canvas_r = self.init_canvas.shape[0]
        canvas_c = self.init_canvas.shape[1]
        p_r = rel_r
        
        if p_c == -1:
            random_pos_iter = []
            for i in range(canvas_c-o_c):
                    random_pos_iter.append(i)
            random.shuffle(random_pos_iter)
            for i in random_pos_iter:
                if self._check_exclusive(p_r, i, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                    potential_pool.append((p_r, i))
                    return p_r, i
        return -1, -1

    def _propose_position_same_col(self, o_r, o_c, p_r, p_c, rel_r, rel_c, 
                                   connect_allow=False, to_placement_obj=None, 
                                   early_stop=True):
        # sanity check
        if p_c != -1:
            if p_c != rel_c:
                return -1, -1

        potential_pool = []
        canvas_r = self.init_canvas.shape[0]
        canvas_c = self.init_canvas.shape[1]
        p_c = rel_c
        
        if p_r == -1:
            random_pos_iter = []
            for i in range(canvas_r-o_r):
                    random_pos_iter.append(i)
            random.shuffle(random_pos_iter)
            for i in random_pos_iter:
                if self._check_exclusive(i, p_c, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                    return i, p_c
        return -1, -1
    
    def _propose_position_is_non_overlap_xy(
        self, o_r, o_c, p_r, p_c, rel_r, rel_c,
        rel_p_r, rel_p_c,
        connect_allow=False, to_placement_obj=None, 
        early_stop=True
    ):
        def intersect(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)
        # we do it differently here comparing to others.
        canvas_r = self.init_canvas.shape[0]
        canvas_c = self.init_canvas.shape[1]
        random_pos_iter = []
        for i in range(0, canvas_r-o_r):
            for j in range(0, canvas_c-o_c):
                rec1 = [i,j,i+o_r,j+o_c]
                rec2 = [rel_p_r,rel_p_c,rel_p_r+rel_r,rel_p_c+rel_c]
                if (intersect(rec1[0], rec1[2], rec2[0], rec2[2]) & \
                    intersect(rec1[1], rec1[3], rec2[1], rec2[3])):
                    continue
                else:
                    random_pos_iter.append((i,j))
                    
        for pos in random_pos_iter:
            i = pos[0]
            j = pos[1]
            if self._check_exclusive(i, j, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                return i, j
        return -1, -1
    
    def _propose_position_is_inside(self, o_r, o_c, p_r, p_c, rel_r, rel_c,
                                    rel_p_r, rel_p_c,
                                    connect_allow=False, to_placement_obj=None, 
                                    early_stop=True):
        
        potential_pool = []
        # only overlook the potential regions of interst
        # we cannot consider pos tag here
        random_pos_iter = []
        for i in range(rel_p_r, (rel_p_r+rel_r-o_r)+1):
            for j in range(rel_p_c, (rel_p_c+rel_c-o_c)+1):
                random_pos_iter.append((i,j))
        # print(o_r, o_c)
        # print(p_r, p_c)
        # print(rel_r, rel_c, rel_p_r, rel_p_c)
        # print(self.init_canvas.shape[0], self.init_canvas.shape[1])
        # print(random_pos_iter)
        random.shuffle(random_pos_iter)
        
        for pos in random_pos_iter:
            i = pos[0]
            j = pos[1]
            if self._check_exclusive(i, j, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                return i, j
                
        return -1, -1

    def _propose_position_is_outside(self, o_r, o_c, p_r, p_c, rel_r, rel_c,
                                    rel_p_r, rel_p_c,
                                    connect_allow=False, to_placement_obj=None, 
                                    early_stop=True):
        
        potential_pool = []
        # only overlook the potential regions of interst
        if o_r < rel_r + 2:
            return -1, -1 # this is not possibble
        # we cannot consider pos tag here
        random_pos_iter = []
        r_lower = max(rel_p_r-(o_r-rel_r), 0)
        c_lower = max(rel_p_c-(o_c-rel_c), 0)
        for i in range(rel_p_r+1, rel_p_r+rel_r-1):
            for j in range(rel_p_c+1, rel_p_c+rel_c-1):
                random_pos_iter.append((i,j))
        random.shuffle(random_pos_iter)
        for pos in random_pos_iter:
            i = pos[0]
            j = pos[1]
            if self._check_exclusive(i, j, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
                return i, j
        return -1, -1
    
    def _propose_position_is_touch(self, o_r, o_c, p_r, p_c, to_relate_obj=None,
                                    connect_allow=False, to_placement_obj=None, 
                                    early_stop=True):
        
        potential_pool = []
        rel_pos = self.opos_map[to_relate_obj]
        rel_p_r, rel_p_c = rel_pos[0], rel_pos[1]
        rel_obj = self.oid_map[to_relate_obj]
        rel_r, rel_c = rel_obj.image_t.shape[0], rel_obj.image_t.shape[1]
        
        # random iterators
        random_pos_iter = []
        for i in range(max(0,rel_p_r-o_r), min(rel_p_r+rel_r+1,self.init_canvas.shape[0]-o_r)):
            for j in range(max(0,rel_p_c-o_c), min(rel_p_c+rel_c+1,self.init_canvas.shape[1]-o_c)):
                random_pos_iter.append((i,j))
        random.shuffle(random_pos_iter)
        
        to_relate_mask = \
            self.generate_mask_inplace([[(rel_p_r, rel_p_c), rel_obj]], connect_allow=True)
        for pos in random_pos_iter:
            i = pos[0]
            j = pos[1]
            # cannot collide with any object, not just the one we relate to
            if self._check_exclusive(i,j,to_placement_obj=to_placement_obj, connect_allow=True):
                # and we need to just touch the edge
                to_placement_mask = \
                    self.generate_mask_inplace([[(i, j), to_placement_obj]], connect_allow=True)

                for a in range(to_placement_mask.shape[0]):
                    for b in range(to_placement_mask.shape[1]):
                        if to_placement_mask[a, b] != 0:
                            if to_relate_mask[a+1, b] != 0 or \
                                to_relate_mask[a-1, b] != 0 or \
                                to_relate_mask[a, b-1] != 0 or \
                                to_relate_mask[a, b+1] != 0:
                                # we found one
                                return i, j
        return -1, -1
        
    def _placement_strategy(self, c_r, c_c, o_r, o_c, position_tags, p_r, p_c, 
                            placement_rule=None, to_placement_obj=None, 
                            to_relate_objs=[], consider_tag=False, 
                            connect_allow=False):
        
        # this part still has some bugs, may not be used for now.
        # some cases we can bypass the pos tag
        if placement_rule not in ['IsInside', 'IsTouch', 'IsOutside', 'IsNonOverlapXY'] and consider_tag:
            # check any pos tags come with this object
            if "upper" in position_tags and \
                "lower" in position_tags:
                if c_r == o_r:
                    p_r = 0 # c is TBD
                else:
                    return -1 # OOB

            if "left" in position_tags and \
                "right" in position_tags:
                if c_c == o_c:
                    p_c = 0 # r is TBD
                else:
                    return -1 # OOB

            if p_r == -1 and "upper" in position_tags:
                p_r = 0
                if p_r + o_r > c_r:
                    return -1 # OOB
            if p_r == -1 and "lower" in position_tags:
                p_r = c_r - o_r
                if p_r < 0:
                    return -1 # OOB

            if p_c == -1 and "left" in position_tags:
                p_c = 0
                if p_c + o_c > c_c:
                    return -1 # OOB
            if p_c == -1 and "right" in position_tags:
                p_c = c_c - o_c
                if p_c < 0:
                    return -1 # OOB

        if p_r == -1 or p_c == -1:
            if placement_rule == None or \
                placement_rule == "SameAll" or \
                placement_rule == "SameShape" or \
                placement_rule == "SameColor":
                    p_r, p_c = self._propose_position_simple(o_r, o_c, p_r, p_c, 
                                                             to_placement_obj=to_placement_obj, 
                                                             connect_allow=connect_allow)
            else:
                # otherwise, we need to re-propose
                if placement_rule == "SameRow":
                    to_relate_obj = to_relate_objs[0]
                    rel_pos = self.opos_map[to_relate_obj]
                    rel_r, rel_c = rel_pos[0], rel_pos[1]
                    p_r, p_c = self._propose_position_same_row(o_r, o_c, p_r, p_c, rel_r, rel_c,
                                                               to_placement_obj=to_placement_obj, 
                                                               connect_allow=connect_allow)
                elif placement_rule == "SameCol":
                    to_relate_obj = to_relate_objs[0]
                    rel_pos = self.opos_map[to_relate_obj]
                    rel_r, rel_c = rel_pos[0], rel_pos[1]
                    p_r, p_c = self._propose_position_same_col(o_r, o_c, p_r, p_c, rel_r, rel_c,
                                                               to_placement_obj=to_placement_obj, 
                                                               connect_allow=connect_allow)
                elif placement_rule == "IsInside":
                    to_relate_obj = to_relate_objs[0]
                    rel_pos = self.opos_map[to_relate_obj]
                    rel_p_r, rel_p_c = rel_pos[0], rel_pos[1]
                    rel_obj = self.oid_map[to_relate_obj]
                    rel_r, rel_c = rel_obj.image_t.shape[0], rel_obj.image_t.shape[1]
                    p_r, p_c = self._propose_position_is_inside(o_r, o_c, p_r, p_c, rel_r, rel_c,
                                                                rel_p_r, rel_p_c,
                                                                to_placement_obj=to_placement_obj, 
                                                                connect_allow=connect_allow)
                    
                elif placement_rule == "IsOutside":
                    to_relate_obj = to_relate_objs[0]
                    rel_pos = self.opos_map[to_relate_obj]
                    rel_p_r, rel_p_c = rel_pos[0], rel_pos[1]
                    rel_obj = self.oid_map[to_relate_obj]
                    rel_r, rel_c = rel_obj.image_t.shape[0], rel_obj.image_t.shape[1]
                    p_r, p_c = self._propose_position_is_outside(o_r, o_c, p_r, p_c, rel_r, rel_c,
                                                                 rel_p_r, rel_p_c,
                                                                 to_placement_obj=to_placement_obj, 
                                                                 connect_allow=connect_allow)
                elif placement_rule == "IsTouch":
                    connect_allow = True
                    to_relate_obj = to_relate_objs[0]
                    p_r, p_c = self._propose_position_is_touch(o_r, o_c, p_r, p_c,
                                                               to_relate_obj=to_relate_obj,
                                                                to_placement_obj=to_placement_obj)
                elif placement_rule == "IsNonOverlapXY":
                    to_relate_obj = to_relate_objs[0]
                    rel_pos = self.opos_map[to_relate_obj]
                    rel_p_r, rel_p_c = rel_pos[0], rel_pos[1]
                    rel_obj = self.oid_map[to_relate_obj]
                    rel_r, rel_c = rel_obj.image_t.shape[0], rel_obj.image_t.shape[1]
                    p_r, p_c = self._propose_position_is_non_overlap_xy(o_r, o_c, p_r, p_c, rel_r, rel_c,
                                                                rel_p_r, rel_p_c,
                                                                to_placement_obj=to_placement_obj, 
                                                                connect_allow=connect_allow)
                    
        
        if self._check_exclusive(p_r, p_c, to_placement_obj=to_placement_obj, connect_allow=connect_allow):
            return (p_r, p_c)

        return -1 # OOB
    
    def _placement_by_collision(self, merge_type="None"):
        pass
    
    def _check_position_valid(self, p_r, p_c):
        if p_r == -1 or p_c == -1:
            return False
        return True
    
    def _check_attach_corner(self, position_tags):
        if ("left" in position_tags and "upper" in position_tags) or \
            ("left" in position_tags and "lower" in position_tags) or \
            ("right" in position_tags and "upper" in position_tags) or \
            ("right" in position_tags and "lower" in position_tags):
            return True
        return False
    
    def placement_position_fixed(self, to_placement_obj, placement_r, placement_c, connect_allow=False):
        curr_obj = copy.deepcopy(to_placement_obj)

        if self._check_exclusive(placement_r, placement_c, 
                                 to_placement_obj=curr_obj, 
                                 connect_allow=connect_allow):
            self.oid_map[curr_obj_idx] = curr_obj
            self.opos_map[curr_obj_idx] = (placement_r, placement_c)
            return (placement_r, placement_c)
        return -1
    
    def placement(self, to_placement_obj, to_relate_objs=[], placement_rule=None, merge_type="None", 
                  consider_tag=True, connect_allow=False, in_place=False, to_placement_obj_id=None):
        
        if in_place:
            # there are some corner cases, we want to skip actual placement.
            # e.x. SameColor
            if placement_rule == "SameColor":
                curr_obj = copy.deepcopy(to_placement_obj)
                curr_obj_idx = to_placement_obj_id
                self.oid_map[curr_obj_idx] = curr_obj
                (placement_r, placement_c) = self.opos_map[curr_obj_idx]
                return (placement_r, placement_c)
            # if in_place, it means it is a existing object.
            curr_obj = copy.deepcopy(to_placement_obj)
            canvas_r = self.init_canvas.shape[0]
            canvas_c = self.init_canvas.shape[1]
            obj_r = to_placement_obj.image_t.shape[0]
            obj_c = to_placement_obj.image_t.shape[1]
            curr_obj_idx = to_placement_obj_id
            
            # remove from canvas for now.
            del self.oid_map[curr_obj_idx]
            del self.opos_map[curr_obj_idx]
        else:
            curr_obj = copy.deepcopy(to_placement_obj)
            canvas_r = self.init_canvas.shape[0]
            canvas_c = self.init_canvas.shape[1]
            obj_r = to_placement_obj.image_t.shape[0]
            obj_c = to_placement_obj.image_t.shape[1]
            curr_obj_idx = len(self.oid_map.keys())

        # the placement result r and c values
        placement_r = -1
        placement_c = -1
        placement_results = -1
        if len(self.oid_map.keys()) == 0 or \
            placement_rule == "SameAll" or \
            placement_rule == "SameColor" or \
            placement_rule == "SameShape" or \
            placement_rule == None:
            # canvas empty, only boundary should be fine
            placement_results = \
                self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                         curr_obj.position_tags, 
                                         placement_r, placement_c, 
                                         placement_rule=placement_rule,
                                         to_placement_obj=to_placement_obj, 
                                         consider_tag=consider_tag, 
                                         connect_allow=connect_allow)
        else:
            if placement_rule == "SameRow" or\
                placement_rule == "SameCol":
                placement_results = \
                    self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                             curr_obj.position_tags, 
                                             placement_r, placement_c, 
                                             placement_rule=placement_rule,
                                             to_placement_obj=to_placement_obj,
                                             to_relate_objs=to_relate_objs, 
                                             connect_allow=connect_allow)
            elif placement_rule == "SubsetOf":
                pass
            elif placement_rule == "IsInside":
                # we loop through to see if there is any possibilities
                placement_results = \
                    self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                             curr_obj.position_tags, 
                                             placement_r, placement_c, 
                                             placement_rule=placement_rule,
                                             to_placement_obj=to_placement_obj,
                                             to_relate_objs=to_relate_objs, 
                                             connect_allow=connect_allow)
            elif placement_rule == "IsOutside":
                # we loop through to see if there is any possibilities
                placement_results = \
                    self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                             curr_obj.position_tags, 
                                             placement_r, placement_c, 
                                             placement_rule=placement_rule,
                                             to_placement_obj=to_placement_obj,
                                             to_relate_objs=to_relate_objs, 
                                             connect_allow=connect_allow)
            elif placement_rule == "IsTouch":
                # we loop through to see if there is any possibilities
                placement_results = \
                    self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                             curr_obj.position_tags, 
                                             placement_r, placement_c, 
                                             placement_rule=placement_rule,
                                             to_placement_obj=to_placement_obj,
                                             to_relate_objs=to_relate_objs)
            elif placement_rule == "IsNonOverlapXY":
                # we loop through to see if there is any possibilities
                placement_results = \
                    self._placement_strategy(canvas_r, canvas_c, obj_r, obj_c, 
                                             curr_obj.position_tags, 
                                             placement_r, placement_c, 
                                             placement_rule=placement_rule,
                                             to_placement_obj=to_placement_obj,
                                             to_relate_objs=to_relate_objs, 
                                             connect_allow=connect_allow)
    
        if placement_results == -1:
            return -1
        placement_r, placement_c = placement_results
        self.oid_map[curr_obj_idx] = curr_obj
        self.opos_map[curr_obj_idx] = (placement_r, placement_c)
        return (placement_r, placement_c)

    def generate_render(self, obj_mask=None, is_plot=True):
        ret_canvas = self.init_canvas.clone()
        for oid in self.oid_map.keys():
            (r, c) = self.opos_map[oid]
            obj = self.oid_map[oid]
            image_t = obj.image_t
            for i in range(r, r+image_t.shape[0]):
                for j in range(c, c+image_t.shape[1]):
                    if ret_canvas[i,j] == self.background_color: # if it already has color, don't overwrite
                        ret_canvas[i,j] = image_t[i-r,j-c]
        return ret_canvas
    
    def check_conflict(self, obj_mask=None, is_plot=True, connect_allow=False):
        ret_canvas = self.init_canvas.clone()
        if connect_allow:
            for oid in self.oid_map.keys():
                (r, c) = self.opos_map[oid]
                obj = self.oid_map[oid]
                image_t = obj.image_t
                # some basic boundary checks here
                if r+image_t.shape[0] > ret_canvas.shape[0]:
                    return False
                if c+image_t.shape[1] > ret_canvas.shape[1]:
                    return False     

                for i in range(r, r+image_t.shape[0]):
                    for j in range(c, c+image_t.shape[1]):
                        if image_t[i-r, j-c] != 0:
                            if ret_canvas[i,j] == 0:
                                ret_canvas[i,j] = 1
                            else:
                                return False
        else:
            # the mask need to generated
            for oid in self.oid_map.keys():
                (r, c) = self.opos_map[oid]
                obj = self.oid_map[oid]
                image_t = obj.image_t
                # some basic boundary checks here
                if r+image_t.shape[0] > ret_canvas.shape[0]:
                    return False
                if c+image_t.shape[1] > ret_canvas.shape[1]:
                    return False     

                for i in range(r, r+image_t.shape[0]):
                    for j in range(c, c+image_t.shape[1]):
                        if image_t[i-r, j-c] != 0:
                            if ret_canvas[i,j] == 0:
                                # also look the bounds
                                ret_canvas[i,j] = 1
                            else:
                                return False
                
                # after check we mark boundarys
                for i in range(r, r+image_t.shape[0]):
                    for j in range(c, c+image_t.shape[1]):
                        if image_t[i-r, j-c] != 0:
                            # mark all bounds to be occupied as well
                            ret_canvas[max(i-1,0),j] = 1
                            ret_canvas[min(ret_canvas.shape[0]-1,i+1),j] = 1
                            ret_canvas[i,max(0,j-1)] = 1
                            ret_canvas[i,min(ret_canvas.shape[1]-1,j+1)] = 1
                            ret_canvas[max(i-1,0),max(0,j-1)] = 1
                            ret_canvas[max(i-1,0),min(ret_canvas.shape[1]-1,j+1)] = 1
                            ret_canvas[min(ret_canvas.shape[0]-1,i+1),max(0,j-1)] = 1
                            ret_canvas[min(ret_canvas.shape[0]-1,i+1),min(ret_canvas.shape[1]-1,j+1)] = 1
        return True
    
    def _canvas_oob(self, canvas, i, j):
        if (i < canvas.shape[0] and i >=0) and \
            (j < canvas.shape[1] and j >=0):
            return False
        return True
    
    def generate_objs_mask(self):
        """
        Simply generate object mask considering shapes and positions.
        """
        objs_mask_map = OrderedDict({ })
        for oid in self.oid_map.keys():
            ret_canvas = self.init_canvas.clone()
            (r, c) = self.opos_map[oid]
            obj = self.oid_map[oid]
            image_t = obj.image_t
            for i in range(r, r+image_t.shape[0]):
                for j in range(c, c+image_t.shape[1]):
                    if image_t[i-r,j-c] != self.background_color:
                        ret_canvas[i,j] = 1
            objs_mask_map[oid] = ret_canvas
        return objs_mask_map
    
    def generate_mask_inplace(self, obj_ctx_lists, connect_allow=False):
        ret_canvas = self.init_canvas.clone()
        for obj_ctx in obj_ctx_lists:
            (r, c) = obj_ctx[0]
            obj = obj_ctx[1]
            image_t = obj.image_t
            # print((r, c), (image_t.shape[0], image_t.shape[1]), (self.init_canvas.shape[0], self.init_canvas.shape[1]))
            for i in range(r, r+image_t.shape[0]):
                for j in range(c, c+image_t.shape[1]):
                    if image_t[i-r,j-c] != self.background_color:
                        ret_canvas[i,j] = 1
        
        if not connect_allow:
            copy_canvas = ret_canvas.clone()
            # we need to expand
            for i in range(ret_canvas.shape[0]):
                for j in range(ret_canvas.shape[1]):
                    if ret_canvas[i,j] == 1:
                        if not self._canvas_oob(copy_canvas,i-1,j):
                            copy_canvas[i-1,j] = 1
                        if not self._canvas_oob(copy_canvas,i-1,j+1):
                            copy_canvas[i-1,j+1] = 1
                        if not self._canvas_oob(copy_canvas,i-1,j-1):
                            copy_canvas[i-1,j-1] = 1
                        if not self._canvas_oob(copy_canvas,i+1,j):
                            copy_canvas[i+1,j] = 1
                        if not self._canvas_oob(copy_canvas,i,j-1):
                            copy_canvas[i,j-1] = 1
                        if not self._canvas_oob(copy_canvas,i,j+1):
                            copy_canvas[i,j+1] = 1
                        if not self._canvas_oob(copy_canvas,i+1,j+1):
                            copy_canvas[i+1,j+1] = 1
                        if not self._canvas_oob(copy_canvas,i+1,j-1):
                            copy_canvas[i+1,j-1] = 1

            return copy_canvas
        else:
            return ret_canvas
        
        # print("*****")
        # print(copy_canvas)
        # print(ret_canvas)
        # print("*****")
        
    def repr_as_dict(self, nodes=None, edges=None, minimum_cover=False):
        """
        We will serialize this canvas as a dict.
        Nodes contains outside namings of those labeled objects.
        Edges contains relations between objects recorded.
        """
        repre = OrderedDict({ })
        
        if nodes == None and edges == None:
            # we will use current info
            oid_image_map = OrderedDict({ })
            for k, v in self.oid_map.items():
                oid_image_map[k] = v.image_t
            repre["id_object_map"] = oid_image_map

            updated_canvas, r_diff, c_diff = self.render(is_plot=False, minimum_cover=minimum_cover)
            oid_position_map = OrderedDict({ })
            for k, v in self.opos_map.items():
                oid_position_map[k] = torch.tensor([v[0], v[1]])
            repre["id_position_map"] = oid_position_map

            repre["background_color"] = self.background_color
            repre["node_id_map"] = copy.deepcopy(self.node_id_map)

            # we note this as partial as we don't
            # provide a full parse of relation
            repre["partial_relation_edges"] = copy.deepcopy(self.partial_relation_edges)
            repre["image_t"] = updated_canvas
            repre["id_object_mask"] = self.generate_objs_mask()
            return repre
        
        if minimum_cover:
            """
            the canvas may contain area larger than needed.
            we thus will return the minimum cover required,
            we will update the obj position in the return.
            """
            oid_image_map = OrderedDict({ })
            for k, v in self.oid_map.items():
                oid_image_map[k] = v.image_t
            repre["id_object_map"] = oid_image_map

            updated_canvas, r_diff, c_diff = self.render(is_plot=False, minimum_cover=True)
            oid_position_map = OrderedDict({ })
            for k, v in self.opos_map.items():
                oid_position_map[k] = torch.tensor([v[0]-r_diff, v[1]-c_diff])
            repre["id_position_map"] = oid_position_map

            repre["background_color"] = self.background_color
            repre["node_id_map"] = nodes

            # we note this as partial as we don't
            # provide a full parse of relation
            repre["partial_relation_edges"] = edges
            repre["image_t"] = updated_canvas
            return repre
        else:
            oid_image_map = OrderedDict({ })
            for k, v in self.oid_map.items():
                oid_image_map[k] = v.image_t
            repre["id_object_map"] = oid_image_map

            updated_canvas, r_diff, c_diff = self.render(is_plot=False, minimum_cover=minimum_cover)
            oid_position_map = OrderedDict({ })
            for k, v in self.opos_map.items():
                oid_position_map[k] = torch.tensor([v[0], v[1]])
            repre["id_position_map"] = oid_position_map

            repre["background_color"] = self.background_color
            repre["node_id_map"] = nodes
            node_id_reverse = {}
            for k, v in repre["node_id_map"].items():
                node_id_reverse[v] = k
            # we note this as partial as we don't
            # provide a full parse of relation
            repre["partial_relation_edges"] = edges
            # Let us also add in more relations.
            complete_edges = self.parse_relations()

            updated_relation_edges = OrderedDict({ })
            for k, v in complete_edges.items():
                if v == "Attr" or (isinstance(v, list) and v[0] == "Attr"):
                    pass
                else:
                    new_k = (node_id_reverse[k[0]], node_id_reverse[k[1]])
                    updated_relation_edges[new_k] = v
            for k, v in edges.items():
                if v == "Attr":
                    updated_relation_edges[k] = [v]
            repre["partial_relation_edges"] = updated_relation_edges
            
            repre["image_t"] = updated_canvas
            repre["id_object_mask"] = self.generate_objs_mask()
            return repre
                    
    def render(self, obj_mask=None, is_plot=True, minimum_cover=False, save_file=None, axis_off=False):
        if minimum_cover:
            r_min, c_min = 99, 99
            r_max, c_max = -1, -1
            ret_canvas = self.init_canvas.clone()
            for oid in self.oid_map.keys():
                (r, c) = self.opos_map[oid]
                obj = self.oid_map[oid]
                image_t = obj.image_t
                if r < r_min:
                    r_min = r
                if c < c_min:
                    c_min =c
                if r+image_t.shape[0] > r_max:
                    r_max = r+image_t.shape[0]
                if c+image_t.shape[1] > c_max:
                    c_max =c+image_t.shape[1]
                for i in range(r, r+image_t.shape[0]):
                    for j in range(c, c+image_t.shape[1]):
                        if image_t[i-r,j-c] != self.background_color:
                            ret_canvas[i,j] = image_t[i-r,j-c]

            ret_canvas = ret_canvas[r_min:r_max, c_min:c_max]
            obj_t = ret_canvas
            image = np.zeros((*obj_t.shape, 3))
            for i in range(obj_t.shape[0]):
                for j in range(obj_t.shape[1]):
                    image[i,j] = np.array(self.color_dict[obj_t[i,j].tolist()])
            if is_plot:
                fig, ax = plt.subplots()
                plot_with_boundary(image, plt)
                if axis_off:
                    plt.axis('off')
                if save_file is not None:
                    fig.savefig(save_file)
                    
            return ret_canvas, r_min, c_min
        else:
            ret_canvas = self.init_canvas.clone()
            for oid in self.oid_map.keys():
                (r, c) = self.opos_map[oid]
                obj = self.oid_map[oid]
                image_t = obj.image_t
                for i in range(r, r+image_t.shape[0]):
                    for j in range(c, c+image_t.shape[1]):
                        if image_t[i-r,j-c] != self.background_color:
                            ret_canvas[i,j] = image_t[i-r,j-c]

            obj_t = ret_canvas
            image = np.zeros((*obj_t.shape, 3))
            for i in range(obj_t.shape[0]):
                for j in range(obj_t.shape[1]):
                    image[i,j] = np.array(self.color_dict[obj_t[i,j].tolist()])
            if is_plot:
                fig, ax = plt.subplots()
                plot_with_boundary(image, plt)
                if axis_off:
                    plt.axis('off')
                if save_file is not None:
                    fig.savefig(save_file)

            return ret_canvas, 0, 0
        
    def parse_relations(self):
        """
        parsing relations of existing objects on this canvas.
        """ 
        relation_edges = OrderedDict({})
        # add attributes
        for oid in self.oid_map.keys():
            (r, c) = self.opos_map[oid]
            obj = self.oid_map[oid]
            color_list = obj.image_t.unique().tolist()
            color_list = list(set(color_list) - {self.background_color}) # remove background color
            if len(color_list) == 1:
                color = int(color_list[0])
                # color attr
                relation_edges[(oid, f"color_[{color}]")] = ["Attr"]
            # pos attr
            else:
                pass # we will not record color = -1 case
            relation_edges[(oid, f"pos_[{r},{c}]")] = ["Attr"]
        # add relations
        for oid_left in self.oid_map.keys():
            for oid_right in self.oid_map.keys():
                if oid_left != oid_right:
                    # let us enumerate and see if there exists a relation between
                    (r_left, c_left) = self.opos_map[oid_left]
                    obj_left = self.oid_map[oid_left]
                    (r_right, c_right) = self.opos_map[oid_right]
                    obj_right = self.oid_map[oid_right]
                    
                    if SameShape(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("SameShape")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["SameShape"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("SameShape")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["SameShape"]
                    if SameColor(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("SameColor")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["SameColor"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("SameColor")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["SameColor"]
                    if SameAll(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("SameAll")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["SameAll"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("SameAll")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["SameAll"]
                    if SameRow(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("SameRow")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["SameRow"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("SameRow")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["SameRow"]
                    if SameCol(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("SameCol")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["SameCol"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("SameCol")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["SameCol"]
                    if IsInside(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("IsInside")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["IsInside"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("IsEnclosed")
                        else:
                            relation_edges[(oid_right, oid_left)] = ["IsEnclosed"]
                            
                    if IsTouch(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("IsTouch")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["IsTouch"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("IsTouch")
                        else:             
                            relation_edges[(oid_right, oid_left)] = ["IsTouch"]
                    if IsNonOverlapXY(obj_left.image_t, (r_left, c_left), 
                                 obj_right.image_t, (r_right, c_right)):
                        if (oid_left, oid_right) in relation_edges:
                            relation_edges[(oid_left, oid_right)].append("IsNonOverlapXY")
                        else:
                            relation_edges[(oid_left, oid_right)] = ["IsNonOverlapXY"]
                        if (oid_right, oid_left) in relation_edges:
                            relation_edges[(oid_right, oid_left)].append("IsNonOverlapXY")
                        else:
                            relation_edges[(oid_right, oid_left)] = ["IsNonOverlapXY"]
                            
        filtered_relation_edges = OrderedDict({})
        for k, vs in relation_edges.items():
            filtered_relation_edges[k] = list(set(vs))
        return filtered_relation_edges
    
    def change_obj_color(self, oid, new_color):
        curr_obj = self.oid_map[oid]
        for i in range(curr_obj.image_t.shape[0]):
            for j in range(curr_obj.image_t.shape[1]):
                if curr_obj.image_t[i,j] != self.background_color:
                    self.oid_map[oid].image_t[i,j] = new_color
        
        (placement_r, placement_c) = self.opos_map[oid]
        return (placement_r, placement_c)
    
    def change_obj_pos(self, oid, placement_r, placement_c, connect_allow=False):
        (old_placement_r, old_placement_c) = self.opos_map[oid]
        self.opos_map[oid] = (placement_r, placement_c)
        if self.check_conflict(connect_allow=connect_allow):
            return (placement_r, placement_c)
        else:
            self.opos_map[oid] = (old_placement_r, old_placement_c)
        return -1
    
    def placement_position_fixed(self, to_placement_obj, placement_r, placement_c, connect_allow=False):
        curr_obj_idx = len(self.oid_map.keys())
        curr_obj = copy.deepcopy(to_placement_obj)
        if self._check_exclusive(placement_r, placement_c, 
                                 to_placement_obj=curr_obj, 
                                 connect_allow=connect_allow):
            self.oid_map[curr_obj_idx] = curr_obj
            self.opos_map[curr_obj_idx] = (placement_r, placement_c)
            return (placement_r, placement_c)
        return -1
    
    def placement_inplace(self, to_placement_obj, to_relate_objs=[], placement_rule=None, 
                          connect_allow=False, # deprecated
                          allow_overlap=False, 
                          allow_connect=False):
        """
        This function just change current objects without adding new objects on the canvas.
        If this placement rule is not possible, we will return -1 as before as well.
        """
        assert placement_rule in {"SameAll", "SameShape", "SameColor", 
                                     "SameRow", "SameCol",
                                     "IsInside", "IsTouch"}
        pass
                    
class CanvasEngine:
    def __init__(self, background_color=0, min_length=10, max_length=30):
        self.background_color = background_color
        self.min_length = min_length
        self.small_length = 20
        self.max_length = max_length
        self.small_prob = 0.5
        self.same_prob = 0.5
        
        self.relation_pool = ["SameAll", "SameShape", "SameColor", 
                              "SameRow", "SameCol", "SubsetOf", 
                              "IsInside", "IsTouch"]

    def sameple_canvas(self, n=4):
        if random.uniform(0, 1) <= self.small_prob:
            # small
            unify_canvas_l = random.randint(self.min_length, self.small_length)
            unify_canvas_w = random.randint(self.min_length, self.small_length)
        else:
            # large
            unify_canvas_l = random.randint(self.min_length, self.max_length)
            unify_canvas_w = random.randint(self.min_length, self.max_length)

        if random.uniform(0, 1) <= self.same_prob:
            unify_canvas_max = max(unify_canvas_l, unify_canvas_w)
            canvas = torch.zeros((unify_canvas_max, unify_canvas_max))
        else:
            canvas = torch.zeros((unify_canvas_l, unify_canvas_w))
        
        # repeat canvas for all the examples
        canvas_list = []
        for i in range(n):
            canvas_list.append(Canvas(canvas.clone()))
            
        return canvas_list
    
    def sameple_canvas_by_size(self, n=4, min_length=20, max_length=30):        
        # repeat canvas for all the examples
        canvas_list = []
        for i in range(n):
            unify_canvas_l = random.randint(min_length, max_length)
            unify_canvas_w = random.randint(min_length, max_length)
            canvas = torch.zeros((unify_canvas_l, unify_canvas_w))
            canvas_list.append(Canvas(canvas.clone()))
            
        return canvas_list
    
    def sample_task_canvas_from_arc(self, image_t, parsed_objs):
        return Canvas(arc_image_t=image_t, arc_parsed_objs=parsed_objs)