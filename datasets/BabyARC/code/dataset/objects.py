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
from .utils import find_connected_components

import logging
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)

def randint_exclude(l, u, e):
    r = e[0]
    while r in e:
        r = random.randint(l, u)
    return r

# Object Generator Related Classes and Functions
class Object:
    """
    This is an abstraction of objects in BabyARC
    """
    def __init__(self, image_t, position_tags =[]):
        self.image_t = image_t
        self.position_tags = position_tags
        self.background_c = 0
            
    def get_image_t(self):
        return self.image_t
    
    def get_position_tags(self):
        return self.position_tags

    def fix_color(self):
        pass
    
class ObjectEngine:
    """
    Object Engine is responsible for sampling objects
    for different needs. It can also be used to manipulate
    objects in different way such as changing colors,
    rotating objects, etc..
    """
    def __init__(self, obj_pool=[], background_c=0.0, debug=False):
        self.md5_obj = {}
        self.md5_freq = {}
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
        if len(obj_pool) != 0:
            self.obj_pool = self._iso_obj_pool(obj_pool, debug=debug)
        else:
            self.obj_pool = []
        self.background_c = background_c
    
    def plot_objs(self, img_objs):
        for obj in img_objs:
            obj_t = obj.image_t
            image = np.zeros((*obj_t.shape, 3))
            for i in range(obj_t.shape[0]):
                for j in range(obj_t.shape[1]):
                    image[i,j] = np.array(self.color_dict[obj_t[i,j].tolist()])
            plot_with_boundary(image, plt)
            plt.axis('off')
    
    def sample_objs(self, n=1, is_plot=False, min_cover_exclude=1, is_colordiff=False, 
                     rotation="random", color="random"):
        objs_sampled = []
        while len(objs_sampled) < n:
            # continue to propose obj
            obj = sample(self.obj_pool, 1)
            obj_t = obj[0].image_t
            if is_colordiff:
                obj_p = obj[0].position_tags
                h = obj_t.shape[0]
                w = obj_t.shape[1]
                if h*w > min_cover_exclude:
                    objs_sampled.append(copy.deepcopy(obj[0]))
            else:
                # detect how many color is here
                color_list = obj_t.unique().tolist()
                if (0 in color_list and len(color_list) <= 2) or \
                    (0 not in color_list and len(color_list) == 2):
                    obj_p = obj[0].position_tags
                    h = obj_t.shape[0]
                    w = obj_t.shape[1]
                    if h*w > min_cover_exclude:
                        objs_sampled.append(copy.deepcopy(obj[0]))
        if is_plot:
            for obj in objs_sampled:
                obj_t = obj.image_t
                image = np.zeros((*obj_t.shape, 3))
                for i in range(obj_t.shape[0]):
                    for j in range(obj_t.shape[1]):
                        image[i,j] = np.array(self.color_dict[obj_t[i,j].tolist()])
                plot_with_boundary(image, plt)
                plt.axis('off')
                print("*** obj tags ***")
                print(obj.position_tags)
        return objs_sampled
    
    def _mask_encoding_img_t(self, img_t):
        # we simply get each mask and 
        # serialize, to detect multi-color
        # iso pictures
        color_bit_masks = []
        for i in range(0, 10):
            color_bit_mask = ''.join(str(e) for e in (img_t==i).float().tolist())
            color_bit_masks.append(color_bit_mask)
        color_bit_masks.sort() # sort so that we can consistant hash
        color_bit_masks_str = ''.join(color_bit_masks)
        return hashlib.md5(color_bit_masks_str.encode()).hexdigest()
    
    def _img_variant(self, img_t):
        # rotateA,B,C,D, vflip, hflip, diagflipA,B
        return [
            img_t.clone(),
            img_t.flip(-1),
            img_t.flip(-2),
            torch.rot90(img_t, k=1, dims=(-2, -1)),
            torch.rot90(img_t, k=2, dims=(-2, -1)),
            torch.rot90(img_t, k=3, dims=(-2, -1)),
            torch.rot90(img_t, k=1, dims=(-2, -1)).flip(-1),
            torch.rot90(img_t, k=1, dims=(-2, -1)).flip(-2)
        ]
    
    def _iso_obj_pool(self, obj_pool, debug=False):
        """
        we will shrink the object pool for iso objects.
        """
        if debug:
            logger.info(f"Original obj count = {len(obj_pool)}")
        shrink_obj_pool = []
        for obj in obj_pool:
            img_variant = self._img_variant(obj.image_t)
            is_variant_in = False
            for v in img_variant:
                image_t_str = self._mask_encoding_img_t(v)
                if image_t_str in self.md5_obj.keys():
                    is_variant_in = True
                    self.md5_freq[image_t_str] += 1
                    break
            if not is_variant_in:
                key_md5 = self._mask_encoding_img_t(img_variant[0])
                self.md5_obj[key_md5] = obj
                shrink_obj_pool.append(obj)
                self.md5_freq[key_md5] = 1
        if debug:
            logger.info(f"Iso obj count = {len(shrink_obj_pool)}")
        return shrink_obj_pool
    
    def sample_objs_by_bound_area(
        self, n=1, w_lim=5, h_lim=5, random_generated=True, 
        rainbow_prob=0.2, 
        concept_collection=["line", "Lshape", "rectangle", 
                            "rectangleSolid", "randomShape", "arcShape", 
                            "Tshape", "Eshape", 
                            "Hshape", "Cshape", "Ashape", "Fshape"],
        concept_limits={}
    ):
        """
        sample object within the width and height limits.
        if there is no such object in the pool, the engine
        may randomly generate one based on user preferences.
        """
        objs_sampled = []
        for i in range(n):
            chosen_shape = np.random.choice(concept_collection)

            if chosen_shape in {"line", "Lshape", "rectangle", 
                                "rectangleSolid", "randomShape", "arcShape", 
                                "Tshape", "Eshape", 
                                "Hshape", "Cshape", "Ashape", "Fshape"}:
                if chosen_shape == "line":
                    direction = random.randint(0,1)
                    if chosen_shape in concept_limits:
                        len_lims=concept_limits[chosen_shape]
                    else:
                        if direction == 0:
                            len_lims=[2,h_lim]
                        else:
                            len_lims=[2,w_lim]
                    obj = self.sample_objs_with_line(
                        n=1, len_lims=len_lims, 
                        thickness=1, 
                        direction=["v", "h"][direction],
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "Lshape":
                    direction=random.randint(0,3)
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,h_lim]
                        h_lims=[2,w_lim]
                    obj = self.sample_objs_with_l_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, thickness=1, 
                        rainbow_prob=rainbow_prob, direction=direction
                    )[0]
                elif chosen_shape == "Tshape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[3,5]
                        h_lims=[3,5]
                    obj = self.sample_objs_with_t_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0]
                elif chosen_shape == "Eshape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,4]
                        h_lims=[5,6]
                    obj = self.sample_objs_with_e_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0]
                elif chosen_shape == "Hshape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[3,5]
                        h_lims=[3,5]
                    obj = self.sample_objs_with_h_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0] 
                elif chosen_shape == "Cshape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,4]
                        h_lims=[3,5]
                    obj = self.sample_objs_with_c_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                elif chosen_shape == "Ashape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[3,5]
                        h_lims=[4,6]
                    obj = self.sample_objs_with_a_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                elif chosen_shape == "Fshape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,4]
                        h_lims=[4,6]
                    obj = self.sample_objs_with_f_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                elif chosen_shape == "rectangle":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,w_lim]
                        h_lims=[2,h_lim]
                    obj = self.sample_objs_with_rectangle(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        thickness=1, rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "rectangleSolid":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,w_lim]
                        h_lims=[2,h_lim]
                    obj = self.sample_objs_with_rectangle_solid(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "randomShape":
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        # Special bounds.
                        w = random.randint(2,4)
                        h = random.randint(2,4)
                        w_lims=[w,w]
                        h_lims=[h,h]
                    obj = self.sample_objs_with_random_shape(
                        n=1, w_lims=w_lims, h_lims=h_lims, 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "arcShape":
                    # Minimum we sample a random shape.
                    obj_sub_pool = []
                    shuffle_pool = copy.deepcopy(self.obj_pool)
                    random.shuffle(shuffle_pool)
                    obj = None
                    if chosen_shape in concept_limits:
                        w_lims=concept_limits[chosen_shape]
                        h_lims=concept_limits[chosen_shape]
                    else:
                        w_lims=[2,w_lim]
                        h_lims=[2,h_lim]
                    # Here we simply go through the object pool and sample arc object!
                    for arc_obj in shuffle_pool:

                        obj_img_t = arc_obj.image_t
                        if obj_img_t.shape[0] <= h_lims[1] and \
                            obj_img_t.shape[1] <= w_lims[1] and \
                            obj_img_t.shape[0] >= h_lims[0] and \
                            obj_img_t.shape[1] >= w_lims[0]:
                            # we add in a random rotated version of objs.
                            obj = self.random_rotation(
                                self.fix_color(
                                    arc_obj, random.randint(1,9)
                                )
                            )
                            obj.position_tags = []
                            break
                objs_sampled.append(obj)
            else:
                for i in range(n):
                    for obj in self.obj_pool:
                        obj_img_t = obj.image_t
                        if obj_img_t.shape[0] <= h_lim and \
                            obj_img_t.shape[1] <= w_lim and \
                            obj_img_t.shape[0] >= 2 and \
                            obj_img_t.shape[1] >= 2:
                            objs_sampled.append(self.random_color(obj))
                    if len(objs_sampled) == 0:
                        break
                if len(objs_sampled) >= 1 and random.random() >= 0.5:
                    return [random.choice(objs_sampled)]

                objs_sampled = []
                random_delete = True if random.random() >= 0.5 else False
                if len(objs_sampled) == 0 and random_generated == True:
                    for i in range(n):
                        w = random.randint(2, w_lim)
                        h = random.randint(2, h_lim)

                        img_t = torch.ones(h, w)

                        if random_delete:
                            img_t[random.randint(0, h-1), random.randint(0, w-1)] = self.background_c

                        new_obj = Object(img_t, position_tags=[])
                        if random.random() <= 1-rainbow_prob:
                            objs_sampled.append(self.random_color(new_obj))
                        else:
                            objs_sampled.append(self.random_color_rainbow(new_obj))

        return objs_sampled
    
    def sample_objs_by_fixed_width(
        self, n=1, width=5, h_lim=5, 
        random_generated=True, rainbow_prob=0.2, 
        concept_collection=["line", "Lshape", "rectangle", 
                            "rectangleSolid", "randomShape", "arcShape", 
                            "Tshape", "Eshape", 
                            "Hshape", "Cshape", "Ashape", "Fshape"]
    ):
        """
        sample object within the width and height limits.
        if there is no such object in the pool, the engine
        may randomly generate one based on user preferences.
        """
        objs_sampled = []
        for i in range(n):
            chosen_shape = np.random.choice(concept_collection)
            if chosen_shape in {"line", "Lshape", "rectangle", 
                                "rectangleSolid", "randomShape", "arcShape", 
                                "Tshape", "Eshape", 
                                "Hshape", "Cshape", "Ashape", "Fshape"}:
                if chosen_shape == "line":
                    direction = 1
                    len_lims=[width,width]
                    obj = self.sample_objs_with_line(
                        n=1, len_lims=len_lims, 
                        thickness=1, 
                        direction=["v", "h"][direction],
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "Lshape":
                    direction=random.randint(0,3)
                    obj = self.sample_objs_with_l_shape(
                        n=1, w_lims=[width,width], h_lims=[2,h_lim], thickness=1, 
                        rainbow_prob=rainbow_prob, direction=direction
                    )[0]
                    
                elif chosen_shape == "Tshape":
                    obj = self.sample_objs_with_t_shape(
                        n=1, w_lims=[width,width], h_lims=[3,5], 
                        rainbow_prob=rainbow_prob,
                    )[0]
                    if width < 3:
                        return []
                elif chosen_shape == "Eshape":
                    obj = self.sample_objs_with_e_shape(
                        n=1, w_lims=[width,width], h_lims=[5,6], 
                        rainbow_prob=rainbow_prob,
                    )[0]
                elif chosen_shape == "Hshape":
                    obj = self.sample_objs_with_h_shape(
                        n=1, w_lims=[width,width], h_lims=[3,5], 
                        rainbow_prob=rainbow_prob,
                    )[0] 
                    if width < 3:
                        return []
                elif chosen_shape == "Cshape":
                    obj = self.sample_objs_with_c_shape(
                        n=1, w_lims=[width,width], h_lims=[3,5], 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                elif chosen_shape == "Ashape":
                    obj = self.sample_objs_with_a_shape(
                        n=1, w_lims=[width,width], h_lims=[4,6], 
                        rainbow_prob=rainbow_prob,
                    )[0] 
                    if width < 3:
                        return []
                elif chosen_shape == "Fshape":
                    obj = self.sample_objs_with_f_shape(
                        n=1, w_lims=[width,width], h_lims=[4,6], 
                        rainbow_prob=rainbow_prob,
                    )[0]   

                elif chosen_shape == "rectangle":
                    obj = self.sample_objs_with_rectangle(
                        n=1, w_lims=[width,width], h_lims=[2,h_lim], 
                        thickness=1, rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "rectangleSolid":
                    obj = self.sample_objs_with_rectangle_solid(
                        n=1, w_lims=[width,width], h_lims=[2,h_lim], 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "randomShape":
                    obj = self.sample_objs_with_random_shape(
                        n=1, w_lims=[width,width], h_lims=[2,4], 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "arcShape":
                    # Minimum we sample a random shape.
                    obj_sub_pool = []
                    shuffle_pool = copy.deepcopy(self.obj_pool)
                    random.shuffle(shuffle_pool)
                    obj = None
                    # Here we simply go through the object pool and sample arc object!
                    for arc_obj in shuffle_pool:
                        obj_img_t = arc_obj.image_t
                        if obj_img_t.shape[0] <= h_lim and \
                            obj_img_t.shape[1] == width and \
                            obj_img_t.shape[0] >= 2 and \
                            obj_img_t.shape[1] >= 2:
                            # we add in a random rotated version of objs.
                            obj = self.fix_color(
                                    arc_obj, random.randint(1,9)
                                )
                            obj.position_tags = []
                            break
                objs_sampled.append(obj)
            else:
                for obj in self.obj_pool:
                    obj_img_t = obj.image_t
                    if obj_img_t.shape[1] == width and obj_img_t.shape[0] <= h_lim:
                        objs_sampled.append(self.random_color(obj))
                if len(objs_sampled) == 0:
                    break
                
                if len(objs_sampled) >= 1 and random.random() >= 0.5:
                    return [random.choice(objs_sampled)]

                objs_sampled = []
                random_delete = True if random.random() >= 0.5 else False
                random_delete = False
                if len(objs_sampled) == 0 and random_generated == True:
                    for i in range(n):
                        w = width
                        h = random.randint(2, h_lim)

                        img_t = torch.ones(h, w)

                        if random_delete:
                            img_t[random.randint(0, h-1), random.randint(0, w-1)] = self.background_c

                        new_obj = Object(img_t, position_tags=[])
                        if random.random() <= 1-rainbow_prob:
                            objs_sampled.append(self.random_color(new_obj))
                        else:
                            objs_sampled.append(self.random_color_rainbow(new_obj))

        return objs_sampled
    
    def sample_objs_by_fixed_height(
        self, n=1, height=5, w_lim=5, 
        random_generated=True, rainbow_prob=0.2, 
        concept_collection=["line", "Lshape", "rectangle", 
                            "rectangleSolid", "randomShape", "arcShape", 
                            "Tshape", "Eshape", 
                            "Hshape", "Cshape", "Ashape", "Fshape"]
    ):
        """
        sample object within the width and height limits.
        if there is no such object in the pool, the engine
        may randomly generate one based on user preferences.
        """
        objs_sampled = []
        for i in range(n):
            chosen_shape = np.random.choice(concept_collection)
            if chosen_shape in {"line", "Lshape", "rectangle", 
                                "rectangleSolid", "randomShape", "arcShape", 
                                "Tshape", "Eshape", 
                                "Hshape", "Cshape", "Ashape", "Fshape"}:
                if chosen_shape == "line":
                    direction = 0
                    len_lims=[height,height]
                    obj = self.sample_objs_with_line(
                        n=1, len_lims=len_lims, 
                        thickness=1, 
                        direction=["v", "h"][direction],
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "Lshape":
                    direction=random.randint(0,3)
                    obj = self.sample_objs_with_l_shape(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], thickness=1, 
                        rainbow_prob=rainbow_prob, direction=direction
                    )[0]
                elif chosen_shape == "Tshape":
                    obj = self.sample_objs_with_t_shape(
                        n=1, w_lims=[3,5], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0]
                    if height < 3:
                        return []
                elif chosen_shape == "Eshape":
                    obj = self.sample_objs_with_e_shape(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0]
                    if height < 5:
                        return []
                elif chosen_shape == "Hshape":
                    obj = self.sample_objs_with_h_shape(
                        n=1, w_lims=[3,5], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0] 
                    if height < 3:
                        return []
                elif chosen_shape == "Cshape":
                    obj = self.sample_objs_with_c_shape(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                    if height < 3:
                        return []
                elif chosen_shape == "Ashape":
                    obj = self.sample_objs_with_a_shape(
                        n=1, w_lims=[3,5], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0]   
                    if height < 4:
                        return []
                elif chosen_shape == "Fshape":
                    obj = self.sample_objs_with_f_shape(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob,
                    )[0]  
                    if height < 4:
                        return []
                elif chosen_shape == "rectangle":
                    obj = self.sample_objs_with_rectangle(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], 
                        thickness=1, rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "rectangleSolid":
                    obj = self.sample_objs_with_rectangle_solid(
                        n=1, w_lims=[2,w_lim], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "randomShape":
                    obj = self.sample_objs_with_random_shape(
                        n=1, w_lims=[2,4], h_lims=[height,height], 
                        rainbow_prob=rainbow_prob
                    )[0]
                elif chosen_shape == "arcShape":
                    # Minimum we sample a random shape.
                    obj_sub_pool = []
                    shuffle_pool = copy.deepcopy(self.obj_pool)
                    random.shuffle(shuffle_pool)
                    obj = None
                    # Here we simply go through the object pool and sample arc object!
                    for arc_obj in shuffle_pool:
                        obj_img_t = arc_obj.image_t
                        if obj_img_t.shape[0] == height and \
                            obj_img_t.shape[1] <= w_lim and \
                            obj_img_t.shape[0] >= 2 and \
                            obj_img_t.shape[1] >= 2:
                            # we add in a random rotated version of objs.
                            obj = self.fix_color(
                                    arc_obj, random.randint(1,9)
                                )
                            obj.position_tags = []
                            break
                objs_sampled.append(obj)
            else:
                for obj in self.obj_pool:
                    obj_img_t = obj.image_t
                    if obj_img_t.shape[0] == height and obj_img_t.shape[1] <= w_lim:
                        objs_sampled.append(self.random_color(obj))
                if len(objs_sampled) == 0:
                    break
                
                if len(objs_sampled) >= 1 and random.random() >= 0.5:
                    return [random.choice(objs_sampled)]

                objs_sampled = []
                random_delete = True if random.random() >= 0.5 else False
                random_delete = False
                if len(objs_sampled) == 0 and random_generated == True:
                    for i in range(n):
                        w = random.randint(2, w_lim)
                        h = height

                        img_t = torch.ones(h, w)

                        if random_delete:
                            img_t[random.randint(0, h-1), random.randint(0, w-1)] = self.background_c

                        new_obj = Object(img_t, position_tags=[])
                        if random.random() <= 1-rainbow_prob:
                            objs_sampled.append(self.random_color(new_obj))
                        else:
                            objs_sampled.append(self.random_color_rainbow(new_obj))

        return objs_sampled
    
    def sample_objs_with_composite_shape(
        self, n=1, w_lims=[16,16], h_lims=[16,16], 
        rainbow_prob=0.0, 
        chosen_concept="RectE1a",
        n_retry=30,
        allow_connect=True,
        is_plot=False,
        parsing_check=True,
        color_avail=[1,2,3,4,5,6,7,8,9],
    ):
        from .dataset import BabyARCDataset
        from .canvas import Canvas
        if color_avail == None:
            color_avail=[1,2,3,4,5,6,7,8,9]
        assert chosen_concept in {"RectE1a", "RectE1b", "RectE1c", 
                                   "RectE2a", "RectE2b", "RectE2c",
                                   "RectE3a", "RectE3b", "RectE3c", 
                                   "RectF1a", "RectF1b", "RectF1c", 
                                   "RectF2a", "RectF2b", "RectF2c",
                                   "RectF3a", "RectF3b", "RectF3c",}
        """
        Different from other concepts, here the w and h limit is the
        upper limit only. It is not used to sample anything.
        """
        # we are doing recursive babyARC canvas sampling here.
        canvas_size=min([w_lims[0], w_lims[1], h_lims[0], h_lims[1]])
        _dataset_engine = BabyARCDataset(
            None,
            save_directory="./BabyARCDataset/", 
            object_limit=1, noise_level=0, 
            canvas_size=canvas_size,
            skip_load_pretrain_obj=True,
        ) # canvas makes w=h canvas
        
        # now, depends on the input composite type, we make the corresponding canvas
        obj_spec = None
        upper_bound_size=int(canvas_size*0.8)
        if chosen_concept == "RectE1a" or chosen_concept == "RectF1a" or chosen_concept == "RectE1b" or chosen_concept == "RectF1b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, upper_bound_size)
            out_h = np.random.randint(5, upper_bound_size)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, upper_bound_size//2+1)
            char_h = np.random.randint(5, upper_bound_size//2+1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr')]
        elif chosen_concept == "RectE1c" or chosen_concept == "RectF1c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, upper_bound_size)
            out_h = np.random.randint(5, upper_bound_size)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, upper_bound_size//2+1)
            char_h = np.random.randint(5, upper_bound_size//2+1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE2a" or chosen_concept == "RectF2a" or chosen_concept == "RectE2b" or chosen_concept == "RectF2b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, upper_bound_size)
            out_h = np.random.randint(5, upper_bound_size)
            in_w = np.random.randint(4, upper_bound_size//2)
            in_h = np.random.randint(4, upper_bound_size//2)
            char_w = np.random.randint(3, upper_bound_size//2)
            char_h = np.random.randint(5, upper_bound_size//2)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside')]
        elif chosen_concept == "RectE2c" or chosen_concept == "RectF2c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, upper_bound_size)
            out_h = np.random.randint(5, upper_bound_size)
            in_w = np.random.randint(4, upper_bound_size//2)
            in_h = np.random.randint(4, upper_bound_size//2)
            char_w = np.random.randint(3, upper_bound_size//2)
            char_h = np.random.randint(5, upper_bound_size//2)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE3a" or chosen_concept == "RectF3a" or chosen_concept == "RectE3b" or chosen_concept == "RectF3b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(9, upper_bound_size)
            out_h = np.random.randint(9, upper_bound_size)
            in_w = np.random.randint(7, out_w-1)
            in_h = np.random.randint(7, out_h-1)
            char_w = np.random.randint(3, in_w-1)
            char_h = np.random.randint(5, in_h-1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'Eshape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'IsOutside')]
        elif chosen_concept == "RectE3c" or chosen_concept == "RectF3c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(9, upper_bound_size)
            out_h = np.random.randint(9, upper_bound_size)
            in_w = np.random.randint(7, out_w-1)
            in_h = np.random.randint(7, out_h-1)
            char_w = np.random.randint(3, in_w-1)
            char_h = np.random.randint(5, in_h-1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'Eshape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'IsOutside'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        
        for i in range(0, n_retry):
            canvas_dict = _dataset_engine.sample_single_canvas_by_core_edges(
                OrderedDict(obj_spec),
                allow_connect=allow_connect,
                rainbow_prob=rainbow_prob,
                is_plot=False, # enforced!
                parsing_check=parsing_check,
                color_avail=color_avail,
            )
            if canvas_dict != -1:
                # need extra checks here as well!
                failed = False
                if "1" in chosen_concept:
                    mid_ax_y_l = canvas_dict["id_position_map"][1][0].tolist() + (canvas_dict["id_object_map"][1].shape[0])//2
                    mid_ax_x_l = canvas_dict["id_position_map"][1][1].tolist() + (canvas_dict["id_object_map"][1].shape[1])//2

                    mid_ax_y_r = canvas_dict["id_position_map"][2][0].tolist() + (canvas_dict["id_object_map"][2].shape[0])//2
                    mid_ax_x_r = canvas_dict["id_position_map"][2][1].tolist() + (canvas_dict["id_object_map"][2].shape[1])//2

                    if abs(mid_ax_y_l-mid_ax_y_r) <= 2 or abs(mid_ax_x_l-mid_ax_x_r) <= 2:
                        if ('obj_2', 'obj_0') in canvas_dict["partial_relation_edges"]:
                            if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_0')]:
                                failed = True
                        if ('obj_2', 'obj_1') in canvas_dict["partial_relation_edges"]:
                            if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_1')]:
                                failed = True
                    else:
                        failed = True
                if "2" in chosen_concept:
                    if ('obj_2', 'obj_1') in canvas_dict["partial_relation_edges"]:
                        if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_1')]:
                            failed = True
                if "3" in chosen_concept:
                    pass
                if "b" in chosen_concept:
                    if ('obj_1', 'obj_2') in canvas_dict["partial_relation_edges"]:
                        if "SameColor" in canvas_dict["partial_relation_edges"][('obj_1', 'obj_2')]:
                            failed = True
                if not failed:
                    break
        if canvas_dict != -1 and not failed:
            _cavnas = Canvas(
                repre_dict=canvas_dict
            )
            img_t, _, _ = _cavnas.render(is_plot=False, minimum_cover=True)
            new_obj = Object(img_t, position_tags=[])
            return [new_obj]
        else:
            return [None]

    def random_color(self, img_obj, color="random", rainbow_prob=0.2):
        if random.random() <= 1-rainbow_prob:
            ret = copy.deepcopy(img_obj)
            color_list = ret.image_t.unique().tolist()
            for c in color_list:
                if c != self.background_c:
                    if color == "random":
                        new_c = randint_exclude(0,9,[c, self.background_c])
                        ret.image_t[img_obj.image_t==c] = new_c
                    else:
                        # fixed color
                        pass
        else:
            return self.random_color_rainbow(img_obj, color="random")
        return ret
    
    def _random_color(self, img_t, color="random"):
        ret = copy.deepcopy(img_t)
        color_list = ret.unique().tolist()
        for c in color_list:
            if c != self.background_c:
                if color == "random":
                    new_c = randint_exclude(0,9,[c, self.background_c])
                    ret[img_t==c] = new_c
                else:
                    # fixed color
                    pass
        return []
    
    def random_color_rainbow(self, img_obj, color="random"):
        ret = copy.deepcopy(img_obj)
        color_list = ret.image_t.unique().tolist()
        for i in range(ret.image_t.shape[0]):
            for j in range(ret.image_t.shape[1]):
                if ret.image_t[i,j] != self.background_c:
                    ret.image_t[i,j] = randint_exclude(0,9,[self.background_c])
        return ret
    
    def _random_color_rainbow(self, img_t):
        ret = copy.deepcopy(img_t)
        color_list = ret.unique().tolist()
        for i in range(ret.shape[0]):
            for j in range(ret.shape[1]):
                if ret[i,j] != self.background_c:
                    ret[i,j] = randint_exclude(0,9,[self.background_c])
        return ret
    
    def fix_color(self, img_obj, new_color):
        ret = copy.deepcopy(img_obj)
        for i in range(ret.image_t.shape[0]):
            for j in range(ret.image_t.shape[1]):
                if ret.image_t[i,j] != self.background_c:
                    ret.image_t[i,j] = new_color
        return ret
    
    def _rotate_tag(self, original_tags, n):
        """
        rotate the tag ccw
        """
        curr_position_tags = copy.deepcopy(original_tags)
        for i in range(n):
            new_position_tags = []
            for t in curr_position_tags:
                if t == "upper":
                    new_position_tags.append("left")
                elif t == "left":
                    new_position_tags.append("lower")
                elif t == "lower":
                    new_position_tags.append("right")
                elif t == "right":
                    new_position_tags.append("upper")
                else:
                    pass
            curr_position_tags = copy.deepcopy(new_position_tags) # recurrent
        return curr_position_tags
    
    def _flip_tag(self, original_tags, direction):
        curr_position_tags = copy.deepcopy(original_tags)
        if direction == -1:
            new_position_tags = []
            for t in curr_position_tags:
                if t == "left":
                    new_position_tags.append("right")
                elif t == "right":
                    new_position_tags.append("left")
                else:
                    new_position_tags.append(t)
        elif direction == -2:
            new_position_tags = []
            for t in curr_position_tags:
                if t == "upper":
                    new_position_tags.append("lower")
                elif t == "lower":
                    new_position_tags.append("upper")
                else:
                    new_position_tags.append(t)
        return new_position_tags

    def random_rotation(self, img_obj, rotation="random"):
        ret = copy.deepcopy(img_obj)
        # update image and tag
        if rotation == "random":
            rotation = random.randint(0, 6)
            if rotation == 0:
                ret.image_t = ret.image_t.flip(-1)
                ret.position_tags = self._flip_tag(ret.position_tags, -1)
            elif rotation == 1:
                ret.image_t = ret.image_t.flip(-2)
                ret.position_tags = self._flip_tag(ret.position_tags, -2)
            elif rotation == 2:
                ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1))
                ret.position_tags = self._rotate_tag(ret.position_tags, 1)
            elif rotation == 3:
                ret.image_t = torch.rot90(ret.image_t, k=2, dims=(-2, -1))
                ret.position_tags = self._rotate_tag(ret.position_tags, 2)
            elif rotation == 4:
                ret.image_t = torch.rot90(ret.image_t, k=3, dims=(-2, -1))
                ret.position_tags = self._rotate_tag(ret.position_tags, 3)
            elif rotation == 5:
                ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1)).flip(-1)
                ret.position_tags = self._flip_tag(self._rotate_tag(ret.position_tags, 1), -1)
            elif rotation == 6:
                ret.image_t = torch.rot90(ret.image_t, k=1, dims=(-2, -1)).flip(-2)
                ret.position_tags = self._flip_tag(self._rotate_tag(ret.position_tags, 1), -2)
            else:
                pass
        return ret

    def sample_objs_with_rectangle_solid(self, n=1, w_lims=[5,10], h_lims=[5,10],
                                         rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.ones(h, w)

            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled

    def sample_objs_with_arc(
        self, n=1, w_lims=[4,4], h_lims=[4,4], 
        rainbow_prob=0.2
    ):
        pass
    
    def sample_objs_with_random_rectangle(
        self, n=1, w_lims=[4,4], h_lims=[4,4], 
        rainbow_prob=0.2
    ):
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.ones(h, w)
            
            # Randomly remove pixels by overlapping a cover
            cover_w = random.randint(1, w-1)
            cover_h = random.randint(1, h-1)
            
            # attaching boundary
            select_boundary = random.randint(0,3)
            if select_boundary == 0:
                # |
                pos_r = random.randint(0, h-cover_h)
                img_t[pos_r:pos_r+cover_h, 0:cover_w] = 0
            elif select_boundary == 1:
                # -
                pos_c = random.randint(0, w-cover_w)
                img_t[0:cover_h, pos_c:pos_c+cover_w] = 0
            elif select_boundary == 2:
                #  |
                pos_r = random.randint(0, h-cover_h)
                img_t[pos_r:pos_r+cover_h, -cover_w:] = 0
            elif select_boundary == 3:
                # _
                pos_c = random.randint(0, w-cover_w)
                img_t[-cover_h:, pos_c:pos_c+cover_w] = 0

            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled
    
    def sample_objs_with_random_shape(
        self, n=1, w_lims=[4,4], h_lims=[4,4], 
        rainbow_prob=0.2
    ):
        objs_sampled = []
        for i in range(n):
            
            disconnected = True
            while disconnected:

                w = random.randint(w_lims[0], w_lims[1])
                h = random.randint(h_lims[0], h_lims[1])

                img_t = torch.randint(0, 2, (h, w)) # random bitmap 

                # check if this is valid, if yet, set the flag and add into the return list
                n_objs = find_connected_components(img_t, is_diag=True)
                n_objs = len(n_objs)
                if n_objs == 1:
                    # let us see, if we can shrink the size, if yes, not statisfied!
                    up_check = False
                    for i in range(w):
                        if img_t[0, i] != 0:
                            up_check = True
                            break
                    down_check = False
                    for i in range(w):
                        if img_t[-1, i] != 0:
                            down_check = True
                            break
                    
                    left_check = False
                    for i in range(h):
                        if img_t[i, 0] != 0:
                            left_check = True
                            break
                    right_check = False
                    for i in range(h):
                        if img_t[i, -1] != 0:
                            right_check = True
                            break
                    if up_check and down_check and left_check and right_check:
                        disconnected = False # find!

            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled
    
    def sample_objs_with_rectangle(
        self, n=1, w_lims=[5,10], h_lims=[5,10], thickness=1, 
        rainbow_prob=0.2, concept_limits={},
    ):
        
        objs_sampled = []
        for i in range(n):
            if "rectangle" in concept_limits.keys():
                w = random.randint(concept_limits["rectangle"][0], concept_limits["rectangle"][1])
                h = random.randint(concept_limits["rectangle"][0], concept_limits["rectangle"][1])
            else:
                w = random.randint(w_lims[0], w_lims[1])
                h = random.randint(h_lims[0], h_lims[1])

            thickness = thickness

            img_t = torch.zeros(h, w)

            for t in range(0, thickness):
                for i in range(t, w-t):
                    img_t[t, i] = 1
                    img_t[-1-t, i] = 1
                for i in range(t, h-t):
                    img_t[i, t] = 1
                    img_t[i, -1-t] = 1
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled
       
    def sample_objs_with_t_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, w):
                img_t[0, i] = 1 # -
            for i in range(0, h):
                img_t[i, w//2] = 1 # |
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
        
    def sample_objs_with_f_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, w):
                img_t[0, i] = 1 # -
            for i in range(0, h):
                img_t[i, 0] = 1 # |
            for i in range(0, w):
                img_t[h//2, i] = 1 # _
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
        
    def sample_objs_with_e_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, w):
                img_t[0, i] = 1 # -
            for i in range(0, h):
                img_t[i, 0] = 1 # |
            for i in range(0, w):
                img_t[h//2, i] = 1 # _
            for i in range(0, w):
                img_t[-1, i] = 1 # _
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
        
    def sample_objs_with_h_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, h):
                img_t[i, -1] = 1 # |
            for i in range(0, h):
                img_t[i, 0] = 1 # |
            for i in range(0, w):
                img_t[h//2, i] = 1 # _
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
        
    def sample_objs_with_c_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, h):
                img_t[i, 0] = 1 # |
            for i in range(0, w):
                img_t[0, i] = 1 # _
            for i in range(0, w):
                img_t[-1, i] = 1 # _
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
    
    def sample_objs_with_a_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], rainbow_prob=0.2):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            img_t = torch.zeros(h, w)
            
            for i in range(0, h):
                img_t[i, 0] = 1 # |
            for i in range(0, h):
                img_t[i, -1] = 1 # |
            for i in range(0, w):
                img_t[0, i] = 1 # _
            for i in range(0, w):
                img_t[h//2, i] = 1 # _
                
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_rotation(self.random_color(new_obj, rainbow_prob=rainbow_prob)))
            else:
                objs_sampled.append(self.random_rotation(self.random_color_rainbow(new_obj)))
        return objs_sampled
        
    def sample_objs_with_l_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], thickness=1, rainbow_prob=0.2, direction=0):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            thickness = thickness

            img_t = torch.zeros(h, w)

            if direction == 0:
                for t in range(0, thickness):
                    for i in range(t, w-t):
                        img_t[t, i] = 1
                    for i in range(t, h-t):
                        img_t[i, t] = 1
            elif direction == 1:
                for t in range(0, thickness):
                    for i in range(t, w-t):
                        img_t[t, i] = 1
                    for i in range(t, h-t):
                        img_t[i, -1-t] = 1
            elif direction == 2:
                for t in range(0, thickness):
                    for i in range(t, w-t):
                        img_t[-1-t, i] = 1
                    for i in range(t, h-t):
                        img_t[i, -1-t] = 1
            elif direction == 3:
                for t in range(0, thickness):
                    for i in range(t, w-t):
                        img_t[-1-t, i] = 1
                    for i in range(t, h-t):
                        img_t[i, t] = 1
            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled
        
    def sample_objs_with_enclosure(self, n=1, w_lims=[5,10], h_lims=[5,10], thickness=1, rainbow_prob=0.2, 
                                   gravity=False, irrregular=False):
        
        objs_sampled = []
        for i in range(n):
            w = random.randint(w_lims[0], w_lims[1])
            h = random.randint(h_lims[0], h_lims[1])

            thickness = thickness

            img_t = torch.zeros(h, w)
            
            # i don't think we are supporting thickness here yet
            for t in range(0, thickness):
                for i in range(t, w-t):
                    img_t[t, i] = 1
                    img_t[-1-t, i] = 1
                for i in range(t, h-t):
                    img_t[i, t] = 1
                    img_t[i, -1-t] = 1
                    
            # opening up for the closure
            openup_length = random.randint(0, 3)
            if openup_length > 0 and min(w,h)-openup_length > 1:
                start_point = random.randint(1, min(w,h)-openup_length-2)
                if gravity:
                    openup_dir = random.randint(1, 3)
                    if openup_dir == 1:
                        for i in range(start_point, start_point+openup_length):
                            img_t[0, i] = 0
                    elif openup_dir == 2:
                        for i in range(start_point, start_point+openup_length):
                            img_t[i, -1] = 0
                    elif openup_dir == 3:
                        for i in range(start_point, start_point+openup_length):
                            img_t[i, 0] = 0
                else:
                    openup_dir = random.randint(1, 4)
                    if openup_dir == 1:
                        for i in range(start_point, start_point+openup_length):
                            img_t[0, i] = 0
                    elif openup_dir == 2:
                        for i in range(start_point, start_point+openup_length):
                            img_t[i, -1] = 0
                    elif openup_dir == 3:
                        for i in range(start_point, start_point+openup_length):
                            img_t[-1, i] = 0
                    elif openup_dir == 4:
                        for i in range(start_point, start_point+openup_length):
                            img_t[i, 0] = 0
            
            if irrregular:
                irr_number = random.randint(0, w+h)
                # adding irregular pixels for the closure
                if irr_number > 0:
                    for i in range(irr_number):
                        rand_dir = random.randint(1, 4)

                        if rand_dir == 1:
                            rand_pos = random.randint(1, w-1)
                            img_t[1, rand_pos] = 1
                        elif rand_dir == 2:
                            rand_pos = random.randint(1, h-1)
                            img_t[rand_pos, -1] = 1
                        elif rand_dir == 3:
                            rand_pos = random.randint(1, w-1)
                            img_t[-1, rand_pos] = 1
                        elif rand_dir == 4:
                            rand_pos = random.randint(1, h-1)
                            img_t[rand_pos, 0] = 1

            # color
            new_obj = Object(img_t, position_tags=[])

            if random.random() <= 1-rainbow_prob:
                objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
            else:
                objs_sampled.append(self.random_color_rainbow(new_obj))
        return objs_sampled

    def sample_objs_with_pixel(self, n=1):
        objs_sampled = []
        for i in range(n):
            img_t = torch.ones(1,1)
            # color
            new_obj = Object(img_t, position_tags=[])
            objs_sampled.append(self.random_color(new_obj, rainbow_prob=0.0))
        return objs_sampled
    
    def sample_objs_with_line(self, n=1, len_lims=[5,10], thickness=1, rainbow_prob=0.1, direction="v"):
        objs_sampled = []
        for i in range(n):
            rand_len = random.randint(len_lims[0], len_lims[1])
            if direction == "v":
                img_t = torch.ones(rand_len,thickness)
            elif direction == "h":
                img_t = torch.ones(thickness,rand_len)
            # color
            new_obj = Object(img_t, position_tags=[])
            objs_sampled.append(self.random_color(new_obj, rainbow_prob=rainbow_prob))
        return objs_sampled
    
    def sample_objs_with_symmetry_shape(self, n=1, w_lims=[5,10], h_lims=[5,10], 
                                        rainbow_prob=0.2, axis_list=[0,1], axis_maintain=False, 
                                        solid_prob=0.5):
        # can have 4 symmetry axis [0 (-), 1(|), 2(\), 3(/)]
        objs_sampled = []
        for i in range(n):
            found_valid = False
            while not found_valid:
                # based on the axis, we calculate the lims
                if 2 in axis_list or 3 in axis_list:
                    _lim_1 = min(w_lims[0], h_lims[0])
                    _lim_2 = min(w_lims[1], h_lims[1])
                    s = random.randint(_lim_1, _lim_2)
                    w = s
                    h = s # make it a square
                else: 
                    w = random.randint(w_lims[0], w_lims[1])
                    h = random.randint(h_lims[0], h_lims[1])

                seed_img_t = (torch.rand(size=(h,w)) <= solid_prob).int()
                while len(seed_img_t.unique()) == 1 and seed_img_t[0] == 0:
                    seed_img_t = (torch.rand(size=(h,w)) <= solid_prob).int() # reject sampling

                if random.random() <= 1-rainbow_prob:
                    seed_img_t = self._random_color(seed_img_t)
                else:
                    seed_img_t = self._random_color_rainbow(seed_img_t)

                for axis in axis_list:
                    if axis == 0:
                        sym_img_t = seed_img_t.flip(-2)
                        start_r = int(math.ceil(h/2))
                        for i in range(start_r, h):
                            for j in range(0, w):
                                seed_img_t[i,j] = sym_img_t[i,j]
                    elif axis == 1:
                        sym_img_t = seed_img_t.flip(-1)
                        start_c = int(math.ceil(w/2))
                        for i in range(0, h):
                            for j in range(start_c, w):
                                seed_img_t[i,j] = sym_img_t[i,j]
                    elif axis == 2:
                        assert seed_img_t.shape[0] == seed_img_t.shape[1]
                        sym_img_t = torch.rot90(seed_img_t, k=1, dims=(-2, -1)).flip(-2)
                        for i in range(1, h):
                            for j in range(0, i):
                                seed_img_t[i,j] = sym_img_t[i,j]
                    elif axis == 3:
                        assert seed_img_t.shape[0] == seed_img_t.shape[1]
                        sym_img_t = torch.rot90(seed_img_t, k=1, dims=(-2, -1)).flip(-1)
                        for i in range(1, h):
                            for j in range(w-i, w):
                                seed_img_t[i,j] = sym_img_t[i,j]
                  
                # check if this is valid, if yet, set the flag and add into the return list
                n_objs = find_connected_components(seed_img_t, is_diag=True)
                n_objs = len(n_objs)
                if n_objs == 1:
                    new_obj = Object(seed_img_t, position_tags=[])
                    objs_sampled.append(new_obj)
                    break
        
        return objs_sampled