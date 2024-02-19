# adopted from Tailin Wu
import numpy as np
from copy import deepcopy
import os,random
from collections import OrderedDict
from collections.abc import Iterable
import pprint as pp

import torch
from torch.utils.data import Dataset, DataLoader

from .BabyARC.code.dataset.dataset import *
from .util import REA_PATH, EXP_PATH, get_root_dir, remove_elements,to_one_hot,Dictionary,Printer,visualize_matrices,plot_matrices

p = Printer()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_c_core(c):
    if isinstance(c, list):
        return [get_c_core(ele) for ele in c]
    else:
        assert isinstance(c, str)
        return c.split("[")[0]


def get_c_size(c):
    assert isinstance(c, str)
    if "[" in c:
        string = "[{}]".format(c.split("[")[1][:-1])
        min_size, max_size = eval(string)
    else:
        min_size, max_size = None, None
    return min_size, max_size


def obj_spec_fun(
    concept_collection, 
    min_n_objs, max_n_objs, 
    canvas_size, 
    allowed_shape_concept=None,
    is_conjuncture=True,
    color_avail=None,
    idx_start=0,
    focus_type=None,
):
    """Generate specs for several objects for BabyARC.

    Args:
        idx_start: obj id to start with.
    """
    n_objs = np.random.randint(min_n_objs, max_n_objs+1)
    obj_spec = []
    if focus_type is not None:
        assert focus_type in concept_collection
    if set(get_c_core(concept_collection)).issubset({
        "Line", "Rect", "RectSolid", 
        "Lshape", "Randshape", "ARCshape", 
        "Tshape", "Eshape", 
        "Hshape", "Cshape", "Ashape", "Fshape",
        "RectE1a", "RectE1b", "RectE1c", 
        "RectE2a", "RectE2b", "RectE2c",
        "RectE3a", "RectE3b", "RectE3c", 
        "RectF1a", "RectF1b", "RectF1c", 
        "RectF2a", "RectF2b", "RectF2c",
        "RectF3a", "RectF3b", "RectF3c",
    }):
        if focus_type is None:
            partition = np.sort(np.random.choice(n_objs+1, len(concept_collection)-1, replace=True))
            max_rect_size = canvas_size // 2
            for k in range(idx_start, n_objs + idx_start):
                if len(concept_collection) == 1:
                    chosen_concept = concept_collection[0]
                else:
                    gt = k-idx_start >= partition  # gt: greater_than_vector
                    if gt.any():
                        id = np.where(gt)[0][-1] + 1
                    else:
                        id = 0
                    chosen_concept = concept_collection[id]
                chosen_concept_core = get_c_core(chosen_concept)
                min_size, max_size = get_c_size(chosen_concept)
                if chosen_concept_core == "Line":
                    if min_size is None:
                        obj_spec.append((('obj_{}'.format(k), 'line_[-1,1,-1]'), 'Attr'))
                    else:
                        h = np.random.randint(min_size, max_size+1)
                        obj_spec.append((('obj_{}'.format(k), f'line_[{h},1,-1]'), 'Attr'))
                elif chosen_concept_core == "Rect":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangle_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "RectSolid":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangleSolid_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "Lshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    direction = np.random.randint(4)
                    obj_spec.append((('obj_{}'.format(k), 'Lshape_[{},{},{}]'.format(w,h,direction)), 'Attr'))
                elif chosen_concept_core == "Tshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Eshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(5, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Hshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Cshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(3, max_rect_size+2)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Ashape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+2)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Fshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                elif chosen_concept_core == "Randshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'randomShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "ARCshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'arcShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core in [
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]:
                    w, h = -1, -1 # let the canvas size drives here!
                    obj_spec.append((('obj_{}'.format(k), '{}_[{},{}]'.format(chosen_concept_core, w,h)), 'Attr'))
                else:
                    raise
            obj_spec = np.random.permutation(obj_spec).tolist()
        else:
            concept_collection = deepcopy(concept_collection)
            concept_collection.remove(focus_type)
            partition = np.sort(np.random.choice(n_objs, len(concept_collection)-1, replace=True))
            max_rect_size = canvas_size // 2
            for k in range(idx_start, n_objs + idx_start):
                if k == n_objs + idx_start - 1:
                    chosen_concept = focus_type
                else:
                    if len(concept_collection) == 1:
                        chosen_concept = concept_collection[0]
                    else:
                        gt = k-idx_start >= partition  # gt: greater_than_vector
                        if gt.any():
                            id = np.where(gt)[0][-1] + 1
                        else:
                            id = 0
                        chosen_concept = concept_collection[id]
                chosen_concept_core = get_c_core(chosen_concept)
                min_size, max_size = get_c_size(chosen_concept)
                if chosen_concept_core == "Line":
                    if min_size is None:
                        obj_spec.append((('obj_{}'.format(k), 'line_[-1,1,-1]'), 'Attr'))
                    else:
                        h = np.random.randint(min_size, max_size+1)
                        obj_spec.append((('obj_{}'.format(k), f'line_[{h},1,-1]'), 'Attr'))
                elif chosen_concept_core == "Rect":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangle_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "RectSolid":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangleSolid_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "Lshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    direction = np.random.randint(4)
                    obj_spec.append((('obj_{}'.format(k), 'Lshape_[{},{},{}]'.format(w,h,direction)), 'Attr'))
                elif chosen_concept_core == "Tshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Eshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(5, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Hshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Cshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(3, max_rect_size+2)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Ashape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+2)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Fshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                elif chosen_concept_core == "Randshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'randomShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "ARCshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'arcShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core in [
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]:
                    w, h = -1, -1 # let the canvas size drives here!
                    obj_spec.append((('obj_{}'.format(k), '{}_[{},{}]'.format(chosen_concept_core, w,h)), 'Attr'))
                else:
                    raise
            obj_spec = obj_spec[-1:] + np.random.permutation(obj_spec[:-1]).tolist()
    elif set(concept_collection).issubset({"SameColor", "IsTouch"}):
        if len(concept_collection) > 1:
            if is_conjuncture:
                # Hard code probability
                if color_avail == None:
                    random_color = np.random.randint(1, 10)
                else:
                    random_color = random.choice(color_avail)
                obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'IsTouch'))
                obj_spec.append((('obj_{}'.format(idx_start), f'color_[{random_color}]'), 'Attr'))
                obj_spec.append((('obj_{}'.format(idx_start+1), f'color_[{random_color}]'), 'Attr'))
            else:
                pass # TODO: not implemented
        else:
            if len(concept_collection) == 1:
                chosen_concept = concept_collection[0]
                if chosen_concept == "SameColor":
                    obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'SameColor'))
                else:
                    obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'IsTouch'))
    # complex shape.
    elif set(concept_collection).issubset({"RectE1a", "RectE1b", "RectE1c", 
                                           "RectE2a", "RectE2b", "RectE2c",
                                           "RectE3a", "RectE3b", "RectE3c", 
                                           "RectF1a", "RectF1b", "RectF1c", 
                                           "RectF2a", "RectF2b", "RectF2c",
                                           "RectF3a", "RectF3b", "RectF3c",}):
        chosen_concept = random.choice(concept_collection)
        if chosen_concept == "RectE1a" or chosen_concept == "RectF1a" or chosen_concept == "RectE1b" or chosen_concept == "RectF1b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, 9)
            char_h = np.random.randint(5, 9)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr')]
        elif chosen_concept == "RectE1c" or chosen_concept == "RectF1c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, 9)
            char_h = np.random.randint(5, 9)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE2a" or chosen_concept == "RectF2a" or chosen_concept == "RectE2b" or chosen_concept == "RectF2b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, 8)
            in_h = np.random.randint(4, 8)
            char_w = np.random.randint(3, 8)
            char_h = np.random.randint(5, 8)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside')]
        elif chosen_concept == "RectE2c" or chosen_concept == "RectF2c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, 8)
            in_h = np.random.randint(4, 8)
            char_w = np.random.randint(3, 8)
            char_h = np.random.randint(5, 8)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE3a" or chosen_concept == "RectF3a" or chosen_concept == "RectE3b" or chosen_concept == "RectF3b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(9, 17)
            out_h = np.random.randint(9, 17)
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
            out_w = np.random.randint(9, 17)
            out_h = np.random.randint(9, 17)
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
    else:
        raise Exception("concept_collection {} must be a subset of 'Line', 'Rect', 'Lshape', 'Randshape'!".format(concept_collection))
    return obj_spec


def sample_selector(
    dataset_engine,
    concept_avail=None,
    relation_avail=None,
    additional_concepts=None,
    n_concepts_range=None,
    relation_structure=None,
    max_n_distractors=0,
    min_n_distractors=0,
    canvas_size=8,
    color_avail=None,
    n_examples_per_task=5,
    max_n_trials=5,
    isplot=False,
):
    def is_exist_all(obj_full_relations, key):
        is_all = True
        for tuple_ele, lst in obj_full_relations.items():
            assert isinstance(tuple_ele, tuple)
            if key not in lst and (tuple_ele[0].startswith("obj_") and tuple_ele[1].startswith("obj_")):
                is_all = False
                break
        return is_all
    assert max_n_distractors >= min_n_distractors
    if concept_avail is None:
        concept_avail = [
            "Line", "Rect", "RectSolid", "Randshape", "ARCshape",
            "Lshape", "Tshape", "Eshape",
            "Hshape", "Cshape", "Ashape", "Fshape",
        ]
    if relation_avail is None:
        relation_avail = [
            "SameAll", "SameShape", "SameColor",
            "SameRow", "SameCol", 
            "SubsetOf", "IsInside", "IsTouch",
        ]
    allowed_shape_concept = concept_avail
    if additional_concepts is not None:
        allowed_shape_concept = allowed_shape_concept + additional_concepts
    concept_str_reverse_mapping = {
        "Line": "line", 
        "Rect": "rectangle", 
        "RectSolid": "rectangleSolid", 
        "Lshape": "Lshape", 
        "Tshape": "Tshape", 
        "Eshape": "Eshape", 
        "Hshape": "Hshape", 
        "Cshape": "Cshape", 
        "Ashape": "Ashape", 
        "Fshape": "Fshape",
        "Randshape": "randomShape",
        "ARCshape": "arcShape",
    }

    if relation_structure == "None":
        # Only concept dataset:
        n_concepts_range = (n_concepts_range, n_concepts_range) if (not isinstance(n_concepts_range, tuple)) and n_concepts_range > 0 else n_concepts_range
        assert isinstance(n_concepts_range, tuple)
        # Sample concepts:
        obj_spec_core = obj_spec_fun(
            concept_collection=concept_avail,
            min_n_objs=n_concepts_range[0],
            max_n_objs=n_concepts_range[1],
            canvas_size=canvas_size,
            color_avail=color_avail,
        )
        obj_id = len(obj_spec_core)
        refer_node_id = None
        structure = None
    else:
        structure_dict = {
            "2a": ["pivot", (0,1), "(refer)"],
            "2ai":["pivot:Rect", (1,0,"IsInside"), "(refer)"],
            "3a": ["pivot", (0,1), "(concept)", (1,2), "(refer)"],
            "3ai":["pivot:Rect", (1,0,"IsInside"), "(concept)", (1,2), "(refer)"],
            "3b": ["pivot", "pivot", (0,2), (1,2), "(refer)"],
            "4a": ["pivot", (0,1), "concept", (1,2), "(concept)", (2,3), "(refer)"],
            "4ai":["pivot:Rect", (1,0,"IsInside"), "(concept)", (1,2), "(concept)", (2,3), "(refer)"],
            "4b": ["pivot", "pivot", (0,2), (1,2), "(concept)", (2,3), "(refer)"],
        }

        # Sample pivot concept:
        structure_key = np.random.choice(relation_structure.split("+"))
        structure = structure_dict[structure_key]
        is_valid = False
        for i in range(3):
            obj_id = 0
            obj_spec_core = []
            refer_node_id = None
            relations_all = []
            for j, element in enumerate(structure):
                if isinstance(element, tuple):
                    if len(element) == 2:
                        if structure_key in ["2a", "3a"] and j == 1:
                            relation = np.random.choice(remove_elements(relation_avail, ["SameShape", "SameAll"]))
                        else:
                            relation = np.random.choice(remove_elements(relation_avail, ["SameAll"]))
                    elif len(element) == 3:
                        relation = np.random.choice(element[2].split("+"))
                    obj_spec_core.append(
                        [("obj_{}".format(element[0]),
                          "obj_{}".format(element[1])),
                          relation,
                    ])
                    relations_all.append(relation)
                elif element.startswith("pivot"):
                    if ":" in element:
                        concept_avail_core = element.split(":")[1].split("+")
                    else:
                        concept_avail_core = concept_avail
                    obj_spec = obj_spec_fun(
                        concept_collection=concept_avail_core,
                        min_n_objs=1,
                        max_n_objs=1,
                        canvas_size=canvas_size,
                        color_avail=color_avail,
                        idx_start=obj_id,
                    )[0]
                    obj_spec_core.append(obj_spec)
                    obj_id += 1
                elif element in ["concept", "refer", "(concept)", "(refer)"]:
                    if not element.startswith("("):
                        obj_spec = obj_spec_fun(
                            concept_collection=allowed_shape_concept,
                            min_n_objs=1,
                            max_n_objs=1,
                            canvas_size=canvas_size,
                            color_avail=color_avail,
                            idx_start=obj_id,
                        )[0]
                        obj_spec_core.append(obj_spec)
                    if element in ["refer", "(refer)"]:
                        assert refer_node_id is None
                        refer_node_id = obj_id
                    obj_id += 1
                else:
                    raise
            relations_unique = np.unique(relations_all)
            if len(relations_unique) == 1:
                if structure_key.startswith("3") and relations_unique[0] in ["SameColor"]:
                    pass
                elif relations_unique[0] in ["SameShape"] or structure_key.startswith("3"):
                    continue
            is_valid = True
            break
        if not is_valid:
            return []

    task = []
    for k in range(n_examples_per_task * 4):
        selector_dict = OrderedDict()
        if max_n_distractors > 0:
            n_distractors = np.random.choice(range(min_n_distractors, max_n_distractors + 1))
            obj_spec_distractor = obj_spec_fun(
                concept_collection=additional_concepts,
                min_n_objs=n_distractors,
                max_n_objs=n_distractors,
                canvas_size=canvas_size,
                color_avail=color_avail,
                idx_start=obj_id,
            )
        else:
            obj_spec_distractor = []

        obj_spec_all = obj_spec_core + obj_spec_distractor
        selector_dict = OrderedDict(obj_spec_all)

        is_valid = False
        for j in range(max_n_trials):
            canvas_dict = dataset_engine.sample_single_canvas_by_core_edges(
                selector_dict,
                allow_connect=True, is_plot=False, rainbow_prob=0.0,
                concept_collection=[concept_str_reverse_mapping[s] for s in concept_avail],
                parsing_check=True,
                color_avail=color_avail,
            )
            if canvas_dict == -1:
                continue
            else:
                is_valid = True
                if isplot:
                    canvas = Canvas(repre_dict=canvas_dict)
                    canvas.render()
                    plt.show()
                break

        if is_valid:
            image = to_one_hot(canvas_dict["image_t"])
            info = Dictionary({"obj_masks" : {}})
            for k, v in canvas_dict["node_id_map"].items():
                info["obj_masks"][k] = canvas_dict["id_object_mask"][v]
            info["obj_full_relations"] = canvas_dict["partial_relation_edges"]
            # Make sure that if the structure has "i" (IsInside), only the obj_1 is inside obj_0 and no other objects:
            if relation_structure != "None":
                n_objs = len(info["obj_masks"])
                for i in range(2, n_objs):
                    if (f"obj_{i}", "obj_0") in info["obj_full_relations"] and "IsInside" in info["obj_full_relations"][(f"obj_{i}", "obj_0")]:
                        p.print(f"obj_{i} is also inside the Rect!")
                        is_valid = False
                        break
                if len(info['obj_masks']) == 3 and is_exist_all(info['obj_full_relations'], key="SameColor"):
                    # is_valid = False
                    pass
                if len(info['obj_masks']) == 2 and is_exist_all(info['obj_full_relations'], key="SameShape"):
                    is_valid = False
            if not is_valid:
                continue

            info["obj_spec_core"] = obj_spec_core
            info["obj_spec_all"] = obj_spec_all
            info["obj_spec_distractor"] = obj_spec_distractor
            info["refer_node_id"] = "obj_{}".format(refer_node_id)
            info["structure"] = structure
            masks = None
            chosen_concepts = None
            if relation_structure != "None":
                target = info["obj_masks"][info["refer_node_id"]][None]
                example = ((image, target), masks, chosen_concepts, info)
            else:
                example = (image, masks, chosen_concepts, info)
            task.append(example)
        else:
            continue

        if len(task) >= n_examples_per_task:
            break
    return task





def generate_samples(
    dataset,
    obj_spec_fun,
    n_examples,
    mode,
    concept_collection,
    min_n_objs,
    max_n_objs,
    canvas_size,
    rainbow_prob=0.,
    allow_connect=True,
    parsing_check=False,
    focus_type=None,
    inspect_interval="auto",
    save_interval=-1,
    save_filename=None,
    **kwargs
):
    data = []
    if inspect_interval == "auto":
        inspect_interval = max(1, n_examples // 100)
    for i in range(int(n_examples * 150)):
        info = {}
        obj_spec = obj_spec_fun(
            concept_collection=concept_collection,
            min_n_objs=min_n_objs,
            max_n_objs=max_n_objs,
            canvas_size=canvas_size,
            allowed_shape_concept=kwargs["allowed_shape_concept"],
            color_avail=kwargs["color_avail"],
            focus_type=focus_type,
        )
        info["obj_spec"] = obj_spec

        if mode == "relation":
            concept_limits = {kwargs["concept_str_reverse_mapping"][get_c_core(c)]: get_c_size(c) for c in kwargs["allowed_shape_concept"] if get_c_size(c)[0] is not None}
            canvas_dict = dataset.sample_single_canvas_by_core_edges(
                OrderedDict(obj_spec),
                allow_connect=allow_connect,
                rainbow_prob=rainbow_prob,
                is_plot=False,
                concept_collection=[kwargs["concept_str_reverse_mapping"][s] for s in get_c_core(kwargs["allowed_shape_concept"])],
                parsing_check=parsing_check,
                color_avail=kwargs["color_avail"],
                concept_limits=concept_limits,
            )
        else:
            canvas_dict = dataset.sample_single_canvas_by_core_edges(
                OrderedDict(obj_spec),
                allow_connect=allow_connect,
                rainbow_prob=rainbow_prob,
                is_plot=False,
                parsing_check=parsing_check,
                color_avail=kwargs["color_avail"],
            )
        if canvas_dict != -1:
            info["node_id_map"] = canvas_dict["node_id_map"]
            info["id_object_mask"] = canvas_dict["id_object_mask"]
            n_sampled_objs = len(canvas_dict['id_object_mask'])
            if mode == "concept":
                if focus_type is None:
                    for k in range(n_sampled_objs):
                        # The order of id is the same as its first appearance in the obj_spec:
                        data.append((
                            to_one_hot(canvas_dict["image_t"]),
                            (canvas_dict['id_object_mask'][k][None],),
                            kwargs["concept_str_mapping"][obj_spec[k][0][1].split("_")[0]],
                            Dictionary(info),
                        ))
                        if len(data) >= n_examples:
                            break
                else:
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict['id_object_mask'][0][None],),
                        kwargs["concept_str_mapping"][obj_spec[0][0][1].split("_")[0]],
                        Dictionary(info),
                    ))
            elif mode == "concept-image":
                for k in range(n_sampled_objs):
                    # The order of id is the same as its first appearance in the obj_spec:
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict['id_object_mask'][k][None],),
                        "Image",
                        Dictionary(info),
                    ))
                    if len(data) >= n_examples:
                        break
            elif mode == "relation":
                if set(concept_collection).issubset({"Parallel", "Vertical"}):
                    def get_chosen_rel(canvas_dict, obj_ids):
                        chosen_direction = []
                        for id in obj_ids:
                            shape = canvas_dict['id_object_map'][id].shape
                            if shape[0] > shape[1]:
                                chosen_direction.append("0")
                            elif shape[0] < shape[1]:
                                chosen_direction.append("1")
                            else:
                                raise Exception("Line must have unequal height and width!")
                        if len(set(chosen_direction)) == 1:
                            chosen_concept = "Parallel"
                        else:
                            assert len(set(chosen_direction)) == 2
                            chosen_concept = "Vertical"
                        return chosen_concept

                    chosen_obj_ids = np.random.choice(n_sampled_objs, size=2, replace=False)
                    masks = list(canvas_dict['id_object_mask'].values())
                    chosen_obj_types = [obj_spec[id][0][1].split("_")[0] for id in chosen_obj_ids]
                    assert set(np.unique(chosen_obj_types)) == {"line"}
                    chosen_masks = [masks[id][None] for id in chosen_obj_ids]  # after: each mask has shape [1, H, W]
                    chosen_concept = get_chosen_rel(canvas_dict, chosen_obj_ids)
                    if chosen_concept not in concept_collection:
                        continue
                    # Consider all relations 
                    relations = []
                    for id1 in range(n_sampled_objs):
                        for id2 in range(id1 + 1, n_sampled_objs):
                            relation = get_chosen_rel(canvas_dict, [id1, id2])
                            relations.append((id1, id2, relation))
                    info["relations"] = relations
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        tuple(chosen_masks),
                        chosen_concept,
                        Dictionary(info),
                    ))
                if set(concept_collection).issubset({
                    "SameAll", "SameShape", "SameColor", 
                    "SameRow", "SameCol", "IsInside", "IsTouch",
                    "IsNonOverlapXY", "IsEnclosed",
                }):
                    masks = list(canvas_dict['id_object_mask'].values())
                    chosen_concept = obj_spec[0][1]
                    if chosen_concept not in concept_collection:
                        continue
                    chosen_obj_ids = [0,1] # we assum it is the first two objs have the relation type always!
                    if chosen_concept == "IsInside":
                        chosen_obj_ids = [1,0] # reverse it.
                        chosen_masks = [masks[id][None] for id in chosen_obj_ids]
                    else:
                        chosen_masks = [masks[id][None] for id in chosen_obj_ids]
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        tuple(chosen_masks),
                        chosen_concept,
                        Dictionary(info),
                    ))
            elif mode == "compositional-concept":
                # we need to filter.
                assert len(concept_collection) == 1 # we only allow 1.
                chosen_concept = concept_collection[0]
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
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict["image_t"][None].bool().float(),),
                        "Compositional-Image",
                        Dictionary(info),
                    ))
                    if inspect_interval != "None" and len(data) % inspect_interval == 0:
                        p.print("Finished generating {} out of {} examples.".format(len(data), n_examples))

        if inspect_interval != "None" and len(data) % inspect_interval == 0:
            p.print("Finished generating {} out of {} examples.".format(len(data), n_examples))
        if save_filename is not None and save_interval != -1 and len(data) % save_interval == 0:
            try_call(pdump, args=[data, save_filename])
            p.print("Save intermediate file at {}, with {} examples.".format(save_filename, len(data)))
        if len(data) >= n_examples:
            break
        if i > n_examples * 2 and len(data) < n_examples * 0.005:
            raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(data)))
    return data




# Line with "VerticalMid", "VerticalEdge", "VerticalNot", "Parallel":
def generate_lines_full_vertical_parallel(
    n_examples,
    concept_collection=["VerticalMid", "VerticalEdge", "VerticalNot", "Parallel"],
    min_n_objs=2,
    max_n_objs=4,
    canvas_size=16,
    min_size=2,
    max_size=None,
    color_avail=None,
    isplot=False,
):                  
    if color_avail is None:
        color_avail = [1,2,3,4,5,6,7,8,9]
    data = []
    if max_size is None:
        max_size = canvas_size
    for i in range(int(n_examples*3)):
        if len(data) % 500 == 0:
            p.print(len(data))
        image = torch.zeros(1, canvas_size, canvas_size)
        # Sample relation from concept_collection:
        relation = np.random.choice(concept_collection)
        if relation == "Parallel":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask2, pos2, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos2[0]) or (direction == 1 and pos1[1] == pos2[1]):
                # The two lines cannot be on the same straight line:
                continue
        elif relation == "VerticalMid":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "VerticalEdge":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "VerticalSepa":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_not_touching(mask1, direction=1-direction, min_size=min_size, max_size=max_size, canvas_size=canvas_size, pos=pos1, min_distance=1)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "Fshape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(5, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask3, pos3, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "Eshape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(5, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size, set_orientation=0)
            if pos_new is None:
                continue
            image, mask3, pos3, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size, set_orientation=0)
            if pos_new is None:
                continue
            image, mask4, pos4, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            if (mask3*mask4).sum()>0:
                continue
        elif relation == "Cshape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size,set_orientation=0)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size, set_orientation=1)
            if pos_new is None:
                continue
            image, mask3, pos3, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "Ashape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(5, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask3, pos3, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask4, pos4, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos4[0]) or (direction == 1 and pos1[1] == pos4[1]):
                # The two lines cannot be on the same straight line:
                continue
        elif relation == "Hshape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask3, pos3, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos3[0]) or (direction == 1 and pos1[1] == pos3[1]):
                # The two lines cannot be on the same straight line:
                continue
            if (mask1*mask3).sum()>0 or (mask2*mask3).sum()>0:
                continue
        elif relation == "Pshape":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask3, pos3, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos3[0]) or (direction == 1 and pos1[1] == pos3[1]):
                # The two lines cannot be on the same straight line:
                continue
            if (mask1*mask3).sum()>0 or (mask2*mask3).sum()>0:
                continue
        elif relation == "Rect":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size,set_orientation=0)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size, set_orientation=1)
            if pos_new is None:
                continue
            image, mask3, pos3, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask4, pos4, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos4[0]) or (direction == 1 and pos1[1] == pos4[1]):
                # The two lines cannot be on the same straight line:
                continue
        
        # Randomly permute the mask
        if np.random.choice([0,1]) == 1:
            mask1, mask2 = mask2, mask1
            pos1, pos2 = pos2, pos1

        info = {}
        info["id_object_mask"] = {0: mask1, 1: mask2}
        info["id_object_pos"] = {0: pos1, 1: pos2}
        info["obj_spec"] = [(("obj_0", "line"), "Attr"), (("obj_1", "line"), "Attr")]
        info["node_id_map"] = {"obj_0": 0, "obj_1": 1}
        obj_idx = 2
        if relation in ['Fshape','Cshape','Hshape']:
            info["id_object_mask"] = {0: mask1, 1: mask2, 2: mask3}
            info["id_object_pos"] = {0: pos1, 1: pos2, 2: pos3}
            info["obj_spec"] = [(("obj_0", "line"), "Attr"), (("obj_1", "line"), "Attr"), (("obj_1", "line"), "Attr")]
            info["node_id_map"] = {"obj_0": 0, "obj_1": 1, "obj_2": 2}
            obj_idx=3
        if relation in ['Eshape','Ashape','Rect']:
            info["id_object_mask"] = {0: mask1, 1: mask2, 2: mask3, 3:mask4}
            info["id_object_pos"] = {0: pos1, 1: pos2, 2: pos3, 3:pos4}
            info["obj_spec"] = [(("obj_0", "line"), "Attr"), (("obj_1", "line"), "Attr"), (("obj_1", "line"), "Attr"),(("obj_1", "line"), "Attr")]
            info["node_id_map"] = {"obj_0": 0, "obj_1": 1, "obj_2": 2, "obj_3": 3}
            obj_idx=4

        # Add distractors with position and color:
        max_n_distractors = max_n_objs - min_n_objs
        if max_n_distractors > 0:
            n_distractors = np.random.randint(1, max_n_distractors+1)
            mask = (image != 0).float()
            for k in range(n_distractors):
                mask = (image != 0).float()
                pos_new = get_pos_new_not_touching(mask, direction=np.random.choice([0,1]), min_size=min_size, max_size=max_size, canvas_size=canvas_size, min_distance=0)
                image, obj_mask, obj_pos, _ = get_line(image, direction=None, pos=pos_new, color_avail=color_avail)
                # Update info
                info["id_object_mask"][obj_idx] = obj_mask
                info["id_object_pos"][obj_idx] = obj_pos
                info["obj_spec"].append(((f"obj_{obj_idx}", "line"), "Attr"))
                info["node_id_map"][f"obj_{obj_idx}"] = obj_idx
                obj_idx += 1
        
        # Get all relations
        relations = []
        n_objs = len(info["id_object_mask"])
        for id1 in range(n_objs):
            for id2 in range(id1 + 1, n_objs):
                rel = get_chosen_line_rel(info["id_object_pos"][id1], info["id_object_pos"][id2])
                relations.append((id1, id2, rel))
        info["relations"] = relations

        data.append(
            (to_one_hot(image)[0],
             (mask1, mask2),
             relation,
             Dictionary(info),
            )
        )
        if len(data) >= n_examples:
            break
        if isplot:
            visualize_matrices(image)
            p.print(relation)
            toplot=[mask1.squeeze(), mask2.squeeze()]
            if relation in ['Fshape']: toplot+=[mask3.squeeze()]
            if relation in ['Eshape']: toplot+=[mask4.squeeze()]
            plot_matrices(toplot)
            print('\n\n')
    return data


def get_line(image, direction=None, pos=None, min_size=2, max_size=10, color_avail=[1,2]):
    """
    Direction: 0: horizontal; 1: vertical.
    """
    canvas_size = image.shape[-1]
    mask = torch.zeros(image.shape)
    if pos is None:
        assert direction is not None
        for i in range(10):
            if direction == -1:
                direction_core = np.random.choice([0,1])
            else:
                direction_core = direction
            if direction_core == 0:
                # horizontal:
                h = 1
                w = np.random.randint(min_size, max_size+1)
            elif direction_core == 1:
                h = np.random.randint(min_size, max_size+1)
                w = 1
            row_start = 0
            row_end = canvas_size - h
            if row_end <= row_start:
                continue
            row = np.random.randint(row_start, row_end+1)

            col_start = 0
            col_end = canvas_size - w
            if col_end <= col_start:
                continue
            col = np.random.randint(col_start, col_end+1)
            pos = (row, col, h, w)
    else:
        row, col, h, w = pos

    color = np.random.choice(color_avail)
    image[..., row: row+h, col: col+w] = color
    mask[..., row: row+h, col: col+w] = 1
    return image, mask, pos, direction


def get_pos_new_mid(pos, direction, min_size, canvas_size):
    pos_mid = (pos[0] + pos[2]//2, pos[1] + pos[3]//2)
    pos_new = None
    for k in range(10):
        orientation = np.random.choice([0,1])
        if direction == 1:
            # second line is horizontal:
            if orientation == 0:
                # second line is on the right:
                if canvas_size-pos_mid[1] <= min_size:
                    continue
                pos_new = (pos_mid[0], pos_mid[1]+1, 1, np.random.randint(min_size, canvas_size-pos_mid[1]+1))
            elif orientation == 1:
                # second line is on the left:
                if pos[1] <= min_size:
                    continue
                w_mid = np.random.randint(min_size, pos[1]+1)
                pos_new = (pos_mid[0], pos_mid[1] - w_mid, 1, w_mid)
        elif direction == 0:
            # second line is vertical:
            if orientation == 0:
                # second line is on the bottom:
                if canvas_size-pos_mid[0] <= min_size:
                    continue
                pos_new = (pos_mid[0]+1, pos_mid[1], np.random.randint(min_size, canvas_size-pos_mid[0]+1), 1)
            elif orientation == 1:
                # second line is on the top:
                if pos[0] <= min_size:
                    continue
                h_mid = np.random.randint(min_size, pos[0]+1)
                pos_new = (pos_mid[0] - h_mid, pos_mid[1], h_mid, 1)
        if pos_new is not None:
            break
    return pos_new


def get_pos_new_edge(pos, direction, min_size, canvas_size, set_orientation=None):
    pos_new = None
    for k in range(10):
        orientation = np.random.choice([0,1]) if set_orientation is None else set_orientation
        edge_side = np.random.choice([0,1])
        if direction == 1:
            # second line is horizontal:
            if orientation == 0:
                # second line is on the right:
                if canvas_size-pos[1] <= min_size:
                    continue
                if edge_side == 0:
                    pos_new = (pos[0], pos[1]+1, 1, np.random.randint(min_size, canvas_size-pos[1]+1))
                elif edge_side == 1:
                    pos_new = (pos[0]+pos[2]-1, pos[1]+1, 1, np.random.randint(min_size, canvas_size-pos[1]+1))
            elif orientation == 1:
                # second line is on the left:
                if pos[1] <= min_size:
                    continue
                w_mid = np.random.randint(min_size, pos[1]+1)
                if edge_side == 0:
                    pos_new = (pos[0], pos[1] - w_mid, 1, w_mid)
                elif edge_side == 1:
                    pos_new = (pos[0]+pos[2]-1, pos[1] - w_mid, 1, w_mid)
        elif direction == 0:
            # second line is vertical:
            if orientation == 0:
                # second line is on the bottom:
                if canvas_size-pos[0] <= min_size:
                    continue
                if edge_side == 0:
                    pos_new = (pos[0]+1, pos[1], np.random.randint(min_size, canvas_size-pos[0]+1), 1)
                elif edge_side == 1:
                    pos_new = (pos[0]+1, pos[1]+pos[3]-1, np.random.randint(min_size, canvas_size-pos[0]+1), 1)
            elif orientation == 1:
                # second line is on the top:
                if pos[0] <= min_size:
                    continue
                h_mid = np.random.randint(min_size, pos[0]+1)
                if edge_side == 0:
                    pos_new = (pos[0] - h_mid, pos[1], h_mid, 1)
                elif edge_side == 1:
                    pos_new = (pos[0] - h_mid, pos[1]+pos[3]-1, h_mid, 1)
        if pos_new is not None:
            break
    return pos_new


def get_pos_new_not_touching(mask1, direction, min_size, max_size, canvas_size, pos=None, min_distance=0):
    mask = deepcopy(mask1)
    if min_distance > 0:
        mask[...,max(0,pos[0]-min_distance):pos[0]+pos[2]+min_distance, max(0, pos[1]-min_distance):pos[1]+pos[3]+min_distance] = 1
    for k in range(30):
        if direction == 0:
            # horizontal:
            h = 1
            w = np.random.randint(min_size, max_size+1)
        elif direction == 1:
            h = np.random.randint(min_size, max_size+1)
            w = 1
        row_start = 0
        row_end = canvas_size - h
        if row_end <= row_start:
            continue
        row = np.random.randint(row_start, row_end)

        col_start = 0
        col_end = canvas_size - w
        if col_end <= col_start:
            continue
        col = np.random.randint(col_start, col_end)

        mask2 = torch.zeros(mask1.shape)
        mask2[..., row: row+h, col: col+w] = 1
        if (mask + mask2 == 2).any():
            continue
        else:
            break
    pos_new = (row, col, h, w)
    return pos_new



def get_chosen_line_rel(pos1, pos2):
    chosen_direction = []
    # "0" corresponds to upright, "1" corresponds to horizontal
    chosen_direction.append("0" if pos1[2] > pos1[3] else "1")
    chosen_direction.append("0" if pos2[2] > pos2[3] else "1")
    if len(set(chosen_direction)) == 1:
        chosen_concept = "Parallel"
    else:
        assert len(set(chosen_direction)) == 2
        # Set the horizontal and upright 
        if pos1[2] > pos1[3]:
            hori = pos2
            upr = pos1
        else:
            hori = pos1
            upr = pos2
        # Determine whether the intersection is more like a T shape or a rotated T shape
        # by checking if the upright line falls between horizontal line's left and right edges
        isT = upr[1] >= hori[1] and upr[1] < hori[1] + hori[3]
        isRotT= hori[0] >= upr[0] and hori[0] < upr[0] + upr[2]
        if isT:
            # First check for separate by comparing top edge of vertical with hori
            dist1 = abs(upr[0] - hori[0])
            dist2 = abs(upr[0] + upr[2] - 1 - hori[0])
            if min(dist1, dist2) > 1:
                chosen_concept = "VerticalSepa"
            else:
                # Check where the lines are touching w.r.t. horizontal line
                if upr[1] == hori[1] or upr[1] == hori[1] + hori[3] - 1:
                    chosen_concept = "VerticalEdge"
                else:
                    chosen_concept = "VerticalMid"
        elif isRotT:
            dist1 = abs(hori[1] - upr[1])
            dist2 = abs(hori[1] + hori[3] - 1 - upr[1])
            if min(dist1, dist2) > 1:
                chosen_concept = "VerticalSepa"
            else:
                # Check where the lines are touching w.r.t. vertical line
                if hori[0] == upr[0] or hori[0] == upr[0] + upr[2] - 1:
                    chosen_concept = "VerticalEdge"
                else:
                    chosen_concept = "VerticalMid"
        else:
            chosen_concept = "VerticalSepa"
    return chosen_concept



class ConceptDataset(Dataset):
    """Concept Dataset for learning basic concepts for ARC.

    mode:
        Concepts:  E(x; a; c)
            "Pixel": one or many pixels
            "Line": one or many lines
            "Rect": hollow rectangles
            "{}+{}+...": each "{}" can be a concept.

        Symmetries: E(x; a; c)
            "hFlip", "vFlip": one image where some object has property of symmetry w.r.t. hflip
            "Rotate": one image where some object has property of symmetry w.r.t. rotation.

        Relations: E(x; a1, a2; c)
            "Vertical": lines where some of them are vertical
            "Parallel": lines where some of them are parallel
            "Vertical+Parallel": lines where some of them are vertical or parallel
            "IsInside": obj_1 is inside obj_2
            "SameRow": obj_1 and obj_2 are at the same row
            "SameCol": obj_1 and obj_2 are at the same column

        Operations: E(x1,x2; a1,a2; c1,c2)
            "RotateA+vFlip(Line+Rect)": two images where some object1 in image1 is rotated or vertically-flipped w.r.t. some object2 in image2, and the objects are chosen from Line or Rect.
            "hFlip(Lshape)", "vFlip(Lshape+Line)": two images where some object1 in image1 is flipped w.r.t. some object2 in image2.

        ARC+:
            "arc^{}": ARC images with property "{}" masked as above.
        ""
    """
    def __init__(
        self,
        mode,
        canvas_size=8,
        n_examples=10000,
        rainbow_prob=0.,
        data=None,
        idx_list=None,
        concept_collection=None,
        allowed_shape_concept=None,
        w_type="image+mask",
        color_avail="-1",
        min_n_distractors=0,
        max_n_distractors=-1,
        n_operators=1,
        allow_connect=True,
        parsing_check=False,
        focus_type=None,
        transform=None,
        save_interval=-1,
        save_filename=None,
    ):
        if allowed_shape_concept is None:
            allowed_shape_concept=[
                "Line", "Rect", "RectSolid", "Lshape", "Randshape", "ARCshape",
                "Tshape", "Eshape",
                "Hshape", "Cshape", "Ashape", "Fshape",
                "RectE1a", "RectE1b", "RectE1c", 
                "RectE2a", "RectE2b", "RectE2c",
                "RectE3a", "RectE3b", "RectE3c", 
                "RectF1a", "RectF1b", "RectF1c", 
                "RectF2a", "RectF2b", "RectF2c",
                "RectF3a", "RectF3b", "RectF3c",
            ]
        self.mode = mode
        self.canvas_size = canvas_size
        self.rainbow_prob = rainbow_prob
        self.n_examples = n_examples
        self.allowed_shape_concept = allowed_shape_concept
        self.min_n_distractors = min_n_distractors
        self.max_n_distractors = max_n_distractors
        self.n_operators = n_operators
        self.w_type = w_type
        self.allow_connect = allow_connect
        self.parsing_check = parsing_check
        self.focus_type = focus_type
        if isinstance(color_avail, str):
            if color_avail == "-1":
                self.color_avail = None
            else:
                self.color_avail = [int(c) for c in color_avail.split(",")]
                for c in self.color_avail:
                    assert c >= 1 and c <= 9
        else:
            self.color_avail = color_avail
        if idx_list is None:
            assert data is None
            if mode.startswith("arc^"):
                if "(" in mode:
                    self.data = []
                    # Operator:
                    concept_raw = mode.split("(")[0].split("+")
                    concept_collection = []
                    for c in concept_raw:
                        if "^" in c:
                            concept_collection.append(c.split("^")[1])
                        else:
                            concept_collection.append(c)
                    self.concept_collection = concept_collection
                    arcDataset = ARCDataset(
                        n_examples=n_examples*2,
                        canvas_size=canvas_size,
                    )
                    babyArcDataset = BabyARCDataset(
                        pretrained_obj_cache=os.path.join(get_root_dir(), 'datasets/arc_objs.pt'),
                        save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                        object_limit=None,
                        noise_level=0,
                        canvas_size=canvas_size,
                    )
                    if set(self.concept_collection).issubset({"RotateA", "RotateB", "RotateC", 
                                                              "hFlip", "vFlip", "DiagFlipA", "DiagFlipB", 
                                                              "Identity", 
                                                              "Move"}):
                        for arc_example_one_hot in arcDataset:
                            arc_image = torch.zeros_like(arc_example_one_hot[0])
                            for i in range(0, 10):
                                arc_image += arc_example_one_hot[i]*i
                            arc_image = arc_image.type(torch.int32)

                            repre_dict = babyArcDataset.sample_task_canvas_from_arc(
                                arc_image,
                                color=np.random.choice([True, False], p=[0.6, 0.4]),
                                is_plot=False,
                            )
                            if repre_dict == -1:
                                continue
                            in_canvas = Canvas(repre_dict=repre_dict)

                            # Operate on the input:
                            if len(list(repre_dict["node_id_map"].keys())) == 0:
                                continue # empty arc canvas
                            chosen_obj_key = np.random.choice(list(repre_dict["node_id_map"].keys()))
                            chosen_obj_id = repre_dict["node_id_map"][chosen_obj_key]
                            chosen_op = np.random.choice(self.concept_collection)
                            if chosen_op in ["Identity"]:
                                inplace = True if random.random() < 0.5 else False
                                out_canvas_list, concept = OperatorEngine().operator_identity(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    inplace=inplace,
                                )
                                if out_canvas_list == -1:
                                    continue
                            elif chosen_op in ["Move"]:
                                # create operator spec as move is a complex operator
                                move_spec = OperatorMoveSpec(
                                                autonomous=False,
                                                direction=random.randint(0,3), 
                                                distance=-1, 
                                                hit_type=None, # either wall, agent or None
                                                linkage_move=False, 
                                                linkage_move_distance_ratio=None,
                                            )
                                out_canvas_list, concept = OperatorEngine().operator_move(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    [[move_spec]], 
                                    allow_overlap=False, 
                                    allow_shape_break=False,
                                    allow_connect=self.allow_connect,
                                    allow_stay=False,
                                )
                                if out_canvas_list == -1:
                                    continue
                            elif chosen_op in ["RotateA", "RotateB", "RotateC", "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"]:
                                out_canvas_list, concept = OperatorEngine().operate_rotate(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    operator_tag=f"#{chosen_op}",
                                    allow_connect=self.allow_connect,
                                    allow_shape_break=False,
                                )
                                if out_canvas_list == -1:
                                    continue
                            else:
                                raise Exception(f"operator={chosen_op} is not supported!")

                            # Add to self.data:
                            in_canvas_dict = in_canvas.repr_as_dict()
                            out_canvas_dict = out_canvas_list[0].repr_as_dict()
                            in_mask = in_canvas_dict["id_object_mask"][chosen_obj_id][None]
                            out_mask = out_canvas_dict["id_object_mask"][chosen_obj_id][None]
                            self.data.append(
                                ((to_one_hot(in_canvas_dict["image_t"]), to_one_hot(out_canvas_dict["image_t"])),
                                 (in_mask, out_mask),
                                 chosen_op,
                                 Dictionary({}),
                                )
                            )
                            if len(self.data) >= n_examples:
                                break
                            if i > n_examples * 2 and len(self.data) < n_examples * 0.05:
                                raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(self.data)))
                else:
                    mode_core = mode.split("^")[1]
                    self.concept_collection = mode_core.split("+")
                    dataset = ARCDataset(
                        n_examples=n_examples,
                        canvas_size=canvas_size,
                    )
                    examples_all = []
                    masks_all = []
                    concepts_all = []
                    examples = dataset.data
                    examples_argmax = examples.argmax(1)
                    self.data = []
                    for i in range(len(examples)):
                        concept_dict = seperate_concept(examples_argmax[i])
                        masks, concepts = get_masks(concept_dict, allowed_concepts=self.concept_collection, canvas_size=canvas_size)
                        if masks is not None:
                            for mask, concept in zip(masks, concepts):
                                self.data.append((
                                        examples[i],
                                        (mask,),
                                        concept,
                                        Dictionary({}),
                                    )
                                )
            else:
                if "(" in mode:
                    # Operator:
                    self.concept_collection = mode.split("(")[0].split("+")
                    input_concepts = mode.split("(")[1][:-1].split("+")
                else:
                    self.concept_collection = mode.split("-")[-1].split("+")
                    input_concepts = [""]
                dataset = BabyARCDataset(
                    pretrained_obj_cache=os.path.join(get_root_dir(), 'arc_objs.pt'),
                    save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                    object_limit=None,
                    noise_level=0,
                    canvas_size=canvas_size,
                )
                concept_str_mapping = {
                    "line": "Line", 
                    "rectangle": "Rect", 
                    "rectangleSolid": "RectSolid",
                    "Lshape": "Lshape", 
                    "Tshape": "Tshape", 
                    "Eshape": "Eshape", 
                    "Hshape": "Hshape", 
                    "Cshape": "Cshape", 
                    "Ashape": "Ashape", 
                    "Fshape": "Fshape",
                    "randomShape": "Randshape",
                    "arcShape": "ARCshape"}  # Mapping between two conventions
                concept_str_reverse_mapping = {
                    "Line": "line", 
                    "Rect": "rectangle", 
                    "RectSolid": "rectangleSolid", 
                    "Lshape": "Lshape", 
                    "Tshape": "Tshape", 
                    "Eshape": "Eshape", 
                    "Hshape": "Hshape", 
                    "Cshape": "Cshape", 
                    "Ashape": "Ashape", 
                    "Fshape": "Fshape",
                    "Randshape": "randomShape",
                    "ARCshape": "arcShape"}  # Mapping between two conventions
                composite_concepts = [                
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]
                for c in composite_concepts:
                    concept_str_mapping[c] = c
                    concept_str_reverse_mapping[c] = c

                if set(get_c_core(self.concept_collection)).issubset({
                    "Image"
                }):
                    # Image is a collection of all shapes.
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 1 + max_n_distractors # 1 is for the core concept itself.
                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun,
                        n_examples=n_examples,
                        mode="concept-image",
                        concept_collection=["Line", "Rect", "Lshape", 
                                            "RectSolid", "Randshape", "ARCshape", 
                                            "Tshape", "Eshape", 
                                            "Hshape", "Cshape", "Ashape", "Fshape"],
                        min_n_objs=1+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=["Line", "Rect", "Lshape", 
                                               "RectSolid", "Randshape", "ARCshape", 
                                               "Tshape", "Eshape", 
                                               "Hshape", "Cshape", "Ashape", "Fshape"],
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                        save_interval=10,
                        save_filename=save_filename,
                    )
                elif set(self.concept_collection).issubset({"VerticalMid", "VerticalEdge", "VerticalSepa", "Parallel",
                                                            "Fshape","Eshape",'Cshape','Ashape','Hshape','Rect','Pshape'}):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 2 + max_n_distractors # 2 is for the core concept itself.
                    self.data = generate_lines_full_vertical_parallel(
                        n_examples=n_examples,
                        concept_collection=self.concept_collection,
                        min_n_objs=2+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        min_size=3,
                        max_size=canvas_size-2,
                        color_avail=self.color_avail,
                        isplot=False,
                    )
                elif set(self.concept_collection).issubset({
                    "SameAll", "SameShape", "SameColor", 
                    "SameRow", "SameCol", "IsInside", 
                    "IsTouch", "IsNonOverlapXY",
                    "IsEnclosed",
                }):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 2 + max_n_distractors # 2 is for the core relation itself.
                    def obj_spec_fun_re(
                        concept_collection, min_n_objs, max_n_objs, 
                        canvas_size, allowed_shape_concept=None, 
                        color_avail=None,
                        focus_type=None,
                    ):
                        assert allowed_shape_concept != None
                        n_objs = np.random.randint(min_n_objs, max_n_objs+1)
                        # two slots are for the relation
                        sampled_relation = np.random.choice(concept_collection)
                        obj_spec = [(('obj_0', 'obj_1'), sampled_relation)]
                        max_rect_size = canvas_size//2
                        # choose distractors
                        for k in range(2, n_objs):
                            # choose a distractor shape
                            distractor_shape = np.random.choice(allowed_shape_concept)
                            if distractor_shape == "Line":
                                obj_spec += [(('obj_{}'.format(k), 'line_[-1,-1,-1]'), 'Attr')]
                            elif distractor_shape == "Rect":
                                obj_spec += [(('obj_{}'.format(k), 'rectangle_[-1,-1]'), 'Attr')]
                            elif distractor_shape == "RectSolid":
                                obj_spec += [(('obj_{}'.format(k), 'rectangleSolid_[-1,-1]'), 'Attr')]
                            elif distractor_shape == "Lshape":
                                obj_spec += [(('obj_{}'.format(k), 'Lshape_[-1,-1,-1]'), 'Attr')]
                            elif distractor_shape == "Tshape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Eshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(5, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Hshape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Cshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Ashape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(4, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Fshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(4, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                            elif distractor_shape == "Randshape":
                                max_rect_size = canvas_size // 2
                                w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                obj_spec += [(('obj_{}'.format(k), f'randomShape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "ARCshape":
                                max_rect_size = canvas_size // 2
                                w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                obj_spec += [(('obj_{}'.format(k), f'arcShape_[{w},{h}]'), 'Attr')]
                        return obj_spec
                    if len(input_concepts) == 1 and input_concepts[0] == "":
                        _shape_concept=[c for c in self.allowed_shape_concept]
                    else:
                        _shape_concept=[c for c in input_concepts]

                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun_re,
                        n_examples=n_examples,
                        mode="relation",
                        concept_collection=self.concept_collection,
                        min_n_objs=2+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=_shape_concept,
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                    )
                elif set(self.concept_collection).issubset({
                    "RotateA", "RotateB", "RotateC", 
                    "hFlip", "vFlip", "DiagFlipA", 
                    "DiagFlipB", "Identity", "Move"
                }):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 1 + max_n_distractors # 1 is for the core operator itself.
                    self.data = []
                    for i in range(self.n_examples * 5):
                        # Initialize input concept instance:
                        obj_spec = obj_spec_fun(
                            concept_collection=input_concepts,
                            min_n_objs=1+self.min_n_distractors,
                            max_n_objs=max_n_objs,
                            canvas_size=canvas_size,
                        )
                        # get the number of the objects
                        operatable_obj_set = set([])
                        for spec in obj_spec:
                            if spec[1] == "Attr":
                                operatable_obj_set.add(spec[0][0])
                            else:
                                operatable_obj_set.add(spec[0][0])
                                operatable_obj_set.add(spec[0][1])
                        operatable_obj_set = list(operatable_obj_set)
                        # let us enable distractors
                        if set(input_concepts).issubset({"SameColor", "IsTouch"}):
                            n_distractors = np.random.randint(0, max_n_distractors+1)
                            max_rect_size = canvas_size//2
                            for i in range(n_distractors):
                                k = i+len(operatable_obj_set)
                                distractor_shape = np.random.choice(self.allowed_shape_concept)
                                if distractor_shape == "Line":
                                    obj_spec += [(('obj_{}'.format(k), 'line_[-1,-1,-1]'), 'Attr')]
                                elif distractor_shape == "Rect":
                                    obj_spec += [(('obj_{}'.format(k), 'rectangle_[-1,-1]'), 'Attr')]
                                elif distractor_shape == "RectSolid":
                                    obj_spec += [(('obj_{}'.format(k), 'rectangleSolid_[-1,-1]'), 'Attr')]
                                elif distractor_shape == "Lshape":
                                    obj_spec += [(('obj_{}'.format(k), 'Lshape_[-1,-1,-1]'), 'Attr')]
                                elif distractor_shape == "Tshape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Eshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(5, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Hshape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Cshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Ashape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(4, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Fshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(4, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]     
                                elif distractor_shape == "Randshape":
                                    max_rect_size = canvas_size // 2
                                    w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                    obj_spec += [(('obj_{}'.format(k), f'randomShape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "ARCshape":
                                    max_rect_size = canvas_size // 2
                                    w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                    obj_spec += [(('obj_{}'.format(k), f'arcShape_[{w},{h}]'), 'Attr')]
                        # get all objects include distractors
                        all_obj_set = set([])
                        for spec in obj_spec:
                            if spec[1] == "Attr":
                                all_obj_set.add(spec[0][0])
                            else:
                                all_obj_set.add(spec[0][0])
                                all_obj_set.add(spec[0][1])

                        repre_dict = dataset.sample_single_canvas_by_core_edges(
                            OrderedDict(obj_spec),
                            allow_connect=self.allow_connect,
                            rainbow_prob=rainbow_prob,
                            is_plot=False,
                            color_avail=self.color_avail,
                        )
                        if repre_dict == -1:
                            continue
                        in_canvas = Canvas(repre_dict=repre_dict)

                        # Operate on the input:
                        chosen_obj_id = np.random.choice(len(operatable_obj_set))
                        chosen_obj_name = operatable_obj_set[chosen_obj_id]
                        chosen_op = np.random.choice(self.concept_collection)
                        if chosen_op in ["Identity"]:
                            inplace = True if random.random() < 0.5 else False
                            out_canvas_list, concept = OperatorEngine().operator_identity(
                                [in_canvas],
                                [[chosen_obj_name]],
                                inplace=inplace,
                            )
                            if out_canvas_list == -1:
                                continue
                        elif chosen_op in [
                            "RotateA", "RotateB", "RotateC", 
                            "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"
                        ]:
                            out_canvas_list, concept = OperatorEngine().operate_rotate(
                                [in_canvas],
                                [[chosen_obj_name]],
                                operator_tag=f"#{chosen_op}",
                                allow_connect=self.allow_connect,
                                allow_shape_break=False,
                            )
                            if out_canvas_list == -1:
                                continue
                        elif chosen_op in ["Move"]:
                            # create operator spec as move is a complex operator
                            move_spec = OperatorMoveSpec(
                                            autonomous=False,
                                            direction=random.randint(0,3), 
                                            distance=-1, 
                                            hit_type=None, # either wall, agent or None
                                            linkage_move=False, 
                                            linkage_move_distance_ratio=None,
                                        )
                            out_canvas_list, concept = OperatorEngine().operator_move(
                                [in_canvas],
                                [[chosen_obj_name]],
                                [[move_spec]], 
                                allow_overlap=False, 
                                allow_shape_break=False,
                                allow_connect=self.allow_connect,
                                allow_stay=False,
                            )
                            if out_canvas_list == -1:
                                continue
                        else:
                            raise Exception(f"operator={chosen_op} is not supported!")
                        
                        if n_operators > 1:
                            # operator distractor can act on all objects
                            addition_operators = min(len(all_obj_set)-1,n_operators-1) # we need to have minimum number of objs
                            operated_obj_name = set([])
                            operated_obj_name.add(chosen_obj_name)
                            
                            exclude_ops = set([chosen_op])
                            
                            # we need to operate on other objects.
                            for _ in range(n_operators-1):
                                addition_obj_set = set(all_obj_set) - operated_obj_name
                                addition_obj_name = np.random.choice(list(addition_obj_set))
                                
                                addition_ops = set(self.concept_collection) - exclude_ops
                                addition_op = np.random.choice(list(addition_ops))
                                exclude_ops.add(addition_op)
                                
                                # operate the the previous ouput canvas
                                if addition_op in ["Identity"]:
                                    inplace = True if random.random() < 0.5 else False
                                    out_canvas_list, concept = OperatorEngine().operator_identity(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        inplace=inplace,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                elif addition_op in ["RotateA", "RotateB", "RotateC", "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"]:
                                    out_canvas_list, concept = OperatorEngine().operate_rotate(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        operator_tag=f"#{addition_op}",
                                        allow_connect=self.allow_connect,
                                        allow_shape_break=False,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                elif addition_op in ["Move"]:
                                    # create operator spec as move is a complex operator
                                    move_spec = OperatorMoveSpec(
                                                    autonomous=False,
                                                    direction=random.randint(0,3), 
                                                    distance=-1, 
                                                    hit_type=None, # either wall, agent or None
                                                    linkage_move=False, 
                                                    linkage_move_distance_ratio=None,
                                                )
                                    out_canvas_list, concept = OperatorEngine().operator_move(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        [[move_spec]], 
                                        allow_overlap=False, 
                                        allow_shape_break=False,
                                        allow_connect=self.allow_connect,
                                        allow_stay=False,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                else:
                                    raise Exception(f"operator={addition_op} is not supported!")
                                operated_obj_name.add(addition_obj_name)
                        if out_canvas_list == -1:
                            continue
                        # Add to self.data:
                        in_canvas_dict = in_canvas.repr_as_dict()
                        out_canvas_dict = out_canvas_list[0].repr_as_dict()
                        
                        in_mask = in_canvas_dict["id_object_mask"][in_canvas_dict["node_id_map"][chosen_obj_name]][None]
                        out_mask = out_canvas_dict["id_object_mask"][in_canvas_dict["node_id_map"][chosen_obj_name]][None]
                        # TODO: remove deprecated codes.
                        # in_mask = in_canvas_dict["id_object_mask"][chosen_obj_id][None]
                        # out_mask = out_canvas_dict["id_object_mask"][chosen_obj_id][None]
                        info = {"obj_spec": obj_spec}
                        self.data.append(
                            ((to_one_hot(in_canvas_dict["image_t"]), to_one_hot(out_canvas_dict["image_t"])),
                             (in_mask, out_mask),
                             chosen_op,
                             Dictionary(info),
                            )
                        )
                        if len(self.data) >= n_examples:
                            break
                        if i > n_examples * 2 and len(self.data) < n_examples * 0.05:
                            raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(self.data)))
                else:
                    raise Exception("concept_collection {} is out of scope!".format(self.concept_collection))
            if "obj" in self.w_type and "mask" not in self.w_type:
                self.data = mask_to_obj(self.data)
            self.idx_list = list(range(len(self.data)))
            if len(self.idx_list) < n_examples:
                p.print("Dataset created with {} examples, less than {} specified.".format(len(self.idx_list), n_examples))
            else:
                p.print("Dataset for {} created.".format(mode))
        else:
            self.data = data
            self.idx_list = idx_list
            self.concept_collection = concept_collection
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __repr__(self):
        return "ConceptDataset({})".format(len(self))

    def __getitem__(self, idx):
        """Get data instance, where idx can be a number or a slice."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        elif isinstance(idx, slice):
            return self.__class__(
                mode=self.mode,
                canvas_size=self.canvas_size,
                n_examples=self.n_examples,
                rainbow_prob=self.rainbow_prob,
                data=self.data,
                idx_list=self.idx_list[idx],
                concept_collection=self.concept_collection,
                w_type=self.w_type,
                color_avail=self.color_avail,
                max_n_distractors=self.max_n_distractors,
                n_operators=self.n_operators,
                transform=self.transform,
            )
        sample = self.data[self.idx_list[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            sample = self[index]
            if len(sample) == 4:
                p.print("example {}, {}:".format(index, sample[2]))
                if isinstance(sample[0], tuple):
                    visualize_matrices([sample[0][0].argmax(0), sample[0][1].argmax(0)])
                else:
                    visualize_matrices([sample[0].argmax(0)])
                plot_matrices([sample[1][i].squeeze() for i in range(len(sample[1]))], images_per_row=6)






class ConceptCompositionDataset(Dataset):
    """Concept Composition dataset for learning to compose concepts from elementary concepts.

    mode:
        Concepts:  E(x; a; c)
            "Pixel": one or many pixels
            "Line": one or many lines
            "Rect": hollow rectangles
            "{}+{}+...": each "{}" can be a concept.

        Relations: E(x; a1, a2; c)
            "Vertical": lines where some of them are vertical
            "Parallel": lines where some of them are parallel
            "Vertical+Parallel": lines where some of them are vertical or parallel
            "IsInside": obj_1 is inside obj_2
            "SameRow": obj_1 and obj_2 are at the same row
            "SameCol": obj_1 and obj_2 are at the same column

        Operations: E(x1,x2; a1,a2; c1,c2)
            "RotateA+vFlip(Line+Rect)": two images where some object1 in image1 is rotated or vertically-flipped w.r.t. some object2 in image2, and the objects are chosen from Line or Rect.
            "hFlip(Lshape)", "vFlip(Lshape+Line)": two images where some object1 in image1 is flipped w.r.t. some object2 in image2.

        ARC+:
            "arc^{}": ARC images with property "{}" masked as above.
        ""
    """
    def __init__(
        self,
        canvas_size=8,
        n_examples=10000,
        concept_avail=None,
        relation_avail=None,
        additional_concepts=None,
        n_concepts_range=(2,3),
        relation_structure="None",
        rainbow_prob=0.,
        data=None,
        idx_list=None,
        color_avail="-1",
        min_n_distractors=0,
        max_n_distractors=0,
        n_examples_per_task=5,
        transform=None,
    ):
        self.canvas_size = canvas_size
        self.n_examples = n_examples
        self.concept_avail = concept_avail
        self.relation_avail = relation_avail
        self.additional_concepts = additional_concepts
        self.n_concepts_range = n_concepts_range
        self.relation_structure = relation_structure
        self.rainbow_prob = rainbow_prob
        self.min_n_distractors = min_n_distractors
        self.max_n_distractors = max_n_distractors
        self.n_examples_per_task = n_examples_per_task

        if isinstance(color_avail, str):
            if color_avail == "-1":
                self.color_avail = None
            else:
                self.color_avail = [int(c) for c in color_avail.split(",")]
                for c in self.color_avail:
                    assert c >= 1 and c <= 9
        else:
            self.color_avail = color_avail

        if idx_list is None:
            assert data is None
            dataset_engine = BabyARCDataset(
                pretrained_obj_cache=os.path.join(get_root_dir(), 'arc_objs.pt'),
                save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                object_limit=None,
                noise_level=0,
                canvas_size=canvas_size,
            )
            self.data = []
            for i in range(self.n_examples * 3):
                task = sample_selector(
                    dataset_engine=dataset_engine,
                    concept_avail=concept_avail,
                    relation_avail=relation_avail,
                    additional_concepts=self.additional_concepts,
                    n_concepts_range=n_concepts_range,
                    relation_structure=relation_structure,
                    min_n_distractors=min_n_distractors,
                    max_n_distractors=max_n_distractors,
                    canvas_size=canvas_size,
                    color_avail=self.color_avail,
                    n_examples_per_task=n_examples_per_task,
                    max_n_trials=5,
                    isplot=False,
                )
                if len(task) == n_examples_per_task:
                    self.data.append(task)
                if len(self.data) % 100 == 0:
                    p.print("Number of tasks generated: {}".format(len(self.data)))
                if len(self.data) >= self.n_examples:
                    break

            self.idx_list = list(range(len(self.data)))
            if len(self.idx_list) < n_examples:
                p.print("Dataset created with {} examples, less than {} specified.".format(len(self.idx_list), n_examples))
            else:
                p.print("Dataset created with {} examples.".format(len(self.idx_list)))
        else:
            self.data = data
            self.idx_list = idx_list
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __repr__(self):
        return "ConceptDataset({})".format(len(self))

    def __getitem__(self, idx):
        """Get data instance, where idx can be a number or a slice."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        elif isinstance(idx, slice):
            return self.__class__(
                canvas_size=self.canvas_size,
                n_examples=self.n_examples,
                concept_avail=self.concept_avail,
                relation_avail=self.relation_avail,
                additional_concepts=self.additional_concepts,
                n_concepts_range=self.n_concepts_range,
                relation_structure=self.relation_structure,
                rainbow_prob=self.rainbow_prob,
                data=self.data,
                idx_list=self.idx_list[idx],
                color_avail=self.color_avail,
                min_n_distractors=self.min_n_distractors,
                max_n_distractors=self.max_n_distractors,
                n_examples_per_task=self.n_examples_per_task,
                transform=self.transform,
            )

        sample = self.data[self.idx_list[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def to_dict(self):
        Dict = {}
        for id in self.idx_list:
            Dict[str(id)] = self.data[id]
        return Dict

    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            task = self[index]
            if len(task) == self.n_examples_per_task:
                info = task[0][3]
                p.print("structure: {}".format(info["structure"]))
                p.print("obj_spec_core:")
                pp.pprint(info["obj_spec_core"])
                for k, example in enumerate(task):
                    p.print("task {}, example {}:".format(index, k))
                    if isinstance(example[0], tuple):
                        visualize_matrices([example[0][0].argmax(0), example[0][1].argmax(0) if example[0][1].shape[0] == 10 else example[0][1].squeeze(0)])
                    else:
                        visualize_matrices([example[0].argmax(0)])


class NoiseBabyARC(Dataset):
    def __init__(self, name, size=(16,16), noise=False):
        dataset=torch.load('./datasets/files/'+name+'.pt') # call on root
        self.size=size
        self.data=[]
        self.info=[]
        self.noise=noise
        for i in dataset:
            for j in i:
                x,_,_,g=j
                self.data.append(x)
                self.info.append(dict(g))
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        h,w=self.size
        x=self.data[idx]
        if self.noise:
            s=(x+torch.randint(2,(10,h,w))).argmax(0)
            x_=torch.nn.functional.one_hot(s,10).permute(2,0,1)
        else: x_=x
        y=x#.argmax(0)
        return y, x_
    
    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable): idx = [idx]
        for index in idx:
            x, x_ = self[index]
            info=self.info[index]
            p.print("structure: {}".format(info["structure"]))
            p.print("obj_spec_core:")
            pp.pprint(info["obj_spec_core"])
            visualize_matrices([x,x_])


class ParseBabyARC(Dataset):
    def __init__(self, name):
        dataset=torch.load('./datasets/files/'+name+'.pt') # call on root
        self.data=[]
        self.objs=[]
        self.rels=[]
        rel_dict={ # leave 0 as None
            "Parallel":1,
            "VerticalMid":2, 
            "VerticalEdge":3, 
            "VerticalSepa":4
        }
        for i in dataset:
            x,_,_,g=i
            self.data.append(x[:3])
            obj=torch.stack([g['id_object_mask'][i] for i in g['id_object_mask']])
            self.objs.append(obj)
            self.rels.append([(r[0],r[1],rel_dict[r[2]]) for r in g['relations']])
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        x=self.data[idx]
        o=self.objs[idx]
        r=self.rels[idx]
        return x,o,r
    
    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable): idx = [idx]
        for index in idx:
            x, o, r = self[index]
            visualize_matrices([x.argmax(0)])
            toplot=[i.squeeze() for i in o]
            plot_matrices(toplot)

def collate_fn_lwp(list_items):
    x,o,r = [],[],[]
    for x_, o_, r_ in list_items:
        x.append(x_)
        o.append(o_)
        r.append(r_)
    return torch.stack(x), o, r


