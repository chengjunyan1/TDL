import torch
import matplotlib.pylab as plt
import scipy as sp
from torch.autograd import Variable
from numbers import Number
import scipy.ndimage
import numpy as np

# Baby-ARC related imports

def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list
    
def get_obj_from_mask(input, obj_mask=None):
    """Get the object from the mask."""
    if obj_mask is None:
        return input
    assert input.shape[-2:] == obj_mask.shape
    if isinstance(input, np.ndarray):
        input = torch.FloatTensor(input)
    if isinstance(obj_mask, np.ndarray):
        obj_mask = torch.BoolTensor(obj_mask.astype(bool))
    shape = input.shape
    if len(shape) == 3:
        output = torch.zeros_like(input).reshape(input.shape[0], -1)
        idx = obj_mask.flatten().bool()
        output[:, idx] = input.reshape(input.shape[0], -1)[:, idx]
    else:
        output = torch.zeros_like(input).flatten()
        idx = obj_mask.flatten().bool()
        output[idx] = input.flatten()[idx]
    return output.reshape(shape)

def shrink(input):
    """ Find the smallest region of your matrix that contains all the nonzero elements """
    if not isinstance(input, torch.Tensor):
        input = torch.FloatTensor(input)
        is_numpy = True
    else:
        is_numpy = False
    if input.abs().sum() == 0:
        return input, (0, 0, input.shape[-2], input.shape[-1])
    if len(input.shape) == 3:
        input_core = input.mean(0)
    else:
        input_core = input
    rows = torch.any(input_core.bool(), axis=-1)
    cols = torch.any(input_core.bool(), axis=-2)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    shrinked = input[..., ymin:ymax+1, xmin:xmax+1]
    pos = (ymin.item(), xmin.item(), shrinked.shape[-2], shrinked.shape[-1])
    if is_numpy:
        shrinked = to_np_array(shrinked)
    return shrinked, pos

def find_connected_components(input, is_diag=True, is_mask=False):
    """Find all the connected components, regardless of color."""
    input = to_np_array(input)
    shape = input.shape
    if is_diag:
        structure = [[1,1,1], [1,1,1], [1,1,1]]
    else:
        structure = [[0,1,0], [1,1,1], [0,1,0]]
    if len(shape) == 3:
        input_core = input.mean(0)
    else:
        input_core = input
    labeled, ncomponents = sp.ndimage.measurements.label(input_core, structure)

    objects = []
    for i in range(1, ncomponents + 1):
        obj_mask = (labeled == i).astype(int)
        obj = shrink(get_obj_from_mask(input, obj_mask))
        if is_mask:
            objects.append(obj + (obj_mask,))
        else:
            objects.append(obj)
    return objects

def find_connected_components_colordiff(input, is_diag=True, color=True):
    """Find all the connected components, considering color."""
    input = to_np_array(input)
    shape = input.shape

    if len(shape) == 3:
        assert shape[0] == 3
        color_list = np.unique(input.reshape(shape[0], -1), axis=-1).T
        bg_color = np.zeros(shape[0])
    else:
        input_core = input
        color_list = np.unique(input)
        bg_color = 0

    objects = []
    for c in color_list:
        if not (c == bg_color).all():
            if len(shape) == 3:
                mask = np.array(input!=c[:,None,None]).any(0, keepdims=True).repeat(shape[0], axis=0)
            else:
                mask = np.array(input!=c, dtype=int)
            color_mask = np.ma.masked_array(input, mask)
            color_mask = color_mask.filled(fill_value=0)
            objs = find_connected_components(color_mask, is_diag=is_diag)
            objects += objs
    return objects

    
# general util functions like plots, helpers to manipulate matrics
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_indices(tensor, pos=None, includes_neighbor=False, includes_self=True):
    """Get the indices of nonzero elements of an image.

    Args:
        tensor: 2D or 3D tensor. If 3D, it must have the shape of [C, H, W] where C is the number of channels.
        pos: position of the upper-left corner pixel of the tensor in the larger image. If None, will default as (0, 0).
        includes_neighbor: whether to include indices of neighbors (up, down, left, right).
        includes_self: if includes_neighbor is True, whether to include its own indices.

    Returns:
        indices: a list of indices satisfying the specification.
    """
    mask = tensor > 0
    if len(mask.shape) == 3:
        mask = mask.any(0)
    pos_add = (int(pos[0]), int(pos[1]))  if pos is not None else (0, 0)
    indices = []
    self_indices = []
    for i, j in torch.stack(torch.where(mask)).T:
        i, j = int(i) + pos_add[0], int(j) + pos_add[1]
        self_indices.append((i, j))
        if includes_neighbor:
            indices.append((i + 1, j))
            indices.append((i - 1, j))
            indices.append((i, j + 1))
            indices.append((i, j - 1))
    if includes_neighbor:
        if not includes_self:
            indices = list(set(indices).difference(set(self_indices)))
        else:
            indices = remove_duplicates(indices)
    else:
        indices = self_indices
    return indices

def plot_with_boundary(image, plt):
    im = plt.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal');
    height, width = np.array(image).shape[:2]
    ax = plt.gca();

    # Major ticks
    ax.set_xticks(np.arange(0, width, 1));
    ax.set_yticks(np.arange(0, height, 1));

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, width + 1, 1));
    ax.set_yticklabels(np.arange(1, height + 1, 1));

    # Minor ticks
    ax.set_xticks(np.arange(-.5, width, 1), minor=True);
    ax.set_yticks(np.arange(-.5, height, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    plt.xticks([])
    plt.yticks([])

def get_object_position_tags(obj_pos_t, root_shape_t):
    position_tags = []
    # using the first two bit of determine whether attach to left or up
    if obj_pos_t[0] == 0:
        position_tags.append("upper")
    if obj_pos_t[1] == 0:
        position_tags.append("left")
    if (obj_pos_t[0]+obj_pos_t[2]) == root_shape_t[0]:
        position_tags.append("lower")
    if (obj_pos_t[1]+obj_pos_t[3]) == root_shape_t[1]:
        position_tags.append("right")
    return position_tags

def randint_exclude(l, u, e):
    r = e[0]
    while r in e:
        r = random.randint(l, u)
    return r

def single_task_obj_parser(task_id):
    """
    return a list of objects contained in this task.
    """
    task_objs = []
    task_id += ".json"
    (inputs, _), (_, _) = load_task(task_id, isplot=False)
    # parse objects in 3 ways
    inputs_graph = parse_obj(inputs)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
            
    inputs_graph = parse_obj(inputs, is_colordiff=False)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
            
    inputs_graph = parse_obj(inputs, is_diag=False)
    for i in range(len(inputs_graph[0])):
        root_shape_t = inputs_graph[0][i].get_node_value("Image").shape
        for obj_n in inputs_graph[0][i].objs:
            obj_img_t = inputs_graph[0][i].get_node_value(obj_n)
            obj_pos_t = inputs_graph[0][i].get_node_value(obj_n.split(":")[0]+"^pos:Pos")
            obj_position_tags = get_object_position_tags(obj_pos_t, root_shape_t)
            obj_fmt = Object(obj_img_t, obj_position_tags)
            task_objs.append(obj_fmt)
    return task_objs

# relation parser as we need for multihop reasonings
def SameShape(image1, pos1, image2, pos2):
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1.bool() == image2.bool()).all()

def SameColor(image1, pos1, image2, pos2):
    color1 = -1 if len(image1.unique()) > 2 else image1.unique()[0]
    color2 = -1 if len(image2.unique()) > 2 else image2.unique()[0]
    if image1.unique()[0] == 0:
        color1 = image1.unique()[1]
    if image2.unique()[0] == 0:
        color2 = image2.unique()[1]
        
    if color1 == -1 or color2 == -1:
        return False
    else:
        return color1 == color2

def SameAll(image1, pos1, image2, pos2):
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1 == image2).all()

def SameRow(image1, pos1, image2, pos2):
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[0] == pos2[0] and pos1[2] == pos2[2]:
        return True
    else:
        return False

def SameCol(image1, pos1, image2, pos2):
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[1] == pos2[1] and pos1[3] == pos2[3]:
        return True
    else:
        return False

def IsInside(image1, pos1, image2, pos2):
    """Whether obj1 is inside obj2."""
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    if pos1[0] > pos2[0] and pos1[1] > pos2[1] and pos1[0] + pos1[2] < pos2[0] + pos2[2] and pos1[1] + pos1[3] < pos2[1] + pos2[3]:
        image2_patch = image2[int(pos1[0] - pos2[0]): int(pos1[0] + pos1[2] - pos2[0]), 
                              int(pos1[1] - pos2[1]): int(pos1[1] + pos1[3] - pos2[1])]
        overlap = (image1 != 0) & (image2_patch != 0)
        if overlap.any():
            return False
        else:
            return True
    else:
        return False
    
def IsNonOverlapXY(image1, pos1, image2, pos2):
    """Whether obj1 is inside obj2."""
    rec1 = (pos1[0], pos1[1], pos1[0]+image1.shape[0], pos1[1]+image1.shape[1])
    rec2 = (pos2[0], pos2[1], pos2[0]+image2.shape[0], pos2[1]+image2.shape[1])
    def intersect(p_left, p_right, q_left, q_right):
        return min(p_right, q_right) > max(p_left, q_left)
    if (intersect(rec1[0], rec1[2], rec2[0], rec2[2]) & \
        intersect(rec1[1], rec1[3], rec2[1], rec2[3])):
        return False
    return True

def IsTouch(image1, pos1, image2, pos2):
    """Whether the "obj"'s leftmost/rightmost/upmost/downmost part touches any other pixels (up, down, left, right) or boundary in the "image"."""
    pos1 = (pos1[0], pos1[1], image1.shape[0], image1.shape[1])
    pos2 = (pos2[0], pos2[1], image2.shape[0], image2.shape[1])
    obj_indices = get_indices(
        image1,
        pos1,
        includes_self=False,
        includes_neighbor=True,
    )
    obj2_indices = get_indices(image2, pos2)
    is_torch = len(set(obj_indices).intersection(set(obj2_indices))) > 0
    return is_torch