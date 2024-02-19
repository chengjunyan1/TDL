# Adopted from https://github.com/snap-stanford/zeroc 

import argparse
from collections import OrderedDict, Counter
from copy import copy, deepcopy
import itertools
import json
import matplotlib.pylab as plt
from multiset import Multiset
from numbers import Number
import numpy as np
import pdb
import pickle
import random
import scipy as sp
from scipy import ndimage
import sys, os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
 

REA_PATH = 'datasets'
EXP_PATH = 'results'


COLOR_DICT_3D = {
    0: [0, 0, 0],
    1: [0.2, 0.9, 0.2],
    2: [0.9, 0.1, 0.9],
    3: [0.3, 0.3, .9],
    4: [0.9, 0.2, 0.2],
    5: [.5, .55, .5],
    6: [.5, 0.1, .5],
    7: [1, .64, 0.3],
    8: [0.2, 0.9, 0.9],
    9: [1, 0.3, 1],
}

def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("zeroc")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def load_dataset(filename, directory=None, isTorch=True):
    if directory is None:
        directory = 'datasets/ARC/data/training'
    with open(os.path.join(get_root_dir(), directory, filename)) as json_file:
        dataset = json.load(json_file)
    if isTorch:
        dataset = to_Variable_recur(dataset, type="long")
    return dataset


color_dict = {0: [1, 1, 1],
              1: [0, 0, 1],
              2: [1, 0, 0],
              3: [0, 1, 0],
              4: [1, 1, 0],
              5: [.5, .5, .5],
              6: [.5, 0, .5],
              7: [1, .64, 0],
              8: [0, 1, 1],
              9: [.64, .16, .16],
              10: [1, 0, 1],
              11: [.5, .5, 0],
             }

COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]


def onehot_to_RGB(tensor, scale=1):
    """Transform 10-channel ARC image to 3-channel RGB image.

    Args:
        tensor: [B, C:10, H, W]

    Returns:
        tensor: [B, C:3, H, W]
    """
    tensor = torch.LongTensor(to_np_array(tensor)).argmax(1)  # [B, C:10, H, W] -> [B, H, W]
    collection = torch.FloatTensor(list(color_dict.values())) * scale  # [10, 3]
    return collection[tensor].permute(0,3,1,2)  # collection[tensor]: [B, H, W, C:3]; after permute: [B, C:3, H, W]


def to_one_hot(tensor, n_channels=10):
    """
    Args:
        tensor: [[B], H, W], where each values are integers in 0,1,2,... n_channels-1

    Returns:
        tensor: [[B], n_channels, H, W]
    """
    if isinstance(tensor, torch.Tensor):
        collection = torch.eye(n_channels)   # [n_channels, n_channels]
        if len(tensor.shape) == 2:  # [H, W]
            return collection[tensor.long()].permute(2,0,1)  # [C, H, W]
        elif len(tensor.shape) == 3: # [B, H, W]
            return collection[tensor.long()].permute(0,3,1,2)  # [B, C, H, W]
        else:
            raise Exception("tensor must be 2D or 3D!")
    elif isinstance(tensor, np.ndarray):
        collection = np.eye(n_channels)
        if len(tensor.shape) == 2:  # [H, W]
            return collection[tensor.astype(int)].transpose(2,0,1)
        elif len(tensor.shape) == 3:  # [H, W]
            return collection[tensor.astype(int)].transpose(0,3,1,2)
        else:
            raise Exception("tensor must be 2D or 3D!")
    else:
        raise Exception("tensor must be PyTorch or numpy tensor!")


def visualize_masks(imgs, masks, recons, vis):
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    if imgs.shape[1] != 3 and recons.shape[1] != 3:
        imgs = onehot_to_RGB(imgs, scale=1)
        recons = onehot_to_RGB(recons, scale=1)

    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


def to_tuple(tensor):
    """Transform a PyTorch tensor into a tuple of ints."""
    assert len(tensor.shape) == 1
    return tuple([int(item) for item in tensor])


def get_root_dir():
    dirname = os.getcwd()
    # dirname_split = dirname.split("/")
    # index = dirname_split.index("TAE")
    # dirname = "/".join(dirname_split[:index + 1])
    return dirname


def get_filenames(path, include=""):
    if not isinstance(include, list):
        include = [include]
    with os.scandir(path) as it:
        filenames = []
        for entry in it:
            if entry.is_file():
                is_in = True
                for element in include:
                    if element not in entry.name:
                        is_in = False
                        break
                if is_in:
                    filenames.append(entry.name)
        return filenames
    
def get_directory_tasks(directories, split_json=False, include_dir=False):
    file_list = []
    for directory in directories:
        files = get_filenames(os.path.join(get_root_dir(), directory),  include=".json")
        file_list += [(directory, file.split('.json')[0] if split_json else file) for file in files]
    if not include_dir:
        file_list = [file for _, file in file_list]
        # Important: Sort the list of files to make sure the order of files is consistent
        file_list.sort()
    return file_list

def sample_dataset(directories, task_list=None):
    """Sample one dataset from the given directories.
    """
    file_list = []
    for directory in directories:
        files = get_filenames(os.path.join(get_root_dir(), directory), include=".json")
        file_list += [(directory, file) for file in files]
    # Check if the file is in task_list:
    if task_list is not None:
        file_list_union = []
        for directory, file in file_list:
            if file.split(".json")[0] in task_list:
                file_list_union.append((directory, file))
    else:
        file_list_union = file_list
    assert len(file_list_union) > 0, "Did not find task {} in {}.".format(task_list, directories)
    id_chosen = np.random.choice(len(file_list_union))
    directory_chosen, file_chosen = file_list_union[id_chosen]
    dataset = load_dataset(file_chosen, directory=directory_chosen)
    return dataset, file_chosen


def get_task_list(task_file):
    """Obtain task_list from task_file."""
    task_list = []
    with open(os.path.join(get_root_dir(), task_file), "r") as f:
        for line in f.readlines():
            if len(line) > 1:
                line_core = line.split("#")[0].strip()
                if len(line_core) > 1:
                    task_list.append(line.split("#")[0].strip())
    return task_list


def get_inputs_targets(dataset):
    """Get inputs (list of OrderedDict) and targets (OrderedDict) from train or test dataset."""
    inputs = OrderedDict()
    targets = OrderedDict()
    for i in range(len(dataset)):
        input, target = dataset[i]["input"], dataset[i]["output"]
        inputs[i] = input
        targets[i] = target
    inputs = [inputs]
    return inputs, targets


def get_inputs_targets_EBM(dataset):
    """Get inputs (list of OrderedDict) and targets (OrderedDict) from ConceptCompositionDataset."""
    inputs = OrderedDict()
    targets = OrderedDict()
    infos = OrderedDict()
    for i in range(len(dataset)):
        if isinstance(dataset[i][0], tuple) or isinstance(dataset[i][0], list):
            assert len(dataset[i][0]) == 2
            input = dataset[i][0][0]
            target = dataset[i][0][1]
        else:
            input = target = dataset[i][0]
        inputs[i] = input
        targets[i] = target
        infos[i] = dataset[i][3]
    inputs = [inputs]
    return inputs, targets, infos


def plot_with_boundary(image, plt):
    im = plt.imshow(image, interpolation='none', vmin=0, vmax=1, aspect='equal');
#     height, width = np.array(image).shape[:2]
#     ax = plt.gca();

#     # Major ticks
#     ax.set_xticks(np.arange(0, width, 1));
#     ax.set_yticks(np.arange(0, height, 1));

#     # Labels for major ticks
#     ax.set_xticklabels(np.arange(1, width + 1, 1));
#     ax.set_yticklabels(np.arange(1, height + 1, 1));

#     # Minor ticks
#     ax.set_xticks(np.arange(-.5, width, 1), minor=True);
#     ax.set_yticks(np.arange(-.5, height, 1), minor=True);

#     # Gridlines based on minor ticks
#     ax.grid(which='minor', color='w', linestyle='-', linewidth=2)


def visualize_matrices(matrices, num_rows=None, row=0, images_per_row=None, plt=None, is_show=True, filename=None, title=None, subtitles=None, use_color_dict=True, masks=None, is_round_mask=False, **kwargs):
    """
    :param matrices: if use_color_dict is False, shape [N, 3, H, W] where each value is in [0, 1]
                     otherwise, shape [N, H, W] where each value is a number from 0-10
                     if masks is not None, shape [N, 1, H, W] where each value is in [0, 1]
    """
    if images_per_row is None:
        images_per_row = min(6, len(matrices))
    num_plots = len(matrices)
    if num_rows is None:
        num_rows = int(np.ceil(num_plots / images_per_row))
    if plt is None:
        import matplotlib.pylab as plt
        plt_show = True
        plt.figure(figsize=(3*images_per_row, 3*num_rows))
    else:
        plt_show = False

    for i, matrix in enumerate(matrices):
        matrix = to_np_array(matrix, full_reduce=False)
        if isinstance(matrix, bool):
            continue
        if not use_color_dict:
            plt.subplot(num_rows, images_per_row, i + 1 + 2 * row)
            image = np.zeros((matrix.shape[-2], matrix.shape[-1], 3))
            for k in range(matrix.shape[-2]):
                for l in range(matrix.shape[-1]):
                    image[k, l] = np.array(matrix[:, k, l])
        else:
            if matrix.dtype.name.startswith("float"):
                # Typically for masks, we want to round them such that a 0.5+ value
                # is considered as being "part of" the mask.
                matrix = np.round(matrix).astype("int")
            if matrix.dtype.name.startswith("int") or matrix.dtype.name == "bool":
                plt.subplot(num_rows, images_per_row, i + 1 + 2 * row)
                image = np.zeros((*matrix.shape, 3))
                for k in range(matrix.shape[0]):
                    for l in range(matrix.shape[1]):
                        image[k, l] = np.array(color_dict[matrix[k, l]])

        if masks is not None:
            mask = to_np_array(masks[i], full_reduce=False)
            for k in range(mask.shape[-2]):
                for l in range(mask.shape[-1]):
                    image[k, l] *= (np.around(mask[:, k, l]) if is_round_mask else mask[:, k, l])
        plot_with_boundary(image, plt)
        if subtitles is not None:
            plt.title(subtitles[i])
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelsize=0,
            labelbottom=False)
        # plt.axis('off')
    ax = plt.gca()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    if title is not None:
        plt.suptitle(title, fontsize=14, y=0.9)
    if filename is not None:
        ax = plt.gca()
        ax.set_rasterized(True)
        plt.savefig(filename, bbox_inches="tight", dpi=400, **kwargs)
    if is_show and plt_show:
        plt.show()


def visualize_dataset(dataset, filename=None, is_show=True, title=None, **kwargs):
    def to_value(input):
        if not isinstance(input, torch.Tensor):
            input = input.get_node_value()
        return input
    length = len(dataset["train"]) + 1
    plt.figure(figsize=(7, 3.5 * (length)))
    for i, data in enumerate(dataset["train"]):
        visualize_matrices([to_value(data["input"]), to_value(data["output"])], images_per_row=2, num_rows=length, row=i, plt=plt, is_show=is_show, title=title if i == 0 else None)
    if "test" in dataset:
        visualize_matrices([to_value(dataset["test"][0]["input"])], images_per_row=2, num_rows=length, row=i+1, plt=plt, is_show=is_show)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", **kwargs)
    if is_show:
        plt.show()
        
        
# def plot_matrices(matrices):
#     if not isinstance(matrices, list):
#         matrices = [matrices]
#     if len(matrices[0].shape) == 2:
#         visualize_matrices(matrices)
#     elif len(matrices[0].shape) == 3:
#         length = len(matrices)
#         plt.figure(figsize=(3.5*length, 4))
#         for i, matrix in enumerate(matrices):
#             vmax = 255 if matrix.max() > 1 else 1
#             plt.subplot(1, length, 1 + i)
#             plt.imshow(to_np_array(matrix).transpose(1, 2, 0).astype(int), vmax=vmax)
#             plt.axis('off')
#         plt.show()

def plot_matrices(
    matrix_list, 
    shape = None, 
    images_per_row = 10, 
    scale_limit = None,
    figsize = None,
    x_axis_list = None,
    filename = None,
    title = None,
    subtitles = [],
    highlight_bad_values = True,
    plt = None,
    pdf = None,
    verbose = False,
    no_xlabel = False,
    cmap = None,
    is_balanced = False,
    ):
    """Plot the images for each matrix in the matrix_list.
    Adapted from https://github.com/tailintalent/pytorch_net/blob/c1cfda5e90fef9503c887f5061cb7b1262133ac0/util.py#L54
    """
    import matplotlib
    from matplotlib import pyplot as plt
    n_rows = max(len(matrix_list) // images_per_row, 1)
    fig = plt.figure(figsize=(20, n_rows*7) if figsize is None else figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize = 18, horizontalalignment = 'left', x=0.1)
    
    # To np array. If None, will transform to NaN:
    matrix_list_new = []
    for i, element in enumerate(matrix_list):
        if element is not None:
            matrix_list_new.append(to_np_array(element))
        else:
            matrix_list_new.append(np.array([[np.NaN]]))
    matrix_list = matrix_list_new
    
    num_matrixs = len(matrix_list)
    rows = int(np.ceil(num_matrixs / float(images_per_row)))
    try:
        matrix_list_reshaped = np.reshape(np.array(matrix_list), (-1, shape[0],shape[1])) \
            if shape is not None else np.array(matrix_list)
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
        if is_balanced:
            scale_min, scale_max = -max(abs(scale_min), abs(scale_max)), max(abs(scale_min), abs(scale_max))
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = copy(plt.cm.get_cmap("binary" if cmap is None else cmap))
            cmap.set_bad('red', alpha = 0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis = 0)
            image = np.ma.array(image, mask = mask_key)
        else:
            cmap = matplotlib.cm.binary if cmap is None else cmap
        if scale_limit is None:
            ax.matshow(image, cmap = cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            if is_balanced:
                scale_min, scale_max = scale_limit
                scale_limit = -max(abs(scale_min), abs(scale_max)), max(abs(scale_min), abs(scale_max)) 
            ax.matshow(image, cmap = cmap, vmin = scale_limit[0], vmax = scale_limit[1])
        if len(subtitles) > 0:
            ax.set_title(subtitles[i])
        if not no_xlabel:
            try:
                xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(np.min(image), np.max(image), image.shape[0], image.shape[1])
                if x_axis_list is not None:
                    xlabel += "\n{}".format(x_axis_list[i])
                plt.xlabel(xlabel)
            except:
                pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    # if cmap is not None:
    #     cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    #     plt.colorbar(cax=cax)
        # cbar_ax = fig.add_axes([0.92, 0.3, 0.01, 0.4])
        # plt.colorbar(cax=cbar_ax)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=400)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        if verbose:
            print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


def get_op_shape(result):
    """Get the shape of the result on an operator."""
    device = result[list(result.keys())[0]].device
    op_shape_dict = OrderedDict()
    for key, item in result.items():
        op_shape = torch.zeros(2).long().to(device)
        if not (isinstance(item, torch.Tensor) or isinstance(item, np.ndarray)):
            item = item.get_node_value()
        if hasattr(item, "shape"):
            shape = item.shape
            if len(shape) == 2:
                op_shape = torch.LongTensor(tuple(item.shape)).to(device)
            elif len(shape) == 1:
                op_shape[0] = torch.LongTensor(tuple(item.shape))
        op_shape_dict[key] = op_shape
    return op_shape_dict


def combine_pos(*pos_list):
    """Obtain a minimum bounding box in the form of 'pos' to the list of input pos."""
    pos_list = np.array([to_np_array(pos) for pos in pos_list])
    pos_list[:, 2] += pos_list[:, 0]
    pos_list[:, 3] += pos_list[:, 1]

    new_pos_min = pos_list[:, :2].min(0)
    new_pos_max = pos_list[:, 2:].max(0)
    new_pos = tuple(np.concatenate([new_pos_min, new_pos_max - new_pos_min]).round().astype(int).tolist())
    return new_pos


def to_Graph(dataset, base_concept):
    dataset_Graph = deepcopy(dataset)
    for mode in ["train", "test"]:
        for i in range(len(dataset[mode])):
            for key in ["input", "output"]:
                dataset_Graph[mode][i][key] = base_concept.copy().set_node_value(dataset[mode][i][key])
                if base_concept.name == "Image":
                    dataset_Graph[mode][i][key].set_node_value([0, 0, dataset[mode][i][key].shape[0], dataset[mode][i][key].shape[1]], "pos")
    return dataset_Graph


def masked_equal(input1, input2, exclude=0):
    """Return True for each elements only if the corresponding elements are equal and nonzero."""
    if exclude is None:
        return input1 == input2
    else:
        nonzero_mask = (input1 != exclude) | (input2 != exclude)
        return (input1 == input2) & nonzero_mask


def get_input_output_mode_dict(operators, is_inherit=True):
    """Get the dictionary of input (output) modes mapped to the body operator."""
    concepts = combine_dicts(operators)
    input_mode_dict = {}
    output_mode_dict = {}
    for mode, operator in operators.items():
        if operator.__class__.__name__ == "Graph":
            # Input dict:
            input_modes = tuple(sorted([input_node.split(":")[-1] for input_node in operator.input_nodes.keys()]))
            record_data(input_mode_dict, [mode], [input_modes])

            # Output dict:
            output_mode = operator.get_output_nodes(types=["fun-out"], allow_goal_node=True)[0].split(":")[-1]
            record_data(output_mode_dict, mode, output_mode, ignore_duplicate=True)
    return input_mode_dict, output_mode_dict


def get_inherit_modes(mode, concepts, type=None):
    """Get all the modes that is inherit from (or to) the current mode."""
    mode = split_string(mode)[0]
    modes_inherit = [mode]
    if type == "from":
        if hasattr(concepts[mode], "inherit_from"):
            modes_inherit += concepts[mode].inherit_from
    elif type == "to":
        if hasattr(concepts[mode], "inherit_from"):
            modes_inherit += concepts[mode].inherit_from
    else:
        raise
    return modes_inherit


def combine_dicts(dicts):
    """Combine multiple concepts into a single one."""
    if isinstance(dicts, list):
        dicts_cumu = {}
        for dicts_ele in dicts:
            dicts_cumu.update(dicts_ele)
        dicts = dicts_cumu
    return dicts


def accepts(operator_input_modes, input_modes, concepts, mode="exists"):
    """Check if the operator (specified by operator_input_modes) accepts input_modes."""
    # If concepts is a list of concepts dictionaries, accumulate them:
    concepts = combine_dicts(concepts)

    if mode == "fully-cover":
        """True only if the operator's input can fully cover the input_modes"""
        if len(operator_input_modes) < len(input_modes):
            return False
        # Get input_modes_all:
        input_modes_all = []
        for mode in input_modes:
            mode_inherit = get_inherit_modes(mode, concepts, type="from")
            input_modes_all.append(mode_inherit)
        is_accept = False
        for comb_ids in itertools.combinations(range(len(operator_input_modes)), len(input_modes)):
            selected_operator_modes = [operator_input_modes[id] for id in comb_ids]
            is_accept_selected = False
            for keys in itertools.product(*input_modes_all):
                if Multiset(keys).issubset(Multiset(selected_operator_modes)):
                    is_accept_selected = True
                    break
            if is_accept_selected:
                is_accept = True
                break
    elif mode == "exists":
        """is_accept is True as long as there is an input_mode that can fed to the operator."""
        is_accept = False
        for input_mode in input_modes:
            mode_inherit = get_inherit_modes(input_mode, concepts, type="from")
            is_accept_selected = False
            for mode in mode_inherit:
                if mode in operator_input_modes:
                    is_accept_selected = True
                    break
            if is_accept_selected:
                is_accept = True
                break
    else:
        raise
    return is_accept


def find_valid_operators(nodes, operators, concepts, input_mode_dict, arity=2, exclude=None):
    """Given out_nodes, find operators that is compatible to {arity} number of them."""
    assert arity <= len(nodes)
    concepts = combine_dicts(concepts)
    if exclude is None:
        exclude = []
    modes = [node.split(":")[-1] for node in nodes]
    modes_all = []
    valid_options = {}
    for mode in modes:
        modes_inherit = get_inherit_modes(mode, concepts, type="from")
        modes_all.append(modes_inherit)

    for comb_ids in itertools.combinations(range(len(modes)), arity):
        selected_modes = [modes_all[id] for id in comb_ids]
        for keys in itertools.product(*selected_modes):
            for operator_input_modes in input_mode_dict:
                if Multiset(keys).issubset(Multiset(operator_input_modes)):
                    for operator_name in input_mode_dict[operator_input_modes]:
                        if operator_name in valid_options or operator_name in exclude:
                            continue
                        input_nodes = list(operators[operator_name].input_nodes.keys())
                        input_modes = [input_node.split(":")[-1] for input_node in input_nodes]
                        assign_list = []
                        chosen = []
                        """
                        selected_modes: [['Line', 'Image'], ['Line', 'Image']]  # The original mode and all the modes it inherit from, considering n choose k
                        chosen_modes: ["Line", "Line"]                          # The original mode considering n choose k
                        keys: ["Image", "Line"]                                 # The specific combination selected that has instance in INPUT_MODE_DICT
                        input_nodes: ['concept1:Line', 'concept1:Image']        # The input nodes for the specific operator
                        input_modes: ["Line", "Image"]                          # The input modes for the specific operator
                        """
                        chosen_nodes = [nodes[id] for id in comb_ids]
                        for j, node in enumerate(chosen_nodes):
                            chosen_key = keys[j]
                            # Assign input_mode to the chosen_modes:
                            for k, input_mode in enumerate(input_modes):
                                if input_mode == chosen_key and k not in chosen:
                                    assign_list.append([node, input_nodes[k]])
                                    break
                            chosen.append(k)

                        valid_options[operator_name] = assign_list
    return valid_options


def canonical(mode):
    """Return canonical mode."""
    return mode.split("*")[-1]


def view_values(Dict):
    """View the value in each item of Dict."""
    new_dict = OrderedDict()
    for key, item in Dict.items():
        if isinstance(item, Concept):
            item = item.get_root_value()
        new_dict[key] = deepcopy(item)
    return new_dict


def get_last_output(results, is_Graph=True):
    """Obtain the last output as a node in results"""
    output = results[list(results.keys())[-1]]
    if is_Graph and len(output.shape) == 2:
        output = concepts["Image"].copy().set_node_value(output)
    return output


def canonicalize_keys(Dict):
    """Make sure the keys of Dict have the same length. If not, make it so."""
    keys_core = [key if isinstance(key, tuple) else (key,) for key in Dict]
    length_list = [len(key_core) for key_core in keys_core]

    assert len(np.unique(length_list)) <= 2
    if len(np.unique(length_list)) <= 1:
        return Dict
    else:
        idx_max = max(np.unique(length_list))
        idx_min = min(np.unique(length_list))
        assert idx_max - idx_min == 1
        str_list = []
        for key_core in keys_core:
            if len(key_core) == idx_max:
                str_list.append(key_core[-1].split("-")[0])
        assert len(np.unique(str_list)) == 1
        string = "{}-{}".format(str_list[0], 0)
        key_map = {}
        for key, key_core in zip(list(Dict.keys()), keys_core):
            if len(key_core) == idx_min:
                new_key = key_core + (string,)
                key_map[key] = new_key

        for key, new_key in key_map.items():
            Dict[new_key] = Dict.pop(key)
    return Dict


def get_first_key(key):
    if isinstance(key, tuple):
        return key[0]
    else:
        assert isinstance(key, Number)
        return key


def get_Dict_with_first_key(Dict):
    if isinstance(Dict, OrderedDict):
        return OrderedDict([[get_first_key(key), item] for key, item in Dict.items()])
    else:
        return {get_first_key(key): item for key, item in Dict.items()}


def broadcast_inputs(inputs):
    """Broadcast inputs."""
    is_dict = False
    for input_arg in inputs:
        if isinstance(input_arg, dict):
            is_dict = True
            break
    if not is_dict:
        inputs = [{0: input_arg} for input_arg in inputs]

    # Broadcast input keys, and record inputs into results:
    input_key_list_all = []
    for i, input_arg in enumerate(inputs):
        if isinstance(input_arg, dict):
            input_key_list_all.append(input_arg.keys())
        else:
            input_key_list_all.append(None)
    input_key_dict = broadcast_keys(input_key_list_all)
    input_keys = list(input_key_dict.keys())
    return inputs, input_keys


def get_attr_proper_name(concept, node_name):
    length = node_name.find(concept.name.lower())
    if length == -1:
        return "^".join(node_name.split("^")[1:])
    else:
        return "^".join(node_name[length:].split("^")[1:])


def get_run_status():
    """Get the runing status saved in /web/static/states/run_status.txt, with 0 meaning to stop, and other string meaning the method being run."""
    run_status = "0"
    status_path = get_root_dir() + "/web/static/states/run_status.txt"
    make_dir(status_path)
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            f.write("0")
    with open(status_path, "r") as f:
        run_status = f.readline()
    return run_status


def set_run_status(status):
    status_path = get_root_dir() + "/web/static/states/run_status.txt"
    make_dir(status_path)
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            f.write("")
    with open(status_path, "w") as f:
        f.write("{}".format(status))


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.exp()


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.square()


def get_activation(activation, inplace=False):
    if activation.lower() == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    elif activation.lower() == "leakyrelu0.2":
        return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
    elif activation.lower() == "swish":
        return nn.SiLU(inplace=inplace)
    elif activation.lower() == "tanh":
        return nn.Tanh(inplace=inplace)
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid(inplace=inplace)
    elif activation.lower() == "linear":
        return nn.Identity()
    elif activation.lower() == "elu":
        return nn.ELU(inplace=inplace)
    elif activation.lower() == "softplus":
        return nn.Softplus()
    elif activation.lower() == "rational":
        return Rational()
    elif activation.lower() == "exp":
        return Exp()
    elif activation.lower() == "square":
        return Square()
    else:
        raise


def get_normalization(normalization_type, in_channels=None):
    if normalization_type.lower() == "none":
        return nn.Identity()
    elif normalization_type.lower().startswith("gn"):
        n_groups = eval(normalization_type.split("-")[1])
        return nn.GroupNorm(n_groups, in_channels, affine=True)
    elif normalization_type.lower() == "in":
        return nn.InstanceNorm2d(in_channels, affine=True)
    else:
        raise


class Rational(torch.nn.Module):
    """Rational Activation function.
    Implementation provided by Mario Casado (https://github.com/Lezcano)
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                         [1.5957, 2.383],
                                         [0.5, 0.0],
                                         [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        n_neurons,
        n_layers,
        activation="relu",
        output_size=None,
        last_layer_linear=True,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation = activation
        if n_layers > 1 or last_layer_linear is False:
            self.act_fun = get_activation(activation)
        self.output_size = output_size if output_size is not None else n_neurons
        self.last_layer_linear = last_layer_linear
        for i in range(1, self.n_layers + 1):
            setattr(self, "layer_{}".format(i), nn.Linear(
                self.input_size if i == 1 else self.n_neurons,
                self.output_size if i == self.n_layers else self.n_neurons,
            ))
            torch.nn.init.xavier_normal_(getattr(self, "layer_{}".format(i)).weight)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = getattr(self, "layer_{}".format(i))(x)
            if i != self.n_layers or not self.last_layer_linear:
                x = self.act_fun(x)
        return x


def to_device_recur(iterable, device, is_detach=False):
    if isinstance(iterable, list):
        return [to_device_recur(item, device, is_detach=is_detach) for item in iterable]
    elif isinstance(iterable, tuple):
        return tuple(to_device_recur(item, device, is_detach=is_detach) for item in iterable)
    elif isinstance(iterable, dict):
        return {key: to_device_recur(item, device, is_detach=is_detach) for key, item in iterable.items()}
    elif hasattr(iterable, "to"):
        iterable = iterable.to(device)
        if is_detach:
            iterable = iterable.detach()
        return iterable
    else:
        if hasattr(iterable, "detach"):
            iterable = iterable.detach()
        return iterable


def select_item(list_of_lists, indices=None):
    """Choose items from list of lists using indices, assuming that each element list has the same length."""
    flattened_list = []
    if indices is None:
        for List in list_of_lists:
            flattened_list += List
    else:
        for index in indices:
            q, r = divmod(index, len(list_of_lists[0]))
            flattened_list.append(list_of_lists[q][r])
    return flattened_list


def action_equal(action, num, allowed_modes):
    """Return True if action == num and action is in allowed modes.
    allowed_modes: string, e.g. "012".
    """
    action_int = int(action)
    return str(action_int) in allowed_modes and action_int == num


################################
# For parsing Rect and Lines:
################################

def get_empty_neighbor(matrix, i, j):
    """Calculate the number of empty neighbors of the current pixel.
    If the pixel has value, return which direction of the pixel has value
    If the pixel is 0, return 0"""
    m = len(matrix)
    n = len(matrix[0])
    empty_neighbor = 0
    if matrix[i][j]:
        up, down, left, right = True, True, True, True
        if i == 0 or matrix[i - 1][j] == 0:
            empty_neighbor += 1
            up = False
        if i == m - 1 or matrix[i + 1][j] == 0:
            empty_neighbor += 1
            down = False
        if j == 0 or matrix[i][j - 1] == 0:
            empty_neighbor += 1
            left = False
        if j == n - 1 or matrix[i][j + 1] == 0:
            empty_neighbor += 1
            right = False
        return empty_neighbor, up, down, left, right

    # Return 0 if the pixel has no value
    return 0, False, False, False, False


def add_dict(original_list, new_ele):
    """Add a new element to the dictionary according to its shape. 
    Return the updated list."""
    if new_ele[2] == 1 and new_ele[3] == 1:
        key = 'Pixel'
    elif new_ele[2] == 1 or new_ele[3] == 1:
        key = 'Line'
    else:
        key = 'RectSolid'
    if key not in original_list:
        original_list[key] = []
    original_list[key].append(new_ele)
    return original_list


def eliminate_rectangle(matrix, pos):
    """Eliminate an rectangle from a matrix.
    Return the updated matrix."""
    new_matrix = np.zeros_like(matrix)
    for i in range(pos[2]):
        for j in range(pos[3]):
            new_matrix[pos[0] + i][pos[1] + j] = 1
    return matrix - new_matrix


def maximal_rectangle(matrix, result_list, is_rectangle):
    """Find the rectangle, line, or pixel with the maximal area using dynamic programming.
    Add the maximal rectangle to the dictionary.
    Return the matrix that eliminate this rectangle, the updated dictionary, and whether an object is found."""
    m = len(matrix)
    n = len(matrix[0])

    left = [0] * n
    right = [n] * n
    height = [0] * n

    maxarea = 0
    result = (0, 0, 0, 0)

    for i in range(m):
        cur_left, cur_right = 0, n
        # update height
        for j in range(n):
            if matrix[i][j] == 0:
                height[j] = 0
            else:
                height[j] += 1
        # update left
        for j in range(n):
            if matrix[i][j] == 0:
                left[j] = 0
                cur_left = j + 1
            else:
                left[j] = max(left[j], cur_left)
        # update right
        for j in range(n-1, -1, -1):
            if matrix[i][j] == 0:
                right[j] = n
                cur_right = j
            else:
                right[j] = min(right[j], cur_right)
        # update the area
        for j in range(n):
            if is_rectangle and (height[j] < 2 or (right[j] - left[j]) < 2):
                continue
            tmp = height[j] * (right[j] - left[j])
            if tmp > maxarea:
                maxarea = tmp
                result = (i - height[j] + 1, left[j], height[j], right[j] - left[j])
    # define a matrix with only the max rectangle region has value
    new_matrix = matrix.copy()
    is_found = False
    if result[2] and result[3]:
        new_matrix = eliminate_rectangle(matrix, result)
        result_list = add_dict(result_list, result)
        is_found = True
    return new_matrix, result_list, is_found


def seperate_concept(matrix):
    """Seperate the rectangles, lines, and pixels in a matrix while prioritizing rectangles.
    Return a dictionary of infomations of rectangles, lines, and pixels."""
    matrix = to_np_array(matrix)
    new_matrix = matrix.copy()
    m = len(matrix)
    n = len(matrix[0])
    result = {}
    # Find the maximal rectangle until no rectangles are left in the matrix.
    is_found = True
    while is_found:
        new_matrix, result, is_found = maximal_rectangle(new_matrix, result, True)
    # Find all other lines and pixels left in the matrix.
    is_found = True
    while is_found:
        new_matrix, result, is_found = maximal_rectangle(new_matrix, result, False)
    # Make sure that there are no duplicates:
    for key in ["Pixel", "Line", "RectSolid"]:
        if key in result:
            result[key] = list(set(result[key]))
    return result


################################


def compose_dir_task_source(T_item, show_warning=False):
    if isinstance(T_item, list):
        directories = ["ARC/data/training", "ARC/data/evaluation"]
        task_source = T_item
    elif isinstance(T_item, set):
        directories = list(T_item)
        task_source = None
    elif isinstance(T_item, str):
        directories = ["ARC/data/training", "ARC/data/evaluation"]
        task_source = T_item
    else:
        raise
    # Verify the task directory hash to make sure no tasks have been 
    # added or removed
    for directory in directories:
        file_list = get_directory_tasks([directory])
        hash_str = get_hashing(str(file_list))
        directory = directory + '/' if directory[-1] != '/' else directory
        filename = directory + 'hash.txt'
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                prev_hash = f.readlines()[0].strip()
                assert prev_hash == hash_str, "Hash string doesn't match with saved hash!"
        else:
            try:
                with open(filename, 'w') as f:
                    print(hash_str, file=f)
            except:
                if show_warning:
                    print('Warning: Could not write hash to directory: {}'.format(directory))
                pass
    return directories, task_source


def get_patch(tensor, pos):
    """Get a patch of the tensor based on pos."""
    return tensor[..., int(pos[0]): int(pos[0] + pos[2]), int(pos[1]): int(pos[1] + pos[3])]


def set_patch(tensor, patch, pos, value=None):
    """If value is None, set the certain parts of the tensor with the **non-background** part of the patch.
    Otherwise, set the certain parts of the tensor at the value given according to
    the position of the patch and the **non-background** part.
    """
    pos = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
    shape = tensor.shape
    pos_0_offset = - pos[0] if pos[0] < 0 else 0
    pos_02_offset = shape[-2] - (pos[0] + pos[2]) if pos[0] + pos[2] > shape[-2] else 0
    pos_1_offset = - pos[1] if pos[1] < 0 else 0
    pos_13_offset = shape[-1] - (pos[1] + pos[3]) if pos[1] + pos[3] > shape[-1] else 0

    patch_core = patch[..., pos_0_offset: pos[2] + pos_02_offset, pos_1_offset: pos[3] + pos_13_offset]
    patch_core_g0 = patch_core > 0
    if len(patch_core_g0.shape) == 3:
        patch_core_g0 = patch_core_g0.any(0)
    
    tensor_set = tensor[..., pos[0] + pos_0_offset: pos[0] + pos[2] + pos_02_offset,
                            pos[1] + pos_1_offset: pos[1] + pos[3] + pos_13_offset]
    if patch_core.nelement() != 0 and patch_core[..., patch_core_g0].nelement() != 0 and \
        tensor_set.nelement() != 0:
        if value is None:
            tensor_set[..., patch_core_g0] = patch_core[..., patch_core_g0].type(tensor.dtype)
        else:
            value = value.type(tensor.dtype) if isinstance(value, torch.Tensor) else value
            tensor_set[..., patch_core_g0] = value
    return tensor


def classify_concept(tensor):
    """
    Args:
        Tensor: shape [H, W] and must have one of concept type of "Line", "RectSolid", "Rect", "Lshape".
    
    Returns:
        tensor_type: returns one of "Line", "RectSolid", "Rect", "Lshape".
    """
    assert len(tensor.shape) == 2
    tensor = shrink(tensor)[0]
    shape = tensor.shape
    assert set(np.unique(to_np_array(tensor.flatten())).tolist()).issubset({0., 1.})

    if shape[0] == 1 or shape[1] == 1:
        assert tensor.all()
        tensor_type = "Line"
    elif tensor.all():
        tensor_type = "RectSolid"
    elif tensor[0,0] == 1 and tensor[-1,0] == 1 and tensor[0,-1] == 1 and tensor[-1,-1] == 1:
        tensor_revert = 1 - tensor
        tensor_revert_shrink = shrink(tensor_revert)[0]
        assert tensor_revert_shrink.shape[0] == shape[0] - 2
        assert tensor_revert_shrink.shape[1] == shape[1] - 2
        tensor_type = "Rect"
    elif (tensor[0,0] == 1).long() + (tensor[-1,0] == 1).long() + (tensor[0,-1] == 1).long() + (tensor[-1,-1] == 1).long() == 3:
        tensor_type = "Lshape"
    else:
        assert (tensor == 0).all()
        tensor_type = None
    return tensor_type


def get_pos_intersection(pos1, pos2):
    """Get intersection of the position."""
    pos = [max(pos1[0], pos2[0]),
           max(pos1[1], pos2[1]),
           min(pos1[0] + pos1[2], pos2[0] + pos2[2]) - max(pos1[0], pos2[0]),
           min(pos1[1] + pos1[3], pos2[1] + pos2[3]) - max(pos1[1], pos2[1]),
          ]
    if pos[2] > 0 and pos[3] > 0:
        return pos
    else:
        return None


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
    labeled, ncomponents = ndimage.measurements.label(input_core, structure)

    objects = []
    for i in range(1, ncomponents + 1):
        obj_mask = (labeled == i).astype(int)
        obj = shrink(get_obj_from_mask(input, obj_mask))
        if is_mask:
            objects.append(obj + (obj_mask,))
        else:
            objects.append(obj)
    return objects


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


def find_connected_components_colordiff(input, is_diag=True, color=True, is_mask=False):
    """
    Find all the connected components, considering color.
    
    :param input: Input tensor of shape (10, H, W)
    :param is_diag: whether or not diagonal connections should be considered
    as part of the same object.
    :param color: whether or not to divide by color of each object.
    :returns: list of tuples of the form (row_i, col_i, rows, cols)
    """
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
            objs = find_connected_components(color_mask, is_diag=is_diag, is_mask=is_mask)
            objects += objs
    return objects


def mask_iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """
    Computes the IoU (Intersection over Union) score between two masks.

    Args:
        pred_mask: tensor of shape [..., 1, H, W], where each value is either 0 or 1.
        target_mask: tensor [..., 1, H, W], where each value is either 0 or 1.

    Returns:
        The IoU score, wish shape [...], in the range [0, 1].
    """
    pred_mask = pred_mask.round() if torch.is_floating_point(pred_mask) else pred_mask
    return torch.sum(torch.logical_and(pred_mask, target_mask), dim=(-3, -2, -1)) / \
            torch.sum(torch.logical_or(pred_mask, target_mask), dim=(-3, -2, -1)).clamp(min=1)


def score_fun_IoU(pred,
                  target,
                  exclude: torch.Tensor=None):
    """
    Obtain the matching score between two arbitrary shaped 2D matrices.

    The final score is the number of matched elements for all common patches,
    divided by height_max * width_max that use the maximum dimension of pred and target.
    This is in similar spirit as IoU (Intersection over Union).
    """

    # Obtain the value the Concept graph is holding:
    if not isinstance(pred, torch.Tensor) and not isinstance(pred, np.ndarray):
        pred = pred.get_root_value()
    if not isinstance(target, torch.Tensor) and not isinstance(target, np.ndarray):
        target = target.get_root_value()
    
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    pred_size_compare = None

    if not (len(pred.shape) == len(target.shape) == 2):
        # Pred and target are not both 2D matrices:
        return None, {"error": "Pred and target are not both 2D matrices."}
    
    excluded = 0
    if exclude is not None:
        excluded = torch.logical_and(pred == exclude, target == exclude).sum().item()

    # Find which matrix is larger and which is smaller

    if pred.shape[0] <= target.shape[0] and pred.shape[1] <= target.shape[1] or pred.shape[0] > target.shape[0] and pred.shape[1] > target.shape[1]:
        if pred.shape[0] <= target.shape[0] and pred.shape[1] <= target.shape[1]:
            pred_size_compare = ("smaller", "smaller")
            patch_s, patch_l = pred, target
 
        elif pred.shape[0] > target.shape[0] and pred.shape[1] > target.shape[1]:
            pred_size_compare = ("larger", "larger")
            patch_s, patch_l = target, pred

        shape_s = patch_s.shape
        shape_l = patch_l.shape
        best_idx = None
        max_score = -1
        for i in range(shape_l[0] - shape_s[0] + 1):
            for j in range(shape_l[1] - shape_s[1] + 1):
                patch_l_chosen = patch_l[i: i + shape_s[0], j: j + shape_s[1]]
                score = masked_equal(patch_l_chosen, patch_s, exclude=exclude).sum()
                if score > max_score:
                    best_idx = (i, j)
                    max_score = score
        final_score = max_score.float() / (shape_l[0] * shape_l[1] - excluded)
    else:
        if pred.shape[0] <= target.shape[0] and pred.shape[1] > target.shape[1]:
            pred_size_compare = ("smaller", "larger")
            height_s, height_l = pred.shape[0], target.shape[0]
            width_s, width_l = target.shape[1], pred.shape[1]
            pred_height_smaller = True
        else:
            pred_size_compare = ("larger", "smaller")
            height_l, height_s = pred.shape[0], target.shape[0]
            width_l, width_s = target.shape[1], pred.shape[1]
            pred_height_smaller = False
        best_idx = None
        max_score = -1
        for i in range(height_l - height_s + 1):
            for j in range(width_l - width_s + 1):
                if pred_height_smaller:
                    pred_chosen = pred[:, j: j + width_s]
                    target_chosen = target[i: i + height_s, :]
                else:
                    pred_chosen = pred[i: i + height_s, :]
                    target_chosen = target[:, j: j + width_s]

                score = masked_equal(pred_chosen, target_chosen, exclude=exclude).sum()
                if score > max_score:
                    best_idx = (i, j)
                    max_score = score
        final_score = max_score.float() / (height_l * width_l - excluded)
    info = {"best_idx": best_idx,
            "pred_size_compare": pred_size_compare}
    return final_score, info


def check_result_true(results):
    """Check if there exists a node where the result is True for all examples."""
    is_result_true = False
    node_true = None
    for node_key, result in results.items():
        value_list = []
        for example_key, value in result.items():
            if not isinstance(value, torch.Tensor):
                value = value.get_root_value()
            if isinstance(value, torch.BoolTensor) and tuple(value.shape) == ():
                value_list.append(value)
        if len(value_list) > 0:
            is_result_true = to_np_array(torch.stack(value_list).all())
            if is_result_true:
                node_true = node_key
                break
    return is_result_true, node_true


def get_obj_bounding_pos(objs):
    """Get the pos for the bounding box for a dictionary of objects."""
    row_min = np.Inf
    row_max = -np.Inf
    col_min = np.Inf
    col_max = -np.Inf
    for obj_name, obj in objs.items():
        pos = obj.get_node_value("pos")
        if pos[0] < row_min:
            row_min = int(pos[0])
        if pos[0] + pos[2] > row_max:
            row_max = int(pos[0] + pos[2])
        if pos[1] < col_min:
            col_min = int(pos[1])
        if pos[1] + pos[3] > col_max:
            col_max = int(pos[1] + pos[3])
    pos_bounding = (row_min, col_min, row_max - row_min, col_max - col_min)
    return pos_bounding


def get_comp_obj(obj_dict, CONCEPTS):
    """Get composite object from multiple objects."""
    pos_bounding = get_obj_bounding_pos(obj_dict)
    comp_obj = CONCEPTS["Image"].copy().set_node_value(torch.zeros(int(pos_bounding[2]), int(pos_bounding[3])))
    for obj_name, obj in obj_dict.items():
        obj_copy = obj.copy()
        pos_obj = obj_copy.get_node_value("pos")
        obj_copy.set_node_value([pos_obj[0] - pos_bounding[0], pos_obj[1] - pos_bounding[1], pos_obj[2], pos_obj[3]], "pos")
        comp_obj.add_obj(obj_copy, obj_name=obj_name, change_root=True)
    comp_obj.set_node_value([0, 0, int(pos_bounding[2]), int(pos_bounding[3])], "pos")
    return comp_obj, pos_bounding


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


def get_op_type(op_name):
    """Get the type of the op.

    Node naming convention:
        op:          operator, e.g. "Draw"
        op-in:       operator's input and output nodes, e.g. "Draw-1:Image"
        op-attr:     attribute from an op output, e.g. "Draw-o^color:Image", "input-1^pos:Pos"
        op-sc:       a concept node belonging to an operator's input's selector, e.g. "Draw-1->sc$obj_0:c0"
        op-so:       a relation node belonging to an operator's input's selector, e.g. "Draw-1->so$(obj_0:c0,obj_1:c1):r1"
        op-op:       operator's inner operator, e.g. "ForGraph->op$Copy"
        input:       input_placeholder_nodes, e.g. "input-1:Image"
        concept:     constant concept node, e.g. "concept-1:Image"
        o:           operator definition node, e.g. "o$Draw"
        c:           concept definition node, e.g. "c$Image"
        result:      input or intermediate nodes, e.g. "result$Identity->0:Color", "result$Draw->0->obj_1:Image" (The obj_1 at the 0th example at the outnode of "Draw")
        target:      target nodes, e.g. "target$0:Color", "target$1->obj_1:Image" (The obj_1 at the 1th example at the target)
        opparse:     parsed op, e.g. "opparse$Draw->0->obj_1->RotateA"

    Returns:
        op_type: string indicating the type of the node.
    """
    if isinstance(op_name, tuple) or isinstance(op_name, list):
        op_types = tuple([get_op_type(op_name_ele) for op_name_ele in op_name])
        return op_types
    elif isinstance(op_name, dict) or isinstance(op_name, set):
        raise Exception("op_name can only be a tuple, a list or a string!")
    if op_name.startswith("target$"):
        op_type = "target"
    elif "->" in op_name:
        if op_name.startswith("result$"):
            op_type = "result"
        elif op_name.startswith("opparse$"):
            op_type = "opparse"
        else:
            op_name_core = op_name.split("->")[-1]
            type_name, op_name_core = op_name_core.split("$")
            if type_name == "sc":
                op_type = "op-sc"
            elif type_name == "so":
                op_type = "op-so"
            elif type_name == "op":
                op_type = "op-op"
            else:
                raise
    else:
        if "^" in op_name:
            op_type = "op-attr"
        else:
            if "input" in op_name:
                op_type = "input"
            elif "concept" in op_name:
                op_type = "concept"
            elif op_name.startswith("o$"):
                op_type = "o"
            elif op_name.startswith("c$"):
                op_type = "c"
            elif ":" in op_name:
                if "-o" in op_name:
                    raise Exception("{} is not a valid node for node_type!".format(op_name))
                else:
                    op_type = "op-in"
            else:
                op_type = "op"
    return op_type


def get_edge_path(path):
    """Get the edge_path, where path is e.g. 
        [('target$0->obj_4:Image', 'N-target-parentResult', 'result$Draw->0->obj_2:Image'),
         ('result$Draw->0->obj_2:Image', 'N-result-parent', 'Draw'),
         ('Draw', 'N-op-result', 'result$Draw->3:Image')], and will return
        ('N-target-parentResult', 'N-result-parent', 'N-op-result').
    """
    return tuple([ele[1] for ele in path])


def get_ids_same_value(logits, id, epsilon=1e-6):
    """Get the ids of that logits that has the same value as logits[id]."""
    return to_np_array(torch.where((logits - logits[id]).abs() < epsilon)[0], full_reduce=False).tolist()


def normalize_same_value(logits, is_normalize=True):
    """For repetitive numbers in logits, regard them only appear once."""
    if is_normalize:
        logits_value = logits.detach()
        same_value_array = (logits_value[:, None] == logits_value[None, :]).sum(0).float()
        return logits - torch.log(same_value_array)
    else:
        return logits


def get_normalized_entropy(dist, is_normalize=True):
    """Compute the entropy for probability with repetitive values."""
    entropy = dist.entropy()
    if is_normalize:
        logits_value = dist.logits.detach()
        same_value_array = (logits_value[:, None] == logits_value[None, :]).sum(0).float()
        entropy = entropy - (dist.probs * same_value_array.log()).sum()
    return entropy


def clip_grad(optimizer):
    """Clip gradient"""
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))


def get_module_parameters(List, device):
    """Get the learnable parameters in a dictionary, used for learning the embedding for OPERATORS, CONCEPTS, NEIGHBOR_EMBEDDING, etc."""
    parameters_dict = {}
    for item in List:
        for op_name, op in item.items():
            if not isinstance(op, torch.Tensor):
                op = op.get_node_repr()
            op.data = op.data.to(device)
            parameters_dict[op_name] = op
    return parameters_dict


def assign_embedding_value(embedding_parameters_dict, List):
    """Assign the values to the concept representations."""
    for op_name_save, value in embedding_parameters_dict.items():
        for Dict in List:
            for op_name, op in Dict.items():
                if op_name_save == op_name:
                    if isinstance(op, torch.Tensor):
                        op.data = torch.FloatTensor(value)
                    else:
                        op.get_node_repr().data = torch.FloatTensor(value)


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing


def persist_hash(string_repr):
    import hashlib
    return int(hashlib.md5(string_repr.encode('utf-8')).hexdigest(), 16)


def tensor_to_string(tensor):
    """Transform a tensor into a string, for hashing purpose."""
    if tensor is None:
        return "None"
    return ",".join([str(ele) for ele in np.around(to_np_array(tensor), 4).flatten().tolist()])


def get_repr_dict(dct):
    """Assumes all elements of the dictionary have get_string_repr"""
    string = ""
    for key, item in dct.items():
        if isinstance(item, torch.Tensor):
            string += "({}){}".format(key, tensor_to_string(item))
        else:
            string += "({}){}".format(key, item.get_string_repr())
    return string

def check_same_tensor(List):
    """Returns True if all the element tensors of the List have the same shape and value."""
    element_0 = List[0]
    is_same = True
    for element in List[1:]:
        if tuple(element.shape) != tuple(element_0.shape) or not (element == element_0).all():
            is_same = False
            break
    return is_same


def repeat_n(*args, **kwargs):
    List = []
    for tensor in args:
        if tensor is None:
            result = None
        elif isinstance(tensor, tuple):
            result = tuple([repeat_n(ele, **kwargs) for ele in tensor])
        else:
            result = tensor.repeat(kwargs["n_repeats"], *torch.ones(len(tensor.shape)-1).long())
        List.append(result)
    if len(args) > 1:
        return tuple(List)
    else:
        return List[0]


def identity_fun(input):
    return input


class Task_Dict(dict):
    """
    A dictionary storing the buffer for the top K solution for each task. It has the following structure:

    {task_hash1: {
        TopKList([
            {"graph_hash": graph_hash,
             "score": score,
             "graph_state": graph_state,
             "action_record": action_record,
            }
        ]),
     task_hash2: {...},
     ...
    }
    """
    def __init__(self, K, mode="max"):
        self.K = K
        self.mode = mode

    def init_task(self, task_hash):
        """Initialize a new task."""
        assert task_hash not in self
        self[task_hash] = TopKList(K=self.K, sort_key="score", duplicate_key="graph_hash", mode=self.mode)

    def reset_task(self, task_hash):
        """Reset an existing task or initialize a new task."""
        self[task_hash] = TopKList(K=self.K, sort_key="score", duplicate_key="graph_hash", mode=self.mode)

    @property
    def n_examples_all(self):
        """Return a dictionary of number of examples in a task."""
        return {task_hash: len(top_k_list) for task_hash, top_k_list in self.items()}

    @property
    def n_examples_per_task(self):
        """Get average number of examples per task."""
        return np.mean(list(self.n_examples_all.values()))

    @property
    def std_examples_per_task(self):
        """Get average number of examples per task."""
        return np.std(list(self.n_examples_all.values()))

    @property
    def score_all(self):
        return {task_hash: np.mean([element["score"] for element in top_k_list]) for task_hash, top_k_list in self.items()}

    @property
    def mean_score(self):
        return np.mean(list(self.score_all.values()))

    @property
    def std_score(self):
        return np.std(list(self.score_all.values()))

    def update_score_with_ebm_dict(self, ebm_dict, cache_forward=True, is_update_score=True):
        """Update the score using the new ebm_dict."""
        self.ebm_dict = ebm_dict
        for task_hash, task_top_k_list in self.items():
            for element in task_top_k_list:
                element["graph_state"].set_ebm_dict(ebm_dict)
                element["graph_state"].set_cache_forward(False)  # Important: clear the cache
                if is_update_score:
                    element["score"] = element["graph_state"].get_score()
                if cache_forward:
                    element["graph_state"].set_cache_forward(True)
        return self

    def to(self, device):
        """Move all modules to device."""
        self.ebm_dict.to(device)
        for task_hash, task_top_k_list in self.items():
            for element in task_top_k_list:
                if "graph_state" in element:
                    element["graph_state"].to(device)
                if "selector" in element:
                    element["selector"] = element["selector"].to(device)
        return self


class Combined_Dict(dict):
    """Dictionary holding the EBMs. The parameters of the EBMs can is independent."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ebm_share_param = False

    def set_is_relation_z(self, is_relation_z):
        self.is_relation_z = is_relation_z
        return self

    def parameters(self):
        return itertools.chain.from_iterable([ebm.parameters() for key, ebm in self.items()])

    def to(self, device):
        """Move all modules to device."""
        for key in self:
            self[key] = self[key].to(device)
        return self

    @property
    def model_dict(self):
        """Returns the model_dict for each model."""
        return {key: model.model_dict for key, model in self.items()}


def get_instance_keys(class_instance):
    """Get the instance keys of a class"""
    return [key for key in vars(class_instance) if key[:1] != "_"]


class Model_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.__class__.__name__ == "ConceptEBM"
        self.model = model

    def set_c(self, c_repr):
        self.c_repr = c_repr
        return self

    def forward(self, *args, **kwargs):
        kwargs["c_repr"] = self.c_repr
        return self.model.forward(*args, **kwargs)

    def classify(self, *args, **kwargs):
        return self.model.classify(*args, **kwargs)

    def ground(self, *args, **kwargs):
        return self.model.ground(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def __getattribute__(self, item):
        """Obtain the attributes. Prioritize the instance attributes in self.model."""
        if item == "model":
            return object.__getattribute__(self, "model")
        elif item.startswith("_"):
            return object.__getattribute__(self, item)
        elif item != "c_repr" and item in get_instance_keys(self.model):
            return getattr(self.model, item)
        else:
            return object.__getattribute__(self, item)

    @property
    def model_dict(self):
        return self.model.model_dict


class Shared_Param_Dict(nn.Module):
    """Dictionary holding the EBMs. The parameters of the EBMs can is shared."""
    def __init__(
        self,
        concept_model=None,
        relation_model=None,
        concept_repr_dict=None,
        relation_repr_dict=None,
        is_relation_z=True,
    ):
        super().__init__()
        self.concept_model = concept_model
        self.relation_model = relation_model
        self.concept_repr_dict = {}
        if concept_repr_dict is not None:
            for key, item in concept_repr_dict.items():
                self.concept_repr_dict[key] = torch.FloatTensor(item) if not isinstance(item, torch.Tensor) else item
        self.relation_repr_dict = {}
        if relation_repr_dict is not None:
            for key, item in relation_repr_dict.items():
                self.relation_repr_dict[key] = torch.FloatTensor(item) if not isinstance(item, torch.Tensor) else item
        self.is_ebm_share_param = True
        self.is_relation_z = is_relation_z

    def set_is_relation_z(self, is_relation_z):
        self.is_relation_z = is_relation_z
        return self

    def add_c_repr(self, c_repr, c_str, ebm_mode):
        if ebm_mode == "concept":
            self.concept_repr_dict[c_str] = c_repr
        elif ebm_mode == "operator":
            self.relation_repr_dict[c_str] = c_repr
        else:
            raise Exception("ebm_mode {} is not valid!".format(ebm_mode))
        return self

    def update(self, new_dict):
        assert new_dict.__class__.__name__ == "Shared_Param_Dict"
        if new_dict.concept_model is not None:
            self.concept_model = new_dict.concept_model
        if new_dict.relation_model is not None:
            self.relation_model = new_dict.relation_model
        for key, item in new_dict.concept_repr_dict.items():
            if key in self:
                assert (item == self.concept_repr_dict[key]).all()
            self.concept_repr_dict[key] = item
        for key, item in new_dict.relation_repr_dict.items():
            if key in self:
                assert (item == self.relation_repr_dict[key]).all()
            self.relation_repr_dict[key] = item
        return self

    def __setitem__(self, key, model):
        assert model.__class__.__name__ == "ConceptEBM"
        if model.mode == "concept":
            assert self.concept_model is None
            self.concept_model = model
            self.concept_repr_dict[model.c_str] = model.c_repr
        elif model.mode == "operator":
            assert self.relation_model is None
            self.relation_model = model
            self.relation_repr_dict[model.c_str] = model.c_repr
        else:
            raise Exception("ebm_mode '{}' is not valid!".format(model.mode))

    def __getitem__(self, key):
        if key in self.concept_repr_dict:
            return Model_Wrapper(self.concept_model).set_c(c_repr=self.concept_repr_dict[key])
        elif key in self.relation_repr_dict:
            return Model_Wrapper(self.relation_model).set_c(c_repr=self.relation_repr_dict[key])
        else:
            raise Exception("key '{}' not in concept_repr_dict nor relation_repr_dict.".format(key))

    def keys(self):
        return list(self.concept_repr_dict.keys()) + list(self.relation_repr_dict.keys())

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()

    def has_key(self, k):
        return k in self.keys()

    def is_model_exist(self, ebm_mode):
        if ebm_mode == "concept":
            return self.concept_model is not None
        elif ebm_mode == "operator":
            return self.relation_model is not None
        else:
            raise Exception("ebm_mode {} is not valid!".format(ebm_mode))

    def parameters(self):
        iterables = []
        if self.concept_model is not None:
            iterables.append(self.concept_model.parameters())
        if self.relation_model is not None:
            iterables.append(self.relation_model.parameters())
        return itertools.chain.from_iterable(iterables)

    def to(self, device):
        """Move all modules to device."""
        if self.concept_model is not None:
            self.concept_model.to(device)
        if self.relation_model is not None:
            self.relation_model.to(device)
        for key in self.concept_repr_dict:
            self.concept_repr_dict[key] = self.concept_repr_dict[key].to(device)
        for key in self.relation_repr_dict:
            self.relation_repr_dict[key] = self.relation_repr_dict[key].to(device)
        return self

    @property
    def model_dict(self):
        model_dict = {"type": "Shared_Param_Dict"}
        model_dict["concept_model_dict"] = self.concept_model.model_dict if self.concept_model is not None else None
        model_dict["relation_model_dict"] = self.relation_model.model_dict if self.relation_model is not None else None
        model_dict["concept_repr_dict"] = {key: to_np_array(item) for key, item in self.concept_repr_dict.items()}
        model_dict["relation_repr_dict"] = {key: to_np_array(item) for key, item in self.relation_repr_dict.items()}
        return model_dict


def get_str_value(string_to_split, string):
    string_splited = string_to_split.split("-")
    if string in string_splited:
        return eval(string_splited[string_splited.index(string)+1])
    else:
        return None

def is_diagnose(loc, filename):
    """If the given loc and filename matches that of the diagose.yml, will return True and (later) call an pde.set_trace()."""
    try:
        with open(get_root_dir() + "/experiments/diagnose.yml", "r") as f:
            Dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        return False
    if Dict is None:
        return False
    Dict.pop(None, None)
    if not ("loc" in Dict and "dirname" in Dict and "filename" in Dict):
        return False
    if loc == Dict["loc"] and filename == Dict["dirname"] + Dict["filename"]:
        return True
    else:
        return False


def model_parallel(model, args):
    if args.parallel_mode == "None":
        device = get_device(args)
        model.to(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            if args.parallel_mode == "ddp":
                if args.local_rank >= 0:
                    torch.cuda.set_device(args.local_rank) 
                    device = torch.device(f"cuda:{args.local_rank}")
                torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=torch.cuda.device_count())
                torch.set_num_threads(os.cpu_count()//(torch.cuda.device_count() * 2))
                model.to(device)
                model = torch.nn.parallel.DistributedDataParallel(model)
            elif args.parallel_mode == "dp":
                model = MineDataParallel(model)
                model.to(device)
    return model, device


class MineDataParallel(nn.parallel.DataParallel):
    def __getattribute__(self, key):
        module_attrs = [
            'training',
            'mode',
            'in_channels',
            'repr_dim',
            'w_type',
            'w_dim',
            'mask_mode',
            'channel_base',
            'two_branch_mode',
            'mask_arity',
            'is_spec_norm',
            'is_res',
            'c_repr_mode',
            'c_repr_first',
            'c_repr_base',
            'z_mode',
            'z_first',
            'z_dim',
            'pos_embed_mode',
            'aggr_mode',
            'img_dims',
            'act_name',
            'normalization_type',
            'dropout',
            'self_attn_mode',
            'last_act_name',
            'n_avg_pool',
            'model_dict',
        ]
        if key in module_attrs:
            return object.__getattribute__(self.module, key)
        else:
            if hasattr(MineDataParallel, key):
                return object.__getattribute__(self, key)
            else:
                return super().__getattribute__(key)


def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        is_int = False
        if isinstance(array, Number):
            is_int = True if isinstance(array, int) else False
            array = [array]
        if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, tuple):
            is_int = True if np.array(array).dtype.name == "int64" else False
            array = torch.tensor(array).float()
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad=requires_grad)
        if "preserve_int" in kwargs and kwargs["preserve_int"] is True and is_int:
            array = array.long()
        array = set_cuda(array, is_cuda)
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def set_cuda(tensor, is_cuda):
    if isinstance(is_cuda, str):
        return tensor.cuda(is_cuda)
    else:
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor


def to_Variable_recur(item, type='float'):
    """Recursively transform numpy array into PyTorch tensor."""
    if isinstance(item, dict):
        return {key: to_Variable_recur(value, type=type) for key, value in item.items()}
    elif isinstance(item, tuple):
        return tuple(to_Variable_recur(element, type=type) for element in item)
    else:
        try:
            if type == "long":
                return torch.LongTensor(item)
            elif type == "float":
                return torch.FloatTensor(item)
            elif type == "bool":
                return torch.BoolTensor(item)
        except:
            return [to_Variable_recur(element, type=type) for element in item]


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if array is None:
            array_list.append(array)
            continue
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
        if not ("keep_list" in kwargs and kwargs["keep_list"]):
            array_list = array_list[0]
    return array_list


def record_data(data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False, recent_record=-1):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)
            if recent_record != -1:
                # Only keep the most recent records
                data_record_dict[key] = data_record_dict[key][-recent_record:]


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def split_string(string):
    """Given a string, return the core string and the number suffix.
    If there is no number suffix, the num_core will be None.
    """
    # Get the starting index for the number suffix:
    assert isinstance(string, str)
    i = 1
    for i in range(1, len(string) + 1):
        if string[-i] in [str(k) for k in range(10)]:
            continue
        else:
            break
    idx = len(string) - i + 1
    # Obtain string_core and num_core:
    string_core = string[:idx]
    if len(string[idx:]) > 0:
        num_core = eval(string[idx:])
    else:
        num_core = None

    return string_core, num_core


def broadcast_keys(key_list_all):
    """Return a fully broadcast {new_broadcast_key: list of Arg keys}

    key_list_all: a list of items, where each item is a list of keys.
    For example, key_list_all = [[(0, "s"), (0, "d"), (1, "d)], [0, 1], None], it will return:

    key_dict = {(0, "s"): [(0, "s"), 0, None],
                (0, "d"): [(0, "d"), 0, None],
                (1, "d"): [(1, "d"), 1, None]}
    Here None denotes that there is only one input, which will be broadcast to all.
    """
    # First: get all the combinations
    new_key_list = []
    for i, keys in enumerate(key_list_all):
        if keys is not None:
            keys = [(ele,) if not isinstance(ele, tuple) else ele for ele in keys]
            new_key_list = compose_two_keylists(new_key_list, keys)

    ## new_key_list: a list of fully broadcast keys
    ## key_list_all: a list of original_key_list, each of which corresponds to
    ##               the keys of an OrderedDict()
    key_dict = {}
    for new_key in new_key_list:
        new_key_map_list = []
        is_match_all = True
        for original_key_list in key_list_all:
            if original_key_list is None:
                new_key_map_list.append(None)
                is_match = True
            else:
                is_match = False
                for key in original_key_list:
                    key = (key,) if not isinstance(key, tuple) else key
                    if set(key).issubset(set(new_key)):
                        is_match = True
                        if len(key) == 1:
                            key = key[0]
                        new_key_map_list.append(key)
                        break
            is_match_all = is_match_all and is_match
        if is_match_all:
            if len(new_key) == 1:
                new_key = new_key[0]
            key_dict[new_key] = new_key_map_list
    return key_dict


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


def filter_filename(dirname, include=[], exclude=[], array_id=None):
    """Filter filename in a directory"""
    def get_array_id(filename):
        array_id = filename.split("_")[-2]
        try:
            array_id = eval(array_id)
        except:
            pass
        return array_id
    filename_collect = []
    if array_id is None:
        filename_cand = [filename for filename in os.listdir(dirname)]
    else:
        filename_cand = [filename for filename in os.listdir(dirname) if get_array_id(filename) == array_id]
    
    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]
    
    for filename in filename_cand:
        is_in = True
        for element in include:
            if element not in filename:
                is_in = False
                break
        for element in exclude:
            if element in filename:
                is_in = False
                break
        if is_in:
            filename_collect.append(filename)
    return filename_collect


class TopKList(list):
    """A list that stores the top K dictionaries that has the lowest/highest {sort_key} values."""
    def __init__(self, K, sort_key, duplicate_key, mode="max"):
        """
        Args:
            K: top K elements will be saved in the list.
            sort_key: the key to use for the top-K ranking.
            duplicate_key: the key to check, and if there are already elements that has the same value to the duplicate, do not append.
            mode: choose from "max" (the larger the better) and "min" (the smaller the better).
        """
        self.K = K
        self.sort_key = sort_key
        self.duplicate_key = duplicate_key
        self.mode = mode

    def append(self, item):
        """Insert an item into the list, if the length is less then self.K, simply insert.
        Otherwise if it is better than the worst element in terms of sort_key, replace it.

        Returns:
            is_update: whether the TopKList is updated.
        """
        assert isinstance(item, dict), "item must be a dictionary!"
        assert self.sort_key in item, "item must have sort_key of '{}'!".format(self.sort_key)
        is_update = False

        # If there are already elements that has the same value to the duplicate, do not append:
        is_duplicate = False
        for element in self:
            if element[self.duplicate_key] == item[self.duplicate_key]:
                is_duplicate = True
                break
        if is_duplicate:
            return is_update

        # Append if still space or the value for self.sort_key is better than the worst:
        if len(self) < self.K:
            super().append(item)
            is_update = True
        elif len(self) == self.K:
            sort_value = np.array([to_np_array(ele[self.sort_key]) for ele in self])
            if self.mode == "max":
                argmin = sort_value.argmin()
                if sort_value[argmin] < item[self.sort_key]:
                    self.pop(argmin)
                    super().append(item)
                    is_update = True
            elif self.mode == "min":
                argmax = sort_value.argmax()
                if sort_value[argmax] > item[self.sort_key]:
                    self.pop(argmax)
                    super().append(item)
                    is_update = True
            else:
                raise Exception("mode must be either 'min' or 'max'.")
        else:
            raise Exception("Cannot exceed K={} items".format(self.K))
        return is_update

    def get_items(self, key):
        """Obtain the item corresponding to the key for each element."""
        return [item[key] for item in self]

    def is_available(self):
        """Return True if the number of elements is less than self.K."""
        return len(self) < self.K


def get_device(args):
    """Initialize PyTorch device.

    Args:
        args.gpuid choose from an integer or True or False.
    """
    is_cuda = eval(args.gpuid)
    if not isinstance(is_cuda, bool):
        is_cuda = "cuda:{}".format(is_cuda)
    device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")
    return device


def filter_kwargs(kwargs, param_names=None, contains=None):
    """Build a new dictionary based on the filtering criteria.

    Args:
        param_names: if not None, will find the keys that are in the list of 'param_names'.
        contains: if not None, will find the keys that contain the substrings in the list of 'contains'.

    Returns:
        new_kwargs: new kwargs dictionary.
    """
    new_kwargs = {}
    if param_names is not None:
        assert contains is None
        if not isinstance(param_names, list):
            param_names = [param_names]
        for key, item in kwargs.items():
            if key in param_names:
                new_kwargs[key] = item
    else:
        assert contains is not None
        if not isinstance(contains, list):
            contains = [contains]
        for key, item in kwargs.items():
            for ele in contains:
                if ele in key:
                    new_kwargs[key] = item
                    break
    return new_kwargs


def gather_broadcast(tensor, dim, index):
    """
    Given tensor, gather the index along the dimension dim.
        For example, if tensor has shape [3,4,5,8,9], dim=2, then index must have
        the shape of [3,4], whose value is inside range(5), and returns a tensor_gather
        of size [3,4,8,9].
    """
    dim_size = tensor.shape[dim]
    assert len(index.shape) == dim and tensor.shape[:dim] == index.shape and index.max() < dim_size
    assert dim >= 1, "dim must >= 1!"
    index_onehot = torch.eye(dim_size, dim_size)[index].bool()
    tensor_gathered = tensor[index_onehot].reshape(*index.shape, *tensor.shape[dim+1:])
    return tensor_gathered


def Zip(*data, **kwargs):
    """Recursive unzipping of data structure
    Example: Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)])
    ==> [[['a', 'b', 'c', 'd'], [2, 3, 3, 2]], [1, 2, 3, 4]]
    Each subtree in the original data must be in the form of a tuple.
    In the **kwargs, you can set the function that is applied to each fully unzipped subtree.
    """
    import collections
    function = kwargs["function"] if "function" in kwargs else None
    if len(data) == 1 and function is None:
        return data[0]
    data = [list(element) for element in zip(*data)]
    for i, element in enumerate(data):
        if isinstance(element[0], tuple):
            data[i] = Zip(*element, **kwargs)
        elif isinstance(element, list):
            if function is not None:
                data[i] = function(element)
    return data


def init_args(args_dict):
    """Init argparse from dictionary."""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.__dict__ = args_dict
    return args


def update_args(args, key, value):
    args_update = deepcopy(args)
    setattr(args_update, key, value)
    return args_update


def get_machine_name():
    return os.uname()[1].split('.')[0]


def get_filename(short_str_dict, args_dict, suffix=".p"):
    """Get the filename using given short_str_dict and info of args_dict.

    Args:
        short_str_dict: mapping of long args name and short string, e.g. {"dataset": "data", "epoch": "ep", "lr": "lr#1"} (the #{Number} means multiple
            args share the same short_str).
        args_dict: args.__dict__.
        suffix: suffix for the filename.
    """
    string_list = []
    for k, v in short_str_dict.items():
        if args_dict[k] is None:
            continue
        elif v == "":
            string_list.append(args_dict[k])
        else:
            if len(v.split("#")) == 2:
                id = eval(v.split("#")[1])
                if id == 1:
                    string_list.append("{}_{}".format(v.split("#")[0], args_dict[k]))
                else:
                    string_list.append("{}".format(args_dict[k]))
            elif k == "gpuid":
                string_list.append("{}:{}".format(v, args_dict[k]))
            else:
                string_list.append("{}_{}".format(v, args_dict[k]))
    return "_".join(string_list) + suffix


def get_filename_short(
    args_shown,
    short_str_dict,
    args_dict,
    hash_exclude_shown=False,
    hash_length=8,
    print_excluded_args=False,
    suffix=".p"
):
    """Get the filename using given short_str_dict, args_shown and info of args_dict.
        The args not in args_shown will not be put explicitly in the filename, but the full
        args_dict will be turned into a unique hash.

    Args:
        args_shown: fields of the args that need to appear explicitly in the filename
        short_str_dict: mapping of long args name and short string, e.g. {"dataset": "data", "epoch": "ep"}
        args_dict: args.__dict__.
        hash_exclude_shown: if True, will exclude the args that are in the args_shown when computing the hash.
        hash_length: length of the hash.
        suffix: suffix for the filename.
    """
    # Get the short name from short_str_dict:
    str_dict = {}
    for key in args_shown:
        if key in args_dict:
            str_dict[key] = short_str_dict[key]
        else:
            raise Exception("'{}' not in the short_str_dict. Need to add its short name into it.".format(key))
    short_filename = get_filename(str_dict, args_dict, suffix="")

    # Get the hashing for the full args_dict:
    args_dict_excluded = deepcopy(args_dict)
    for key in args_shown:
        args_dict_excluded.pop(key)
    if print_excluded_args:
        print("Excluded args in explicit filename: {}".format(list(args_dict_excluded)))
    hashing = get_hashing(str(args_dict_excluded) if hash_exclude_shown else str(args_dict), length=hash_length)
    return short_filename + "_Hash_{}{}".format(hashing, suffix)


def to_cpu(state_dict):
    state_dict_cpu = {}
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    return state_dict_cpu


def to_cpu_recur(item, to_target=None):
    if isinstance(item, dict):
        return {key: to_cpu_recur(value, to_target=to_target) for key, value in item.items()}
    elif isinstance(item, list):
        return [to_cpu_recur(element, to_target=to_target) for element in item]
    elif isinstance(item, tuple):
        return tuple(to_cpu_recur(element, to_target=to_target) for element in item)
    elif isinstance(item, set):
        return {to_cpu_recur(element, to_target=to_target) for element in item}
    else:
        if isinstance(item, torch.Tensor):
            if item.is_cuda:
                item = item.cpu()
            if to_target is not None and to_target == "np":
                item = item.detach().numpy()
            return item
        if to_target is not None and to_target == "torch":
            if isinstance(item, np.ndarray):
                item = torch.FloatTensor(item)
                return item
        return item


def try_call(fun, args=None, kwargs=None, time_interval=5, max_n_trials=20, max_exp_time=None):
    """Try executing some function fun with *args and **kwargs for {max_n_trials} number of times
        each separate by time interval of {time_interval} seconds.
    """
    if args is None:
        args = []
    if not isinstance(args, list):
        args = [args]
    if kwargs is None:
        kwargs = {}
    if max_exp_time is None:
        time_interval_list = [time_interval] * max_n_trials
    else:
        time_interval_list = [2 ** k for k in range(20) if 2 ** (k + 1) <= max_exp_time]
    for i, time_interval in enumerate(time_interval_list):
        is_succeed = False
        try:
            output = fun(*args, **kwargs)
            is_succeed = True
        except Exception as e:
            error = str(e)
        if is_succeed:
            break
        else:
            print("Fail to execute function {} for the {}th time, with error: {}".format(fun, i+1, error))
        time.sleep(time_interval)
    if not is_succeed:
        raise Exception("Fail to execute function {} for the {}th time, same as the max_n_trials of {}. Check error!".format(fun, i+1, max_n_trials))
    return output


def extend_dims(tensor, n_dims, loc="right"):
    """Extends the dimensions by appending 1 at the right or left of the shape.

    E.g. if tensor has shape of (4, 6), then 
        extend_dims(tensor, 4, "right") has shape of (4,6,1,1);
        extend_dims(tensor, 4, "left")  has shape of (1,1,4,6).
    """
    if loc == "right":
        while len(tensor.shape) < n_dims:
            tensor = tensor[..., None]
    elif loc == "left":
        while len(tensor.shape) < n_dims:
            tensor = tensor[None]
    else:
        raise
    return tensor


def transform_dict(Dict, mode="array"):
    if mode == "array":
        return {key: np.array(item) for key, item in Dict.items()}
    if mode == "concatenate":
        return {key: np.concatenate(item) for key, item in Dict.items()}
    elif mode == "torch":
        return {key: torch.FloatTensor(item) for key, item in Dict.items()}
    elif mode == "mean":
        return {key: np.mean(item) for key, item in Dict.items()}
    elif mode == "std":
        return {key: np.std(item) for key, item in Dict.items()}
    elif mode == "sum":
        return {key: np.sum(item) for key, item in Dict.items()}
    elif mode == "prod":
        return {key: np.prod(item) for key, item in Dict.items()}
    else:
        raise


def get_soft_IoU(mask1, mask2, dim, epsilon=1):
    """Get soft IoU score for two masks.

    Args:
        mask1, mask2: two masks with the same shape and value between [0, 1]
        dim: dimensions over which to aggregate.
    """
    if isinstance(mask1, np.ndarray):
        soft_IoU = (mask1 * mask2).sum(dim) / (mask1 + mask2 - mask1 * mask2).sum(dim).clip(epsilon, None)
    else:
        soft_IoU = (mask1 * mask2).sum(dim) / (mask1 + mask2 - mask1 * mask2).sum(dim).clamp(epsilon)
    return soft_IoU


def get_soft_Jaccard_distance(mask1, mask2, dim, epsilon=1):
    """Get soft Jaccard distance for two masks."""
    return 1 - get_soft_IoU(mask1, mask2, dim=dim, epsilon=epsilon)


class Attr_Dict(dict):
    def __init__(self, *a, **kw):
        dict.__init__(self, *a, **kw)
        self.__dict__ = self

    def update(self, *a, **kw):
        dict.update(self, *a, **kw)
        return self

    def __getattribute__(self, key):
        if key in self:
            return self[key]
        else:
            return object.__getattribute__(self, key)

    def to(self, device):
        self["device"] = device
        Dict = to_device_recur(self, device)
        return Dict

    def copy(self, detach=True):
        return copy_data(self, detach=detach)

    def clone(self, detach=True):
        return self.copy(detach=detach)

    def type(self, dtype):
        return to_type(self, dtype)

    def detach(self):
        return detach_data(self)


def ddeepcopy(item):
    """Deepcopy with certain custom classes."""
    from pstar import pdict
    if isinstance(item, pdict) or isinstance(item, Attr_Dict):
        return item.copy()
    else:
        return deepcopy(item)


def get_connected_loss(mask_batch, num_pair_samples, dist_weights=[0.1], verbose=0):
    """The more disconnected for the object mask, the larger the loss."""
    # Takes in a batch of masks [B, 1, H, W]. Returns a batch of values [B, 1]
    assert len(mask_batch.shape) == 4
    rounded_masks = (mask_batch > 0.5).flatten(1).float()
    # Whether or not to sample 
    sample_mask = rounded_masks.sum(1)
    mask_batch_inv = 1 - mask_batch
    batch_scores = []
    
    for idx in range(rounded_masks.shape[0]):
        # Scores for each path sampled for a given mask
        mask_scores = []
        if sample_mask[idx] == 0:
            for i in range(num_pair_samples):
                mask_scores.append(torch.zeros(len(dist_weights)).to(mask_batch.device))
        else:
            all_samples = torch.multinomial(rounded_masks[idx], num_pair_samples * 2, replacement=True)
            start_ind, end_ind = all_samples[:num_pair_samples], all_samples[num_pair_samples:]
            mask = mask_batch[idx,].squeeze(0)
            mask_inv = mask_batch_inv[idx,].squeeze(0)
            for start_sample, end_sample in zip(start_ind, end_ind):
                # Last dimension of mask_batch is width
                start_row = (start_sample / mask_batch.shape[-1]).int().item()
                end_row = (end_sample / mask_batch.shape[-1]).int().item()
                start_coord = (start_row, torch.remainder(start_sample, mask_batch.shape[-1]).item())
                end_coord = (end_row, torch.remainder(end_sample, mask_batch.shape[-1]).item())
                # IMPORTANT: pass in the inverted mask
                np_mask_inv = mask_inv.cpu().detach().numpy().astype(np.float32)
                pos = astar_pos(np_mask_inv, start_coord, end_coord, dist_weights=dist_weights)
                mask_scores.append(torch.minimum(mask[start_coord], mask[end_coord]) * torch.maximum(torch.zeros(len(dist_weights)).to(mask_batch.device), 
                                                 mask_inv[(pos[:, 0], pos[:, 1])]))
        batch_scores.append(torch.stack(mask_scores))
    # Batch scores is [B, # pairs, # trials]
    batch_scores = torch.stack(batch_scores)
    # Take the minimum score over all trials for a pair. Over all pairs, take the maximum score
    batch_scores = torch.max(torch.min(batch_scores, 2)[0], 1)[0]
    return batch_scores


def to_string(List, connect = "-", num_digits = None, num_strings = None):
    """Turn a list into a string, with specified format"""
    if not isinstance(List, list) and not isinstance(List, tuple):
        return List
    if num_strings is None:
        if num_digits is None:
            return connect.join([str(element) for element in List])
        else:
            return connect.join(["{0:.{1}f}".format(element, num_digits) for element in List])
    else:
        if num_digits is None:
            return connect.join([str(element)[:num_strings] for element in List])
        else:
            return connect.join(["{0:.{1}f}".format(element, num_digits)[:num_strings] for element in List])


def check_same_set(List_of_List):
    """Return True if each element list has the same set of elements."""
    if len(List_of_List) == 0:
        return None
    List = List_of_List[0]
    for List_ele in List_of_List:
        if set(List) != set(List_ele):
            return False
    return True


def check_same_dict(Dict, value_list, key_list):
    """Check if the value stored is the same as the newly given value_list.
    Return a list of keys whose values are different from the stored ones.
    """
    if len(Dict) == 0:
        for key, value in zip(key_list, value_list):
            Dict[key] = value
        return []
    else:
        not_equal_list = []
        for key, value in zip(key_list, value_list):
            value_stored = Dict[key]
            if isinstance(value, Number) or isinstance(value, tuple) or isinstance(value, list):
                is_equal = value == value_stored
                if not is_equal:
                    not_equal_list.append(key)
            else:
                if tuple(value.shape) != tuple(value_stored.shape):
                    not_equal_list.append(key)
                else:
                    is_equal = (value == value_stored).all()
                    if not is_equal:
                        not_equal_list.append(key)
        return not_equal_list


def check_same_model_dict(model_dict1, model_dict2):
    """Check if two model_dict are the same."""
    assert set(model_dict1.keys()) == set(model_dict2.keys()), "model_dict1 and model_dict2 has different keys!"
    for key, item1 in model_dict1.items():
        item2 = model_dict2[key]
        if not isinstance(item1, dict):
            assert item1 == item2, "key '{}' has different values of '{}' and '{}'.".format(key, item1, item2)
    return True


def get_pdict():
    """Obtain pdict with additional functionalities."""
    from pstar import pdict
    class Pdict(pdict):
        def to(self, device):
            self["device"] = device
            return to_device_recur(self, device)

        def copy(self):
            return Pdict(dict.copy(self))
    return Pdict


def copy_with_model_dict(model, other_attr=None):
    """Copy a model based on its model_dict."""
    if other_attr is None:
        other_attr = []
    kwargs = model.model_dict
    state_dict = kwargs.pop("state_dict")
    assert kwargs.pop("type") == model.__class__.__name__
    other_attr_dict = {}
    for key in other_attr:
        other_attr_dict[key] = kwargs.pop(key)
    new_model = model.__class__(**kwargs)
    for key, value in other_attr_dict.items():
        if isinstance(value, np.ndarray):
            value = torch.FloatTensor(value)
        setattr(new_model, key, value)
    new_model.load_state_dict(state_dict)
    assert new_model.model_dict.keys() == model.model_dict.keys()
    return new_model


def canonicalize_strings(operators):
    """Given a list of strings, return the canonical version.
    
    Example:
        operators = ["EqualRow1", "EqualRow2", "EqualWidth3"]
    
        Returns: mapping = {'EqualRow1': 'EqualRow',
                            'EqualRow2': 'EqualRow1',
                            'EqualWidth3': 'EqualWidth'}
    """
    operators_core = [split_string(ele)[0] for ele in operators]
    counts = Counter(operators_core)
    new_counts = {key: 0 for key in counts}
    mapping = {}
    for operator, operator_core in zip(operators, operators_core):
        count = new_counts[operator_core]
        if count == 0:
            mapping[operator] = operator_core
        else:
            mapping[operator] = "{}{}".format(operator_core, count)
        new_counts[operator_core] += 1
    return mapping


def get_generalized_mean(List, cumu_mode="mean", epsilon=1e-10):
    """Get generalized-mean of elements in the list"""
    List = np.array(list(List))
    assert len(List.shape) == 1

    if cumu_mode[0] == "gm" and cumu_mode[1] == 1:
        cumu_mode = "mean"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == 0:
        cumu_mode = "geometric"
    elif cumu_mode[0] == "gm" and cumu_mode[1] == -1:
        cumu_mode = "harmonic"

    # Obtain mean:
    if cumu_mode == "mean":
        mean = List.mean()
    elif cumu_mode == "min":
        mean = np.min(List)
    elif cumu_mode == "max":
        mean = np.max(List)
    elif cumu_mode == "harmonic":
        mean = len(List) / (1 / (List + epsilon)).sum()
    elif cumu_mode == "geometric":
        mean = (List + epsilon).prod() ** (1 / float(len(List)))
    elif cumu_mode[0] == "gm":
        order = cumu_mode[1]
        mean = (np.minimum((List + epsilon) ** order, 1e30).mean()) ** (1 / float(order))
    else:
        raise
    return mean


def try_eval(string):
    """Try to evaluate a string. If failed, use original string."""
    try:
        return eval(string)
    except:
        return string


def get_rename_mapping(base_keys, adding_keys):
    """Given a list of base_keys and adding keys, return a mapping of how to 
    rename adding_keys s.t. adding_keys do not have name conflict with base_keys.
    """
    mapping = {}
    for key in adding_keys:
        if key in base_keys:
            string_core, num_core = split_string(key)
            num_proposed = num_core + 1 if num_core is not None else 1
            proposed_name = "{}{}".format(string_core, num_proposed)
            while proposed_name in base_keys:
                num_proposed += 1
                proposed_name = "{}{}".format(string_core, num_proposed)
            mapping[key] = proposed_name
    return mapping


def get_next_available_key(iterable, key, midfix="", suffix="", is_underscore=True, start_from_null=False):
    """Get the next available key that does not collide with the keys in the dictionary."""
    if start_from_null and key + suffix not in iterable:
        return key + suffix
    else:
        i = 0
        underscore = "_" if is_underscore else ""
        while "{}{}{}{}{}".format(key, underscore, midfix, i, suffix) in iterable:
            i += 1
        new_key = "{}{}{}{}{}".format(key, underscore, midfix, i, suffix)
        return new_key


def str2bool(v):
    """used for argparse, 'type=str2bool', so that can pass in string True or False."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string


class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))


class MineDataset(Dataset):
    def __init__(
        self,
        data=None,
        idx_list=None,
        transform=None,
    ):
        """User defined dataset that can be used for PyTorch DataLoader"""
        self.data = data
        self.transform = transform
        if idx_list is None:
            self.idx_list = torch.arange(len(self.data))
        else:
            self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def process_sample(self, sample):
        return sample

    def __getitem__(self, idx):
        is_list = True
        if isinstance(idx, torch.Tensor):
            if len(idx.shape) == 0 or (len(idx.shape) == 1 and len(idx) == 1):
                idx = idx.item()
                is_list = False
        elif isinstance(idx, list):
            pass
        elif isinstance(idx, Number):
            is_list = False

        if isinstance(idx, slice) or is_list:
            Dict = self.__dict__.copy()
            Dict["idx_list"] = self.idx_list[idx]
            return self.__class__(**Dict)

        sample = self.process_sample(self.data[self.idx_list[idx]])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"({len(self)})"


def reduce_tensor(tensor, reduction, dims_to_reduce=None, keepdims=False):
    """Reduce tensor using 'mean' or 'sum'."""
    if reduction == "mean":
        if dims_to_reduce is None:
            tensor = tensor.mean()
        else:
            tensor = tensor.mean(dims_to_reduce, keepdims=keepdims)
    elif reduction == "sum":
        if dims_to_reduce is None:
            tensor = tensor.sum()
        else:
            tensor = tensor.sum(dims_to_reduce, keepdims=keepdims)
    elif reduction == "none":
        pass
    else:
        raise
    return tensor


def pdump(file, filename):
    """Dump a file via pickle."""
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def pload(filename):
    """Load a filename saved as pickle."""
    with open(filename, "rb") as f:
        file = pickle.load(f)
    return file


def remove_elements(List, elements):
    """Remove elements in the List if they exist in the List, and return the new list."""
    NewList = deepcopy(List)
    for element in elements:
        if element in NewList:
            NewList.remove(element)
    return NewList


def loss_op_core(pred_core, y_core, reduction="mean", loss_type="mse", normalize_mode="None", **kwargs):
    """Compute the loss. Here pred_core and y_core must both be tensors and have the same shape. 
    Generically they have the shape of [n_nodes, pred_steps, dyn_dims].
    For hybrid loss_type, e.g. "mse+huberlog#1e-3", will recursively call itself.
    """
    if "+" in loss_type:
        loss_list = []
        precision_floor = get_precision_floor(loss_type)
        for loss_component in loss_type.split("+"):
            if precision_floor is not None and not ("mselog" in loss_component or "huberlog" in loss_component or "l1log" in loss_component):
                pred_core_new = torch.exp(pred_core) - precision_floor
            else:
                pred_core_new = pred_core
            loss_ele = loss_op_core(
                pred_core=pred_core_new,
                y_core=y_core,
                reduction=reduction,
                loss_type=loss_component,
                normalize_mode=normalize_mode,
                **kwargs
            )
            loss_list.append(loss_ele)
        loss = torch.stack(loss_list).sum()
        return loss

    if normalize_mode != "None":
        assert normalize_mode in ["targetindi", "target"]
        dims_to_reduce = list(np.arange(2, len(y_core.shape)))  # [2, ...]
        if normalize_mode == "target":
            dims_to_reduce.insert(0, 0)  # [0, 2, ...]

    if loss_type.lower() == "mse":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.mse_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.square(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.mse_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "huber":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.smooth_l1_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "l1":
        if normalize_mode in ["target", "targetindi"]:
            loss = F.l1_loss(pred_core, y_core, reduction='none')
            loss = loss / reduce_tensor(y_core.abs(), "mean", dims_to_reduce, keepdims=True)
            loss = reduce_tensor(loss, reduction)
        else:
            loss = F.l1_loss(pred_core, y_core, reduction=reduction)
    elif loss_type.lower() == "l2":
        first_dim = kwargs["first_dim"] if "first_dim" in kwargs else 2
        if normalize_mode in ["target", "targetindi"]:
            loss = L2Loss(reduction='none', first_dim=first_dim)(pred_core, y_core)
            y_L2 = L2Loss(reduction='none', first_dim=first_dim)(torch.zeros(y_core.shape), y_core)
            if normalize_mode == "target":
                y_L2 = y_L2.mean(0, keepdims=True)
            loss = loss / y_L2
            loss = reduce_tensor(loss, reduction)
        else:
            loss = L2Loss(reduction=reduction, first_dim=first_dim)(pred_core, y_core)
    elif loss_type.lower() == "dl":
        loss = DLLoss(pred_core, y_core, reduction=reduction, **kwargs)
    # loss where the target is taking the log scale:
    elif loss_type.lower().startswith("mselog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.mse_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("huberlog"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.smooth_l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    elif loss_type.lower().startswith("l1log"):
        precision_floor = eval(loss_type.split("#")[1])
        loss = F.l1_loss(pred_core, torch.log(y_core + precision_floor), reduction=reduction)
    else:
        raise Exception("loss_type {} is not valid!".format(loss_type))
    return loss


def set_seed(seed):
    """Set up seed."""
    if seed == -1:
        seed = None
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)


class Early_Stopping(object):
    """Class for monitoring and suggesting early stopping"""
    def __init__(self, patience=100, epsilon=0, mode="min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
        self.best_value = None
        self.wait = 0

    def reset(self, value=None):
        self.best_value = value
        self.wait = 0
        
    def monitor(self, value):
        if self.patience == -1:
            self.wait += 1
            return False
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or \
                   (self.mode == "max" and value > self.best_value + self.epsilon):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop

    def __repr__(self):
        return "Early_Stopping(patience={}, epsilon={}, mode={}, wait={})".format(self.patience, self.epsilon, self.mode, self.wait)


def write_to_config(args, filename):
    """Write to a yaml configuration file. The filename contains path to that file.
    """
    import yaml
    dirname = "/".join(filename.split("/")[:-1])
    config_filename = os.path.join(dirname, "config", filename.split("/")[-1][:-2] + ".yaml")
    make_dir(config_filename)
    with open(config_filename, "w") as f:
        yaml.dump(args.__dict__, f)


class Dictionary(object):
    """Custom dictionary that can avoid the collate_fn in pytorch's dataloader."""
    def __init__(self, Dict=None):
        if Dict is not None:
            for k, v in Dict.items():
                self.__dict__[k] = v

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))


class Batch(object):
    def __init__(self, is_absorb_batch=False, is_collate_tuple=False):
        """
        
        Args:
            is_collate_tuple: if True, will collate inside the tuple.
        """
        self.is_absorb_batch = is_absorb_batch
        self.is_collate_tuple = is_collate_tuple

    def collate(self):
        import re
        if torch.__version__.startswith("1.9") or torch.__version__.startswith("1.10") or torch.__version__.startswith("1.11") or torch.__version__.startswith("1.12"):
            from torch._six import string_classes
            from collections import abc as container_abcs
        else:
            from torch._six import container_abcs, string_classes, int_classes
        from pstar import pdict, plist
        default_collate_err_msg_format = (
            "collate_fn: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        def default_convert(data):
            r"""Converts each NumPy array data field into a tensor"""
            elem_type = type(data)
            if isinstance(data, torch.Tensor):
                return data
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                # array of string classes and object
                if elem_type.__name__ == 'ndarray' \
                        and np_str_obj_array_pattern.search(data.dtype.str) is not None:
                    return data
                return torch.as_tensor(data)
            elif isinstance(data, container_abcs.Mapping):
                return {key: default_convert(data[key]) for key in data}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return elem_type(*(default_convert(d) for d in data))
            elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
                return [default_convert(d) for d in data]
            else:
                return data

        def collate_fn(batch):
            r"""Puts each data field into a tensor with outer dimension batch size, adapted from PyTorch's default_collate."""
            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                if self.is_absorb_batch:
                    tensor = torch.cat(batch, 0, out=out)
                else:
                    tensor = torch.stack(batch, 0, out=out)
                return tensor
            elif elem is None:
                return None
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
                if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                    return collate_fn([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int):
#             elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                Dict = {key: collate_fn([d[key] for d in batch]) for key in elem}
                if isinstance(elem, pdict) or isinstance(elem, Attr_Dict):
                    Dict = elem.__class__(**Dict)
                return Dict
            elif isinstance(elem, My_Freeze_Tuple):
                return batch
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple:
                return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
            elif isinstance(elem, My_Tuple):
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return elem.__class__([collate_fn(samples) for samples in transposed])
            elif isinstance(elem, tuple) and not self.is_collate_tuple:
                return batch
            elif isinstance(elem, container_abcs.Sequence):
                # check to make sure that the elements in batch have consistent size
                it = iter(batch)
                elem_size = len(next(it))
                if not all(len(elem) == elem_size for elem in it):
                    raise RuntimeError('each element in list of batch should be of equal size')
                transposed = zip(*batch)
                return  [collate_fn(samples) for samples in transposed]
            elif elem.__class__.__name__ == 'Dictionary':
                return batch
            elif elem.__class__.__name__ == 'DGLHeteroGraph':
                import dgl
                return dgl.batch(batch)
            raise TypeError(default_collate_err_msg_format.format(elem_type))
        return collate_fn


def sort_two_lists(list1, list2, reverse = False):
    """Sort two lists according to the first list."""
    if reverse:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=operator.itemgetter(0), reverse=True))])
    else:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=operator.itemgetter(0)))])
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]


def try_remove(List, item, is_copy=True):
    """Try to remove an item from the List. If failed, return the original List."""
    if is_copy:
        List = deepcopy(List)
    try:
        List.remove(item)
    except:
        pass
    return List


def print_banner(string, banner_size=100, n_new_lines=0):
    """Pring the string sandwidched by two lines."""
    for i in range(n_new_lines):
        print()
    print("\n" + "=" * banner_size + "\n" + string + "\n" + "=" * banner_size + "\n")


def get_key_of_largest_value(Dict):
    return max(Dict.items(), key=operator.itemgetter(1))[0]


def split_bucket(dictionary, num_common):
    """Split the dictionary into multiple buckets, determined by key[num_common:]."""
    from multiset import Multiset
    keys_common = remove_duplicates([key[:num_common] for key in dictionary.keys()])
    # Find the different keys:
    keys_diff = []
    for key in dictionary.keys():
        if Multiset(keys_common[0]).issubset(Multiset(key)):
            if key[num_common:]  not in keys_diff:
                keys_diff.append(key[num_common:])
    keys_diff = sorted(keys_diff)

    buckets = [OrderedDict() for _ in range(len(keys_diff))]
    for key, item in dictionary.items():
        id = keys_diff.index(key[num_common:])
        buckets[id][key[:num_common]] = item
    return buckets


def switch_dict_keys(Dict, key1, key2):
    inter = Dict[key1]
    Dict[key1] = Dict[key2]
    Dict[key2] = inter
    return Dict


class My_Tuple(tuple):
    def to(self, device):
        self[0].to(device)
        return self
    
    def __getattribute__(self, key):
        if hasattr(My_Tuple, key):
            return object.__getattribute__(self, key)
        else: 
            return self[0].__getattribute__(key)


class My_Freeze_Tuple(tuple):
    pass