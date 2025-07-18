# import re

# def anticipatory_cost_fn(state):
#     """
#     Computes cost based on whether objects are in their ideal cabinets:
#         - mugs → cabinet2
#         - bowls → cabinet1
#         - bottles → cabinet3

#     If an object is being held (i.e., no 'in(obj, loc)'), we use its original location
#     to determine placement correctness.

#     Args:
#         state (dict): {var_name: predicate_str} from C++ symbolic state

#     Returns:
#         int: Total placement cost
#     """
#     total_cost = 0
#     placements = {}

#     # Pass 1: detect in(...) predicates for object placement
#     for _, pred in state.items():
#         if pred.startswith("Atom in("):
#             match = re.match(r"Atom in\(([^,]+), ([^)]+)\)", pred)
#             if match:
#                 obj, loc = match.groups()
#                 placements[obj] = loc

#     # Handle cost for objects with known placements
#     for obj, loc in placements.items():
#         if obj.startswith("mug") and loc != "cabinet2":
#             total_cost += 1000
#         elif obj.startswith("bowl") and loc != "cabinet1":
#             total_cost += 1000
#         elif obj.startswith("bottle") and loc != "cabinet3":
#             total_cost += 1000
  
#     return total_cost
import os
import random
import math
import torch
import matplotlib.pyplot as plt

from graph_data_generator import (
    get_nodes_from_curr_state,
    get_edges,
    graph_format,
    return_graph
)

import prepare_env

def anticipatory_cost_fn(state):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    eval_net = prepare_env.PrepareEnvGNN.get_net_eval_fn(
        network_file="PrepareEnv.pt", device=device
    )
    robot_confs = {'r0': (150, 200)}
    placements = {}
    for _, pred in state.items():
    if pred.startswith("Atom in("):
        match = re.match(r"Atom in\(([^,]+), ([^)]+)\)", pred)
        if match:
            obj, loc = match.groups()
            placements[obj] = loc
    
    # need to find a way to get object state and locations here..
    nodes_arr = get_nodes_from_curr_state(object_state, locations, robot_confs["r0"])
    node_names = []
    node_vals = []
    for node in nodes_arr:
        if {} in list(node.values()):
            continue
        node_vals.extend(list(node.values()))
        node_names.extend(list(node.keys()))

    node_names = {i: node_names[i] for i in range(len(node_names))}
    color_map = [feats["color"] for feats in node_vals]
    edge_index = get_edges(node_vals)
    nodes = graph_format(nodes_arr)
    graph_img = return_graph(nodes, edge_index, node_names, color_map)

    datum = {
        "graph_nodes": torch.tensor(nodes, dtype=torch.float),
        "graph_edge_index": torch.tensor(edge_index, dtype=torch.long),
        "graph_image": graph_img,
        "expected_cost": state_expected_cost,
    }

    return eval_net(datum)

