import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from antplan.utilities.primitives import (
    LOCATION_WIDTH,
    LOCATION_BUFFER,
    COLORS,
    get_object_interval,
    interval_contains
)

from antplan.utilities.utils import LOCATION_COLORS
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
from PIL import Image

# Global variable for node indexing
curr_node_index = 0

# --------------------- Utility Functions --------------------- #

def get_object_type_encoding(obj_type):
    """
    One-hot encode an object type: location, object, or robot.

    Args:
        obj_type (str): One of {"location", "object", "robot"}.

    Returns:
        list: One-hot encoded vector [location, object, robot].
    """
    encoding = [0, 0, 0]
    valid_types = ["location", "object", "robot"]

    if obj_type not in valid_types:
        raise ValueError(f"Invalid object type: {obj_type}. Must be one of {valid_types}")

    index = valid_types.index(obj_type)
    encoding[index] = 1
    return encoding


def get_room_interval(location_center, extent):
    """
    Compute interval bounds for a location.

    Args:
        location_center (list): [x, y] center of location.
        extent (ndarray): Extent array.

    Returns:
        list: [[lower_x, lower_y], [upper_x, upper_y]]
    """
    lower = location_center - extent
    upper = location_center + extent
    return [lower[0].tolist(), upper[0].tolist()]


def is_object_in_location(location_pose, object_pose):
    """
    Check if an object is inside a location's bounds.

    Args:
        location_pose (list): Location center [x, y].
        object_pose (list): Object pose [x, y].

    Returns:
        bool: True if object is inside location.
    """
    object_pose_scaled = [object_pose[2] * LOCATION_WIDTH, object_pose[3] * LOCATION_WIDTH]
    location_center_scaled = [location_pose[0] * LOCATION_WIDTH, location_pose[1] * LOCATION_WIDTH]

    object_extent = get_object_interval("object", object_pose_scaled)
    location_extent = get_room_interval(np.array(location_center_scaled), np.array([(LOCATION_BUFFER, LOCATION_BUFFER)]))

    return interval_contains(location_extent, object_extent)


def get_location_center(location_dims):
    """Compute center of a location from its dimensions."""
    return (
        (location_dims[0][0] + location_dims[1][0]) / 2,
        (location_dims[0][1] + location_dims[1][1]) / 2
    )

# --------------------- Graph Construction --------------------- #

def get_edges(nodes):
    """
    Build graph edges based on node types and spatial relations.

    Args:
        nodes (list): List of node dictionaries.

    Returns:
        list: Edge index [[source_ids], [target_ids]].
    """
    edges_prop = []

    for node in nodes:
        for node2 in nodes:
            if node == node2:
                continue

            # Location-Location edges
            if node["type"] == [1, 0, 0] and node2["type"] == [1, 0, 0]:
                edges_prop.append({"index": [node["id"], node2["id"]]})

            # Location-Object edges (if object is inside location)
            elif node["type"] == [1, 0, 0] and node2["type"] == [0, 1, 0]:
                if is_object_in_location(node["position"], node2["position"]):
                    edges_prop.append({"index": [node["id"], node2["id"]]})

        # Objects not in any location → connect to root (0)
        if node["type"] == [0, 1, 0]:
            location_nodes = [rnode for rnode in nodes if rnode["type"] == [1, 0, 0]]
            inside_any = any(is_object_in_location(loc["position"], node["position"]) for loc in location_nodes)
            if not inside_any:
                edges_prop.append({"index": [0, node["id"]]})

        # Robot always connects to root
        if node["type"] == [0, 0, 1]:
            edges_prop.append({"index": [0, node["id"]]})

    # Format edge index
    edges = [edge["index"] for edge in edges_prop]
    return [[e[0] for e in edges], [e[1] for e in edges]]

def get_object_nodes(objects):
    """Create object nodes from object states."""
    global curr_node_index
    nodes = []
    colors = dict(zip(sorted(objects.keys()), COLORS))
    for idx, (name, pose) in enumerate(objects.items()):
        curr_node_index += 1
        node = {
            "id": curr_node_index,
            "type": get_object_type_encoding("object"),
            "color": colors[name],
            "position": [0, 0, pose[0] / LOCATION_WIDTH, pose[1] / LOCATION_WIDTH, 0, 0],
            "expected_cost": [0]
        }
        nodes.append({name: node})

    return nodes

def get_location_nodes(locations):
    """Create location nodes from environment locations."""
    global curr_node_index
    nodes = []

    for idx, (name, dims) in enumerate(locations.items()):
        curr_node_index = idx
        location_center = get_location_center(dims)
        node = {
            "id": curr_node_index,
            "type": get_object_type_encoding("location"),
            "color": LOCATION_COLORS[name],
            "position": [location_center[0] / LOCATION_WIDTH, location_center[1] / LOCATION_WIDTH, 0, 0, 0, 0],
            "expected_cost": [0]
        }
        nodes.append({name: node})

    return nodes


def get_robot_node(robot_conf):
    """Create a single robot node."""
    global curr_node_index
    node = {
        "id": curr_node_index + 1,
        "type": get_object_type_encoding("robot"),
        "color": "yellow",
        "position": [0, 0, 0, 0, robot_conf[0] / LOCATION_WIDTH, robot_conf[1] / LOCATION_WIDTH],
        "expected_cost": [0]
    }
    return [{"robot": node}]


def get_nodes_from_curr_state(objects, locations, robot_conf):
    """Combine all nodes from the current state."""
    return get_location_nodes(locations) + get_object_nodes(objects) + get_robot_node(robot_conf)


def graph_format(nodes):
    """
    Convert node dictionaries to a numerical feature array for graph representation.

    Args:
        nodes (list): List of node dictionaries.

    Returns:
        list: 2D array of node features.
    """
    node_vals = [val for node in nodes for val in node.values() if val]

    graph_array = []
    for node in node_vals:
        color_rgba = list(colors.to_rgba(node["color"]))
        node["color"] = color_rgba
        values = list(node.values())

        # Remove id and color for feature vector
        feature_values = values[1:-1]
        flattened_features = [item for sublist in feature_values for item in (sublist if isinstance(sublist, list) else [sublist])]
        graph_array.append(flattened_features)

    return graph_array


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def return_graph(nodes, edge_index, node_names, color_map):
    plt.close()
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    G = to_networkx(data, to_undirected=True)
    nx.draw(G, labels=node_names, node_color=color_map)
    fig = plt.gcf()
    img = fig2img(fig)
    return img


def images_together(graph_image, objectworld_image, expected_cost):
    """
    Display two images side by side with a super title showing the expected cost.

    Args:
        graph_image (ndarray): The image representing the graph.
        objectworld_image (ndarray): The image representing the environment state.
        expected_cost (float): The expected cost to display in the super title.
    """
    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Super title with expected cost
    fig.suptitle(f"Expected Cost: {expected_cost:.2f}", fontsize=14, fontweight='bold')

    # Left subplot - Environment state
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(objectworld_image)
    ax1.axis("off")
    ax1.set_title("A State in the Environment")

    # Right subplot - Graph representation
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(graph_image)
    ax2.axis("off")
    ax2.set_title("Corresponding Graph")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    # plt.show()