import re

def anticipatory_cost_fn(state):
    """
    Computes cost based on whether objects are in their ideal cabinets:
        - mugs → cabinet2
        - bowls → cabinet1
        - bottles → cabinet3

    If an object is being held (i.e., no 'in(obj, loc)'), we use its original location
    to determine placement correctness.

    Args:
        state (dict): {var_name: predicate_str} from C++ symbolic state

    Returns:
        int: Total placement cost
    """
    total_cost = 0
    placements = {}

    # Pass 1: detect in(...) predicates for object placement
    for _, pred in state.items():
        if pred.startswith("Atom in("):
            match = re.match(r"Atom in\(([^,]+), ([^)]+)\)", pred)
            if match:
                obj, loc = match.groups()
                placements[obj] = loc

    # Handle cost for objects with known placements
    for obj, loc in placements.items():
        if obj.startswith("mug") and loc != "cabinet2":
            total_cost += 1000
        elif obj.startswith("bowl") and loc != "cabinet1":
            total_cost += 1000
        elif obj.startswith("bottle") and loc != "cabinet3":
            total_cost += 1000
  
    return total_cost