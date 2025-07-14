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
    held_objects = set()

    # Define fallback/original object locations (assumed known)
    original_locations = {
        "mug0": "table1",
        "bottle0": "table2",
        "bowl0": "cabinet1",
        # Add more if needed
    }

    # Pass 1: detect in(...) predicates for object placement
    for _, pred in state.items():
        if pred.startswith("Atom in("):
            match = re.match(r"Atom in\(([^,]+), ([^)]+)\)", pred)
            if match:
                obj, loc = match.groups()
                placements[obj] = loc

    # Pass 2: detect holding(...) predicates
    for _, pred in state.items():
        if pred.startswith("Atom holding("):
            match = re.match(r"Atom holding\([^,]+, ([^)]+)\)", pred)
            if match:
                held_objects.add(match.group(1))

    # Handle cost for objects with known placements
    for obj, loc in placements.items():
        if obj in held_objects:
            # Held object — use fallback/original location
            loc = original_locations.get(obj, None)
            if not loc:
                continue  # unknown object, skip
        if obj.startswith("mug") and loc != "cabinet2":
            total_cost += 100
        elif obj.startswith("bowl") and loc != "cabinet1":
            total_cost += 100
        elif obj.startswith("bottle") and loc != "cabinet3":
            total_cost += 100

    # Handle held objects that are not in placements
    for obj in held_objects:
        if obj not in placements:
            loc = original_locations.get(obj, None)
            if not loc:
                continue  # unknown, skip
            if obj.startswith("mug") and loc != "cabinet2":
                total_cost += 100
            elif obj.startswith("bowl") and loc != "cabinet1":
                total_cost += 100
            elif obj.startswith("bottle") and loc != "cabinet3":
                total_cost += 100

    return total_cost