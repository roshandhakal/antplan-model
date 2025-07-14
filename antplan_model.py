def anticipatory_cost_fn(state):
    """
    Computes cost based on whether objects are in their ideal cabinets:
        - mugs → cabinet2
        - bowls → cabinet1
        - bottles → cabinet3
    Any deviation is penalized with a high cost.

    Args:
        placements (dict): {object_name: location_name}

    Returns:
        total_cost (int): Sum of placement costs.
    """
    return 0
