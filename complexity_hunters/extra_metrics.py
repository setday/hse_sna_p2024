def sets_iou(set_1: set, set_2: set) -> float:
    """
    Calculates Intersection over Union for two sets
    """
    if len(set_1) == 0 or len(set_2) == 0:
        return 0.0

    intersection = len(set_1 & set_2)
    union = len(set_1 | set_2)
    return intersection / union
