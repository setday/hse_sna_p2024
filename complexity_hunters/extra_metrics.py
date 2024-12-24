def sets_iou(set_1: set, set_2: set, min_union_size: int = 1) -> float:
    """
    Calculates Intersection over Union for two sets

    :param set_1: first set
    :param set_2: second set
    :param min_union_size: minimal size of the union to avoid low confidence
    """

    if len(set_1) == 0 or len(set_2) == 0:
        return 0.0

    intersection = len(set_1 & set_2)
    union = len(set_1 | set_2)

    if union < min_union_size:
        return 0.0

    return intersection / union
