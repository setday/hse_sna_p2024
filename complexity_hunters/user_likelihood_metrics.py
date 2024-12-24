import numpy as np
from tqdm import tqdm

from extra_metrics import sets_iou


def _make_user_to_tags(user_tags, user_ids):
    """
    Make a dictionary with user ids as keys and tags as values.

    :param user_tags: list of tags for each user
    :param user_ids: list of user ids
    """

    unique_users = set(user_ids)
    
    user_to_tags = {user: set() for user in unique_users}
    for idx, tag in zip(user_ids, user_tags):
        if np.isnan(idx) or not tag:
            continue

        user_to_tags[idx].add(tag)

    return user_to_tags


def sparse_user_tags_likelihood(user_tags, user_ids, barrier=0.8):
    """
    Calculate the likelihood of a user given the tags he/she has.

    :param user_tags: list of tags for each user
    :param user_ids: list of user ids
    :param barrier: the threshold for the iou
    """

    unique_users = set(user_ids)
    
    user_to_tags = _make_user_to_tags(user_tags, user_ids)

    # excluding users with no tags
    users = [
        user
        for user in unique_users
        if not np.isnan(user) and user_to_tags[user]
    ]
    users_cnt = len(users)

    result = []

    for user1_idx in tqdm(range(users_cnt), desc="extracting user pairs based on tags"):
        for user2_idx in range(user1_idx + 1, users_cnt):
            user1 = users[user1_idx]
            user2 = users[user2_idx]

            w = sets_iou(user_to_tags[user1], user_to_tags[user2])

            if w >= barrier:
                result.append((user1, user2, w))

    return result


def _make_user_to_answers(user_answers, user_ids):
    """
    Make a dictionary with user ids as keys and answers as values.

    :param user_answers: list of answers for each user
    :param user_ids: list of user ids
    """

    unique_users = set(user_ids)
    
    user_to_answers = {user: set() for user in unique_users}
    for idx, answer in zip(user_ids, user_answers):
        if np.isnan(idx) or not answer:
            continue

        user_to_answers[idx].add(answer)

    return user_to_answers
    

def sparse_user_answers_likelihood(user_answers, user_ids, barrier=0.8):
    """
    Calculate the likelihood of a user given the answers he/she has.

    :param user_answers: list of answers for each user
    :param user_ids: list of user ids
    :param barrier: the threshold for the iou
    """

    unique_users = set(user_ids)
    
    user_to_answers = _make_user_to_answers(user_answers, user_ids)

    # excluding users with no answers
    users = [
        user
        for user in unique_users
        if not np.isnan(user) and user_to_answers[user]
    ]
    users_cnt = len(users)

    result = []

    for user1_idx in tqdm(range(users_cnt), desc="extracting user pairs based on answers"):
        for user2_idx in range(user1_idx + 1, users_cnt):
            user1 = users[user1_idx]
            user2 = users[user2_idx]

            w = sets_iou(user_to_answers[user1], user_to_answers[user2])

            if w >= barrier:
                result.append((user1, user2, w))

    return result
