import sys

import numpy
import pandas
import networkx
import igraph
import matplotlib.pyplot
import pickle

sys.path.append("../complexity_hunters/")  # to make utils importable
sys.path.append(".")  # to make utils importable
sys.path.append("..")  # to make utils importable

import utils.data_worker
import utils.consts

from complexity_hunters.extra_metrics import sets_iou
from user_likelihood_metrics import sparse_user_tags_likelihood


USER_QUESTION_WEIGHT_THRESHOLD = 0.3
USER_USER_WEIGHT_THRESHOLD = 0.7
QUESTION_QUESTION_WEIGHT_THRESHOLD = 0.3


def create_question_tags(questions: pandas.DataFrame) -> dict[int, set[str]]:
    tags_list = utils.data_worker.extract_tags_from_str(questions["Tags"])
    question_ids = questions["Id"]
    
    question_to_tags = {
        idx: tags
        for idx, tags in zip(question_ids, tags_list)
    }
    
    return question_to_tags


def apply_user_attributes(graph, users, user_to_tags):
    user_attributes = {}
    for user in users:
        graph.add_node("u" + str(user))
        user_attributes["u" + str(user)] = {"type": "user", "tags": user_to_tags[user]}
    networkx.set_node_attributes(graph, user_attributes)


def apply_question_attributes(graph, questions, question_to_tags):
    question_attributes = {}
    for i in range(len(questions)):
        question = questions.iloc[i]
        graph.add_node("q" + str(question.Id))
        question_attributes["q" + str(question.Id)] = {
            "type": "question",
            "tags": question_to_tags[question.Id],
            # "score": question.Score,
        }
    networkx.set_node_attributes(graph, question_attributes)


def add_user_question_edges(graph, answers, questions, users):
    min_answer_score_for_question = {}
    max_answer_score_for_question = {}

    for i in range(len(answers)):
        answer = answers.iloc[i]
        question_id = answer.ParentId
        if (
            len(questions[questions.Id == question_id]) > 0
            and answer.OwnerUserId in users
        ):
            if question_id not in min_answer_score_for_question:
                min_answer_score_for_question[question_id] = answer.Score
            else:
                min_answer_score_for_question[question_id] = min(
                    min_answer_score_for_question[question_id],
                    answer.Score,
                )

            if question_id not in max_answer_score_for_question:
                max_answer_score_for_question[question_id] = answer.Score
            else:
                max_answer_score_for_question[question_id] = max(
                    max_answer_score_for_question[question_id],
                    answer.Score,
                )

    for i in range(len(answers)):
        answer = answers.iloc[i]
        question_id = answer.ParentId
        if (
            len(questions[questions.Id == question_id]) > 0
            and answer.OwnerUserId in users
        ):
            weight = (answer.Score + 1 - min_answer_score_for_question[question_id]) / (
                max_answer_score_for_question[question_id]
                + 1
                - min_answer_score_for_question[question_id]
            )

            if weight > USER_QUESTION_WEIGHT_THRESHOLD:
                graph.add_edge(
                    "u" + str(answer.OwnerUserId),
                    "q" + str(question_id),
                )


def add_user_user_edges(graph, user_badges):
    user_pairs = sparse_user_tags_likelihood(
        user_badges.Name,
        user_badges.UserId,
        barrier=USER_USER_WEIGHT_THRESHOLD
    )
    for user1, user2, weight in user_pairs:
        graph.add_edge("u" + str(user1), "u" + str(user2))


def add_question_question_edges(graph, questions, question_to_tags):
    for i in range(len(questions)):
        for j in range(i):
            question1 = questions.iloc[i].Id
            question2 = questions.iloc[j].Id

            weight = sets_iou(
                question_to_tags[question1],
                question_to_tags[question2],
                min_union_size=2
            )
            if weight >= QUESTION_QUESTION_WEIGHT_THRESHOLD:
                graph.add_edge("q" + str(question1), "q" + str(question2))



def add_undefined_atributes(graph):
    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            graph.nodes[node]["type"] = "user" if node[0] == "u" else "question"
        if "tags" not in graph.nodes[node]:
            graph.nodes[node]["tags"] = set()
        # if "score" not in graph.nodes[node] and graph.nodes[node]["type"] == "question":
        #    graph.nodes[node]["score"] = 0


def build_graph(show=False):
    posts = utils.data_worker.load_dataset(utils.consts.POSTS_DATA_PATH, debug_slice=True)
    badges = utils.data_worker.load_dataset(utils.consts.BADGES_DATA_PATH, debug_slice=False)
    badges = badges[badges.UserId.isin(posts.OwnerUserId.unique())]

    posts = posts[utils.consts.POST_ESSENTIAL_COLUMNS]
    posts["Body"] = posts["Body"].apply(utils.data_worker.html_to_str)

    posts = utils.data_worker.posts_fill_na(posts)

    users = numpy.sort(posts.OwnerUserId.unique())[1:]  # remove NaN
    questions, answers = utils.data_worker.question_answer_split(posts)

    graph = networkx.Graph()

    user_to_tags = {user: set(badges[badges.UserId == user].Name) for user in users}
    question_to_tags = create_question_tags(questions)

    apply_user_attributes(graph, users, user_to_tags)
    apply_question_attributes(graph, questions, question_to_tags)

    add_user_question_edges(graph, answers, questions, users)
    add_user_user_edges(graph, badges)
    add_question_question_edges(graph, questions, question_to_tags)

    add_undefined_atributes(graph)

    if show:
        pos = {}
        for i, user in enumerate(user_nodes):
            x = i / len(user_nodes)
            pos[user] = (x * (1 - x), x)

        for i, question in enumerate(question_nodes):
            x = i / len(question_nodes)
            pos[question] = (1 - x * (1 - x), x)

        networkx.draw(graph, pos, node_size=20, width=1, alpha=0.1)
        matplotlib.pyplot.show()

    pickle.dump(graph, open("../data/graph.pkl", "wb"))
    print("INFO: Dumped graph into ./data/graph.pkl")

