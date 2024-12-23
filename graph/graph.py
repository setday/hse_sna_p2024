import sys

sys.path.append(".")  # to make utils importable
sys.path.append("..")  # to make utils importable

import utils.data_loader
import numpy
import pandas
import bs4
import utils.consts
import networkx
import igraph
import matplotlib.pyplot


COLUMNS_TO_KEEP = [
    "Id",
    "PostTypeId",
    "AcceptedAnswerId",
    "Score",
    "ViewCount",
    "Body",
    "OwnerUserId",
    "Tags",
    "AnswerCount",
    "CommentCount",
    "ParentId",
]

USER_QUESTION_WEIGHT_THRESHOLD = 0.3
USER_USER_WEIGHT_THRESHOLD = 0.7
QUESTION_QUESTION_WEIGHT_THRESHOLD = 0.3


def parse_html(html_row):
    if isinstance(html_row, str):
        return bs4.BeautifulSoup(html_row, "html.parser").get_text(separator=" ")
    return ""


def fix_column_types(posts):
    posts["AcceptedAnswerId"] = posts["AcceptedAnswerId"].fillna(-1.0)
    posts["ViewCount"] = posts["ViewCount"].fillna(0.0)
    posts["OwnerUserId"] = posts["OwnerUserId"].fillna(-1.0)
    posts["AnswerCount"] = posts["AnswerCount"].fillna(0.0)
    posts["ParentId"] = posts["ParentId"].fillna(-1.0)

    posts = posts.astype(
        {
            "AcceptedAnswerId": int,
            "ViewCount": int,
            "OwnerUserId": int,
            "AnswerCount": int,
            "ParentId": int,
        }
    )

    return posts


def create_question_tags(questions):
    question_to_tags = {}
    for i in range(len(questions)):
        question = questions.iloc[i]
        question_to_tags[question.Id] = (
            set(question.Tags.split("|")) if question.Tags is not None else set()
        )
        question_to_tags[question.Id].remove("")
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
            "score": question.Score,
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


def add_user_user_edges(graph, users, user_to_tags):
    for i, user1 in enumerate(users):
        for user2 in users[:i]:
            user1_tags = user_to_tags[user1]
            user2_tags = user_to_tags[user2]

            intersect = user1_tags & user2_tags
            union = user1_tags | user2_tags

            if len(union) < 4:
                continue

            weight = len(intersect) / len(union)
            if weight >= USER_USER_WEIGHT_THRESHOLD:
                graph.add_edge("u" + str(user1), "u" + str(user2))


def add_question_question_edges(graph, questions, question_to_tags):
    for i in range(len(questions)):
        for j in range(i):
            question1 = questions.iloc[i].Id
            question2 = questions.iloc[j].Id
            question1_tags = question_to_tags[question1]
            question2_tags = question_to_tags[question2]

            intersect = question1_tags & question2_tags
            union = question1_tags | question2_tags

            if len(union) < 2:
                continue

            weight = len(intersect) / len(union)
            if weight >= QUESTION_QUESTION_WEIGHT_THRESHOLD:
                graph.add_edge("q" + str(question1), "q" + str(question2))


def build_partition(graph):
    i_graph = igraph.Graph.from_networkx(graph)
    partition = i_graph.community_leiden(
        objective_function="modularity", n_iterations=1000
    )

    node_to_community = {
        node: partition.membership[i] for i, node in enumerate(i_graph.vs["_nx_name"])
    }
    communities = [set() for _ in range(max(partition.membership) + 1)]
    for node in i_graph.vs["_nx_name"]:
        communities[node_to_community[node]].add(node)

    return node_to_community, communities


def build_graph():
    posts = utils.data_loader.load_dataset("data/Posts.xml", full=True)
    posts = pandas.concat(
        [
            posts[posts.PostTypeId == 1].sample(n=500).reset_index(drop=True),
            posts[posts.PostTypeId == 2].sample(n=10000).reset_index(drop=True),
        ]
    )
    badges = utils.data_loader.load_dataset("data/Badges.xml", full=True)

    posts = posts[COLUMNS_TO_KEEP]
    posts["Body"] = posts["Body"].apply(parse_html)

    posts = fix_column_types(posts)

    users = numpy.sort(posts.OwnerUserId.unique())[1:500]  # remove NaN
    questions = posts[posts.PostTypeId == 1]
    answers = posts[posts.PostTypeId == 2]

    graph = networkx.Graph()

    user_to_tags = {user: set(badges[badges.UserId == user].Name) for user in users}
    question_to_tags = create_question_tags(questions)

    apply_user_attributes(graph, users, user_to_tags)
    apply_question_attributes(graph, questions, question_to_tags)

    add_user_question_edges(graph, answers, questions, users)
    add_user_user_edges(graph, users, user_to_tags)
    add_question_question_edges(graph, questions, question_to_tags)

    users_to_community, user_communities = build_partition(
        graph.subgraph([node for node in graph.nodes if graph.nodes[node]["type"] == "user"])
    )
    print(user_communities)
    
    user_stereotypes = []
    for community in user_communities:
        tags = set()
        for user in community:
            tags |= set(graph.nodes[user]["tags"])

        rates = {tag: 0 for tag in tags}
        for user in community:
            for tag in graph.nodes[user]["tags"]:
                rates[tag] += 1

        for key in rates.keys():
            rates[key] /= len(community)

        user_stereotypes.append(rates)


    question_to_community, question_communities = build_partition(
        graph.subgraph([node for node in graph.nodes if graph.nodes[node]["type"] == "question"])
    )

    question_stereotypes = []
    for community in question_communities:
        tags = set()
        for question in community:
            tags |= set(graph.nodes[question]["tags"])

        rates = {tag: 0 for tag in tags}
        for question in community:
            for tag in graph.nodes[question]["tags"]:
                rates[tag] += 1

        for key in rates.keys():
            rates[key] /= len(community)

        question_stereotypes.append(rates)


    pos = {}
    for i, user in enumerate(users):
        x = i / len(users)
        pos["u" + str(user)] = (x * (1 - x), x)

    for i in range(len(questions)):
        question = questions.iloc[i]
        x = i / len(questions)
        pos["q" + str(int(question.Id))] = (1 - x * (1 - x), x)

    networkx.draw(graph, pos, node_size=20, width=1, alpha=0.1)
    matplotlib.pyplot.show()


build_graph()
