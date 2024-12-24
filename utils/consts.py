# Post dataset constants
POST_ESSENTIAL_COLUMNS = [
    "Id",
    "PostTypeId",
    "AcceptedAnswerId",
    # "CreationDate",
    "Score",
    "ViewCount",
    "Body",
    "OwnerUserId",
    # "LastEditorUserId",
    "Tags",
    "AnswerCount",
    "CommentCount",
    # "ClosedDate",
    # "FavoriteCount",
    "ParentId",
]

# Embedder models
EMBEDDERS = {
    "Albert": "paraphrase-albert-small-v2",
    "Roberta": "all-distilroberta-v1",
    "DistilBert": "multi-qa-distilbert-cos-v1",
    "MiniLM1": "all-MiniLM-L6-v2",
    "MiniLM2": "all-MiniLM-L12-v2",
    "MiniLM3": "paraphrase-MiniLM-L3-v2",
}

# Dataset paths
POSTS_DATA_PATH = "../data/Posts.xml"
USERS_DATA_PATH = "../data/Users.xml"
BADGES_DATA_PATH = "../data/Badges.xml"
