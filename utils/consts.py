COLS_TO_KEEP = ["Id", "PostTypeId", "AcceptedAnswerId", "CreationDate", "Score", "ViewCount", "Body", "OwnerUserId",
                "LastEditorUserId", "Tags", "CommentCount", "ClosedDate", "FavoriteCount"]

EMBEDDERS = {
    'Albert': 'paraphrase-albert-small-v2',
    'Roberta': 'all-distilroberta-v1',
    'DistilBert': 'multi-qa-distilbert-cos-v1',
    'MiniLM1': 'all-MiniLM-L6-v2',
    'MiniLM2': 'all-MiniLM-L12-v2',
    'MiniLM3': 'paraphrase-MiniLM-L3-v2'
}
