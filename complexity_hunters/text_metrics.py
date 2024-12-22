import re


def _get_words_list(text: str) -> list[str]:
    """
    Returns list of words in the text
    """
    return re.findall(r'\b\w+\b', text)


def words_count(texts: list[str]) -> list[int]:
    """
    Returns list of words count in each text
    """
    counts = [
        len(_get_words_list(text))
        for text in texts
    ]
    return counts


def _is_technical_words(words: list[str]) -> list[bool]:
    TECHNICAL_TERMS = {
        "python", "java", "javascript", "api", "html", "css", "sql", "dataframe", "machine learning", "ai",
        "algorithm", "function", "variable", "loop", "object", "class", "framework", "library", "debugging"
    }

    is_technical = [
        word in TECHNICAL_TERMS
        for word in words
    ]
    return is_technical


def _is_dummy_words(words: list[str]) -> list[bool]:
    DUMMY_TERMS = {
        "lorem", "new", "beginer", "logs"
    }

    is_dummy = [
        word in DUMMY_TERMS
        for word in words
    ]


def _count_technical(text: str) -> int:
    words = _get_words_list(text.lower())
    is_technical = _is_technical_words(words)

    return sum(is_technical)


def _count_dummy(text: str) -> int:
    words = _get_words_list(text.lower())
    is_dummy = _is_dummy_words(words)

    return sum(is_dummy)


def tech_words_count(texts: list[str]) -> list[int]:
    """
    Returns list of technical words count in each text
    """
    counts = [
        _count_technical(text)
        for text in texts
    ]
    return counts

def dummy_words_count(texts: list[str]) -> list[int]:
    """
    Returns list of dummy words count in each text
    """
    counts = [
        _count_dummy(text)
        for text in texts
    ]
    return counts


def negative_answers_count(questions, answers, barrier):
    """
    counts answers for the given question with the score less than barrier
    """
    def filter_question_answers(answers, question_id):
        return answers[answers["ParentId"] == question_id]
    
    answers = answers[answers["Score"] < barrier]

    negative_answers_cnt = [
        filter_question_answers(answers, question["Id"]).shape[0]
        for _, question in questions.iterrows()
    ]
    return negative_answers_cnt
