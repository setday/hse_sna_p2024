import re


def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def count_tech_words(text):
    technical_terms = {
        "python", "java", "javascript", "api", "html", "css", "sql", "dataframe", "machine learning", "ai",
        "algorithm", "function", "variable", "loop", "object", "class", "framework", "library", "debugging"
    }

    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    count = sum(1 for word in words if word in technical_terms)
    return count


def count_negative_answers(answers, barrier):
    """
    counts answers for the given question with the score less than barrier
    """
    negative_answers = answers[answers["Score"] < barrier]
    negative_answers_cnt = negative_answers.shape[0]
    return negative_answers_cnt


def markup(questions, answers, barrier=0):
    """
    adds column of negative answers count
    """
    questions = questions.assign(negative_answers=0)

    for id, question in questions.iterrows():
        question_id = question["Id"]
        related_answers = answers[answers["ParentId"] == question_id]
        negative_answers_cnt = count_negative_answers(related_answers, barrier)
        questions.at[id, "negative_answers"] = negative_answers_cnt

    return questions
