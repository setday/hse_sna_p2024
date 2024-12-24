import bs4
import pandas as pd


def load_dataset(filepath: str = "../data/Posts.xml", debug_slice: bool = True) -> pd.DataFrame:
    """
    Load the dataset xml file from the given filepath

    :param filepath: The path to the xml file
    :param debug_slice: Whether to load only a slice of the data (only 10000 rows)
    """

    print(f"INFO: Loading dataset {filepath}...")

    posts = pd.read_xml(filepath, parser="etree")
    
    if debug_slice:
        posts = posts[:10000]

    return posts


def html_to_str(html_row: str) -> str:
    """
    Convert the html row to a string

    :param html_row: The html row
    """

    if not isinstance(html_row, str):
        return ""

    soup = bs4.BeautifulSoup(html_row, "html.parser")
    return soup.get_text(separator=" ")


def posts_fill_na(posts: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the NaN values in the posts dataframe

    :param posts: The posts dataframe
    """
    
    posts["AcceptedAnswerId"] = posts["AcceptedAnswerId"].fillna(-1.0).astype(int)
    posts["ViewCount"]        = posts["ViewCount"]       .fillna( 0.0).astype(int)
    posts["OwnerUserId"]      = posts["OwnerUserId"]     .fillna(-1.0).astype(int)
    posts["AnswerCount"]      = posts["AnswerCount"]     .fillna( 0.0).astype(int)
    posts["ParentId"]         = posts["ParentId"]        .fillna(-1.0).astype(int)


    return posts


def question_answer_split(posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the posts dataframe into questions and answers

    :param posts: The posts dataframe
    """

    questions = posts[posts.PostTypeId == 1]
    answers = posts[posts.PostTypeId == 2]

    return questions, answers


def extract_tags_from_str(tag_series: str) -> set[str]:
    """
    Extract tags from the string

    :param tags_str: The string of tags
    """

    tags = [
        set(str(str_of_tags)[1:-1].split("|"))
        if str(str_of_tags) != "nan" and len(str(str_of_tags)) > 2
        else set()
        for str_of_tags in tag_series
    ]

    return tags
