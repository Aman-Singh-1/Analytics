"""Turn the raw Stack Overflow dump into a clean Q/A corpus.

The raw dataset has no accepted-answer flag we can trust, so we take each
question's highest-scored answer as a stand-in. We keep code blocks intact
(as fenced markdown) because this is a Python assistant and flattening code
into prose destroys most of its value.
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/corpus.parquet")

# defaults tuned for a take-home: small enough to build locally in minutes,
# big enough to cover the common questions. see README for the reasoning.
MIN_QUESTION_SCORE = 5
MAX_PAIRS = 50_000
MAX_QUESTION_CHARS = 1500


def html_to_markdown(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "lxml")

    for pre in soup.find_all("pre"):
        code = pre.get_text().strip("\n")
        pre.replace_with(f"\n```\n{code}\n```\n")

    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    text = soup.get_text()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def best_answers(answers: pd.DataFrame) -> pd.DataFrame:
    ranked = answers.sort_values("Score", kind="stable")
    return ranked.drop_duplicates("ParentId", keep="last")


def build_document(title: str, question: str, answer: str, answer_score: int) -> str:
    q = question if len(question) <= MAX_QUESTION_CHARS else question[:MAX_QUESTION_CHARS] + " ..."
    return (
        f"# {title}\n\n"
        f"{q}\n\n"
        f"## Top answer (score {answer_score})\n\n"
        f"{answer}"
    )


def build_corpus(
    raw_dir: Path = RAW_DIR,
    out_path: Path = OUT_PATH,
    min_score: int = MIN_QUESTION_SCORE,
    max_pairs: int = MAX_PAIRS,
) -> pd.DataFrame:
    questions = pd.read_csv(
        raw_dir / "Questions.csv",
        encoding="latin-1",
        usecols=["Id", "Score", "Title", "Body"],
    )
    answers = pd.read_csv(
        raw_dir / "Answers.csv",
        encoding="latin-1",
        usecols=["ParentId", "Score", "Body"],
    )

    questions = questions[questions["Score"] >= min_score]
    questions = questions.nlargest(max_pairs, "Score")

    top = best_answers(answers).rename(
        columns={"Score": "AnswerScore", "Body": "AnswerBody"}
    )
    paired = questions.merge(top, left_on="Id", right_on="ParentId", how="inner")
    paired = paired[paired["AnswerBody"].str.strip().astype(bool)]

    rows = []
    for r in paired.itertuples(index=False):
        question_md = html_to_markdown(r.Body)
        answer_md = html_to_markdown(r.AnswerBody)
        if not answer_md:
            continue
        rows.append(
            {
                "question_id": int(r.Id),
                "title": r.Title,
                "score": int(r.Score),
                "answer_score": int(r.AnswerScore),
                "url": f"https://stackoverflow.com/questions/{int(r.Id)}",
                "text": build_document(r.Title, question_md, answer_md, int(r.AnswerScore)),
            }
        )

    corpus = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_path, index=False)
    return corpus


if __name__ == "__main__":
    df = build_corpus()
    print(f"wrote {len(df):,} Q/A pairs to {OUT_PATH}")
    print(f"question score: min {df['score'].min()} median {int(df['score'].median())} max {df['score'].max()}")
