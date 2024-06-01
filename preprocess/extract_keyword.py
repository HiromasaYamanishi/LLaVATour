import pandas as pd
import re
import string
from statistics import StatisticsError
from tqdm import tqdm
import neologdn
from rake_ja import Tokenizer, JapaneseRake


class KeywordExtractor:
    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:

        self.data = data
        self.data = self.data.fillna("")

    # 前処理
    def _preprocess(self, x: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        print(x)
        x = emoji_pattern.sub(r"", x)

        x = neologdn.normalize(x)
        x = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", x)
        x = re.sub(r"[!-/:-@[-`{-~]", r" ", x)
        x = re.sub("[■-♯]", " ", x)
        x = re.sub(r"(\d)([,.])(\d+)", "\1\3", x)
        x = re.sub(r"\d+", "0", x)
        x = re.sub(r"・", ", ", x)
        x = re.sub(r"[\(\)「」【】]", "", x)

        return x

    def extract_phrases(self, data: pd.Series) -> tuple[list[float], list[str]]:
        raise NotImplementedError

    def apply_keywords_extract(self) -> pd.DataFrame:
        tqdm.pandas()
        self.data[["scores", "keywords"]] = self.data.progress_apply(
            self.extract_phrases, axis=1, result_type="expand"
        )

        return self.data

class Rake(KeywordExtractor):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

        self.tokenizer = Tokenizer()
        self.punctuations = string.punctuation + ",.。、"
        self.stopwords = (
            "か な において にとって について する これら から と も が は て で に を は し た の ない よう いる という".split()
            + "により 以外 それほど ある 未だ さ れ および として といった られ この ため こ たち ・ ご覧".split()
        )
        self.rake = JapaneseRake(
            max_length=3,
            punctuations=self.punctuations,
            stopwords=self.stopwords,
        )

    def extract_phrases(self, data: pd.Series) -> tuple[list[float], list[str]]:
        tokens = self.tokenizer.tokenize(self._preprocess(data["text"]))

        self.rake.extract_keywords_from_text(tokens)
        scrs_kwds = self.rake.get_ranked_phrases_with_scores()

        if len(scrs_kwds) > 0:
            return [x[0] for x in scrs_kwds], [x[1] for x in scrs_kwds]
        else:
            return [], []