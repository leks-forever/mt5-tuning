import random
import re
import sys
import typing as tp
import unicodedata

from sacremoses import MosesPunctNormalizer
from torch.utils.data import Dataset
from transformers import NllbTokenizer


class TextPreprocessor:
    def __init__(self, lang: str = "en", replace_by: str = " "):
        # Инициализация MosesPunctNormalizer
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        # Инициализация функции замены непечатаемых символов
        self.replace_nonprint = self.get_non_printing_char_replacer(replace_by)

    def get_non_printing_char_replacer(self, replace_by: str) -> tp.Callable[[str], str]:
        non_printable_map = {
            ord(c): replace_by
            for c in (chr(i) for i in range(sys.maxunicode + 1))
            # same as \p{C} in perl
            # see https://www.unicode.org/reports/tr44/#General_Category_Values
            if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
        }

        def replace_non_printing_char(line: str) -> str:
            return line.translate(non_printable_map)

        return replace_non_printing_char

    def preprocess(self, text: str) -> str:
        # Шаг 1: Нормализация пунктуации с использованием MosesPunctNormalizer
        clean = self.mpn.normalize(text)
        # Шаг 2: Замена непечатаемых символов
        clean = self.replace_nonprint(clean)
        # Шаг 3: Нормализация текста (например, замена "𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞" на "Francesca")
        clean = unicodedata.normalize("NFKC", clean)
        return clean


class ThisDataset(Dataset):
    def __init__(self, df, random: bool):
        self.df = df
        self.random = random

    def __getitem__(self, idx):
        if self.random:
            item = self.df.iloc[random.randint(0, len(self.df)-1)]  # noqa: S311
        else:
            item = self.df.iloc[idx]

        return item
    
    def __len__(self):
        return len(self.df)

class CollateFn():
    def __init__(self, tokenizer: NllbTokenizer, ignore_index = -100, max_length = 128) -> None:
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.text_preprocessor = TextPreprocessor()

        self.prefixes = {
            "translate Russian to Lezghian: " : {
                "df_src_lang": "ru",
                "df_tgt_lang": "lez",
            },
            "translate Lezghian to Russian: " :{
                "df_src_lang": "lez",
                "df_tgt_lang": "ru",
            }
        }

    def __call__(self, batch: list) -> dict:
        prefix = random.choice(list(self.prefixes.keys())) # Random choice between [ru->lez, lez->ru]  # noqa: S311
        df_src_lang = self.prefixes[prefix]["df_src_lang"]
        df_tgt_lang = self.prefixes[prefix]["df_tgt_lang"]

        return self.pad_batch(batch, prefix, df_src_lang, df_tgt_lang)

    def pad_batch(self, batch: list, prefix: str, df_src_lang: str, df_tgt_lang: str) -> dict:
        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(prefix + self.text_preprocessor.preprocess(item[df_src_lang]))
            y_texts.append(self.text_preprocessor.preprocess(item[df_tgt_lang]))

        # x = self.tokenizer(x_texts, return_tensors='pt', padding='longest')
        x = self.tokenizer(x_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        # y = self.tokenizer(y_texts, return_tensors='pt', padding='longest')
        y = self.tokenizer(y_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = self.ignore_index

        return {
            "x": x,
            "y": y,
        }
    
class LangCollateFn(CollateFn):
    def __init__(self, tokenizer: NllbTokenizer,  prefix, ignore_index = -100, max_length = 128) -> None:
        super().__init__(tokenizer, ignore_index, max_length)

        self.df_src_lang = self.prefixes[prefix]["df_src_lang"]
        self.df_tgt_lang = self.prefixes[prefix]["df_tgt_lang"]
        self.selected_prefix = prefix

    def __call__(self, batch: list) -> dict:
        return self.pad_batch(batch, self.selected_prefix, self.df_src_lang, self.df_tgt_lang)


class TestCollateFn(CollateFn):
    def __init__(self, tokenizer: NllbTokenizer, prefix, a=32, b=3, max_input_length=1024, num_beams=4, max_length = 128):
        super().__init__(tokenizer, max_length)

        self.df_src_lang = self.prefixes[prefix]["df_src_lang"]
        self.df_tgt_lang = self.prefixes[prefix]["df_tgt_lang"]
        self.selected_prefix = prefix
        
        self.prefix = prefix

        self.text_preprocessor = TextPreprocessor()

        self.a = a
        self.b = b
        self.max_input_length = max_input_length
        self.num_beams = num_beams

    def __call__(self, batch: list) -> dict:
        return self.pad_batch(batch)

    def pad_batch(self, batch: list) -> dict:        
        x_texts, y_texts = [], []
        for item in batch:
            x_texts.append(self.selected_prefix +self.text_preprocessor.preprocess(item[self.df_src_lang]))
            y_texts.append(self.text_preprocessor.preprocess(item[self.df_tgt_lang]))

        inputs = self.tokenizer(x_texts, return_tensors='pt', padding='longest')
        # inputs = self.tokenizer(x_texts,  return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        return {
            "x": inputs,
            "max_new_tokens": int(self.a + self.b * inputs.input_ids.shape[1]), # TODO: Think about it
            "num_beams": self.num_beams,
            "tgt_text": y_texts,
        }