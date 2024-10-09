from dataclasses import dataclass
from typing import Iterable, Optional, Union
import json
import csv


# used only during conversion from CFG output into SentenceData
@dataclass
class TokenData:
    text: str

    def __str__(self):
        return f"text: {self.text}, crit: {self.critical_region}, surpr: {self.surprisal}"

    def __repr__(self):
        return str(self) + "\n"

    def to_dict(self):
        return {
            'text': self.text,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class SentenceData:
    original_tokens: tuple[str, ...]
    # processed_tokens: tuple[TokenData, ...]
    processed_tokens: tuple[str]
    grammatical: bool
    # critical_token: TokenData
    critical_tokens: list[str]
    # assigned when get_surprisal is calculated (& assign = True)
    critical_surprisal: float = None

    def __str__(self) -> str:
        # return " ".join(x.text for x in self.processed_tokens)
        return " ".join(self.processed_tokens)

    def __repr__(self) -> str:
        return str(self)

    def to_dict(self):
        return {
            'original_tokens': self.original_tokens,
            # 'processed_tokens': [token.to_dict() for token in self.processed_tokens],
            'processed_tokens': self.processed_tokens,
            'grammatical': self.grammatical,
            'critical_tokens': self.critical_tokens,
            'critical_surprisal': self.critical_surprisal,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            original_tokens=data["original_tokens"],
            # processed_tokens=[TokenData.from_dict(
            #     token_data) for token_data in data["processed_tokens"]],
            processed_tokens=data["processed_tokens"],
            grammatical=data["grammatical"],
            # critical_token=TokenData.from_dict(data["critical_token"])
            critical_tokens=data["critical_tokens"],
            critical_surprisal=data["critical_surprisal"],
        )


@dataclass
class TupleSentenceData:
    s_ab: SentenceData = None
    s_xb: SentenceData = None
    s_ax: SentenceData = None
    s_xx: SentenceData = None

    def __str__(self) -> str:
        return f"S_AB: {str(self.s_ab)}, S_XB: {str(self.s_xb)}, S_AX: {str(self.s_ax)}, S_XX: {str(self.s_xx)}"

    def __repr__(self) -> str:
        return str(self)

    def to_dict(self):
        return {
            's_ab': self.s_ab.to_dict() if self.s_ab else None,
            's_xb': self.s_xb.to_dict() if self.s_xb else None,
            's_ax': self.s_ax.to_dict() if self.s_ax else None,
            's_xx': self.s_xx.to_dict() if self.s_xx else None
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            s_ab=SentenceData.from_dict(
                data.get("s_ab")) if data.get("s_ab") else None,
            s_xb=SentenceData.from_dict(
                data.get("s_xb")) if data.get("s_xb") else None,
            s_ax=SentenceData.from_dict(
                data.get("s_ax")) if data.get("s_ax") else None,
            s_xx=SentenceData.from_dict(
                data.get("s_xx")) if data.get("s_xx") else None,
        )

    def insert_sentence(self, what_type: str, sentence_to_insert: SentenceData):
        if what_type == 's_ab':
            self.s_ab = sentence_to_insert
        elif what_type == 's_xb':
            self.s_xb = sentence_to_insert
        elif what_type == 's_ax':
            self.s_ax = sentence_to_insert
        elif what_type == 's_xx':
            self.s_xx = sentence_to_insert
        else:
            raise ValueError("Invalid type")

    def is_full(self):
        return (self.s_ab != None) and (self.s_xb != None) and (self.s_ax != None) and (self.s_xx != None)


# save sentence data to a json
# works on both SentenceData objects and TupleSentenceData objects
def corpus_to_json(input_data: Union[Iterable[SentenceData], Iterable[TupleSentenceData]], filename: str = None) -> str:
    if filename == None:
        filename = input("Provide (relative) file path: ")

    output = [possibly_tuple_sentence_data.to_dict()
              for possibly_tuple_sentence_data in input_data]
    try:
        with open(filename, "w") as json_file:
            json.dump(output, json_file, indent=2)
    except Exception as e:
        print(f"An error occurred: {e}")

    return filename


# load tuples of sentences from json, get tuple of the tuples (2x2s)
# user must define whether or not the corpus is made up of SentenceData objects or TupleSentenceData objects
def corpus_from_json(where_to_load: str = None, is_tuples: bool = False) -> Union[tuple[SentenceData], tuple[TupleSentenceData]]:
    if where_to_load == None:
        where_to_load = input("Provide (relative) file path: ")

    try:
        with open(where_to_load, "r") as json_file:
            loaded_data = json.load(json_file)
    except Exception as e:
        print(f"An error occurred: {e}")

    # user must tell function whether it is looking for tuples or just sentences
    if is_tuples:
        output = tuple([TupleSentenceData.from_dict(loaded_dict)
                        for loaded_dict in loaded_data])
    else:
        output = tuple([SentenceData.from_dict(loaded_dict)
                        for loaded_dict in loaded_data])
    return output


# more human readable format, not meant to interface w/ the rest of the code
def simple_convert_to_csv(data: Iterable[TupleSentenceData], filename: str, a_condition: str = "a", b_condition: str = "b", is_tuples: bool = True):
    group = 0

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # write header line
        writer.writerow(['group', 'grammatical', 'sentence',
                        'a condition', 'b condition'])

        # write data rows
        for quad in data:
            # ab
            writer.writerow(
                [group, quad.s_ab.grammatical, str(quad.s_ab),
                 f"+{a_condition}", f"+{b_condition}"])
            # xb
            writer.writerow(
                [group, quad.s_xb.grammatical, str(quad.s_xb),
                 f"-{a_condition}", f"+{b_condition}"])
            # ax
            writer.writerow(
                [group, quad.s_ax.grammatical, str(quad.s_ax),
                 f"+{a_condition}", f"-{b_condition}"])
            # xx
            writer.writerow(
                [group, quad.s_xx.grammatical, str(quad.s_xx),
                 f"-{a_condition}", f"-{b_condition}"])
            group = group + 1  # next group of sentences
