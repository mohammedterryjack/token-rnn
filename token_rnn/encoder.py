from typing import List

from numpy import ndarray
from ffast import load

class Encoder:
    def __init__(self) -> None:
        self.BOS_TOKEN = "<bos>"
        self.EOS_TOKEN = "<eos>"
        self.tokeniser = load('poincare')
        self.tokeniser.add_special_token(self.BOS_TOKEN)
        self.tokeniser.add_special_token(self.EOS_TOKEN)
        BOS,EOS = self.tokeniser.encode(f"{self.BOS_TOKEN} {self.EOS_TOKEN}").ids
        self.BOS_TOKEN_ID = BOS 
        self.EOS_TOKEN_ID = EOS
        self.UNKNOWN_TOKEN_ID = len(self.tokeniser)-1

    def token_ids_vectoriser(self, token_ids:List[int]) -> List[ndarray]:
        return list(map(
            lambda token_vector:token_vector.reshape((-1,1)),
            self.tokeniser.decode(token_ids).semantics()
        ))

    def condition_vectoriser(self, text:str) -> ndarray:
        return self.tokeniser.encode(text).vector.reshape(-1,1)

    def format_sentence(self, sentence:str) -> str:
        return f"{self.BOS_TOKEN} {sentence} {self.EOS_TOKEN}"

    @staticmethod
    def format_condition(style:str,sentiment:str,keywords:List[str]) -> str:
        return f"{style} {sentiment} {' '.join(keywords)}"

