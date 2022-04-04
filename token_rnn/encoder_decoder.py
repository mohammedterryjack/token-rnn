from typing import Generator, List, Optional

from numpy import ndarray
from windML import RNN

from token_rnn.utils import ModelSettings
from token_rnn.encoder import Encoder

class ConditionalNLG:
    def __init__(self) -> None:
        self.STYLES = ("question","statement")
        self.SENTIMENTS = ("positive","neutral","negative")
        self.encoder = Encoder()
        self.decoder = RNN(
            load_path=ModelSettings.MODEL_PATH.value,
            token_vector_size=self.encoder.tokeniser.token_size,
            token_vocabulary_size=len(self.encoder.tokeniser),
            hidden_dimension=self.encoder.tokeniser.size
        )

    def train(self, epochs:int=100, data_path:str=ModelSettings.DATAPATH.value) -> None:
        self.decoder.fit(
            token_ids_vectoriser=self.encoder.token_ids_vectoriser,
            token_ids=list(self.read_sentences(data_path)), 
            encoded_contexts=list(self.read_conditions(data_path)),
            epochs=epochs
        )
        self.decoder.save(ModelSettings.MODEL_PATH.value)

    def generate(self, style:str, sentiment:str, keywords:List[str], prompt:Optional[str]=None) -> str:
        assert style in self.STYLES and sentiment in self.SENTIMENTS
        generated_token_ids = self.decoder.generate(
            bos_id=self.encoder.BOS_TOKEN_ID, 
            eos_id=self.encoder.EOS_TOKEN_ID, 
            token_ids_vectoriser=self.encoder.token_ids_vectoriser, 
            prompt_ids=list() if prompt is None else self.encoder.tokeniser.encode(prompt).ids,
            condition_vector=self.encoder.condition_vectoriser(
                self.encoder.format_condition(
                    style=style,
                    sentiment=sentiment, 
                    keywords=keywords
                )
            ),
            greedy=True
        )
        return str(self.encoder.tokeniser.decode(generated_token_ids))

    def read_conditions(self,data_path:str) -> Generator[ndarray,None,None]:
        with open(data_path+'conditions.txt') as condition_file:
            for condition in condition_file.readlines():
                yield self.encoder.condition_vectoriser(condition.strip()) 

    def read_sentences(self,data_path:str) -> Generator[List[int],None,None]:
        with open(data_path+'sentences.txt') as sentence_file:
            for sentence in sentence_file.readlines():
                yield self.encoder.tokeniser.encode(sentence).ids 