from dataset_creation.read.reader import Reader
from token_rnn.encoder import Encoder
from token_rnn.utils import ModelSettings

def create_dataset(dataset_size:int) -> None:
    encoder = Encoder()
    n_categories = 6
    counters = dict(
        statement_positive=0,
        statement_neutral=0,
        statement_negative=0,
        question_positive=0,
        question_neutral=0,
        question_negative=0
    )
    datastyle_size = dataset_size//n_categories
    seen = set()
    with open(ModelSettings.DATAPATH.value+'conditions.txt','w') as condition_file,open(ModelSettings.DATAPATH.value+'sentences.txt','w') as sentence_file:
        for (style,sentiment,keywords),sentence in Reader().read(shuffled=False, skip_keywords_greater_than=1,skip_keywords_less_than=1):
            if counters[f"{style}_{sentiment}"] > datastyle_size:
                continue
            sentence = encoder.format_sentence(sentence)
            tokens_ids = encoder.tokeniser.encode(sentence).ids
            if len(tokens_ids) < 2:
                continue
            if encoder.UNKNOWN_TOKEN_ID in tokens_ids:
                continue
            condition = encoder.format_condition(style,sentiment,keywords)
            if condition in seen:
                continue
            counters[f"{style}_{sentiment}"] += 1
            seen.add(condition)
            condition_file.write(f"{condition}\n")
            sentence_file.write(f"{sentence}\n")
