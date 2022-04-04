from enum import Enum 

class ModelSettings(Enum):
    MODEL_PATH = 'token_rnn/saved_models/2apr2021'
    MAX_SEQUENCE_LENGTH = 30
    DATAPATH = "train_data/"