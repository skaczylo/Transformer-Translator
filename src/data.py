import pandas as pd
from torch.utils.data import Dataset
from translator_tokenizer import TranslatorTokenizer

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tokenizers import Tokenizer as HFTokenizer

class TranslatorTokenizer:
    """
    TOKENIZADOR
    Envuelve el tokenizador de HuggingFace y gestiona la inyección de 
    tokens especiales, truncado y padding.
    """

    def __init__(self, path: str, context_length: int):
        self.encoder = HFTokenizer.from_file(path)
        self.context_length = context_length
        
        # Extraemos y guardamos los IDs
        self._pad_id = self.encoder.token_to_id("<PAD>")
        self._start_id = self.encoder.token_to_id("<START>")
        self._end_id = self.encoder.token_to_id("<END>")

    def __len__(self) -> int:
        return self.encoder.get_vocab_size()

    # Usar @property permite acceder a ellos sin paréntesis: tokenizer.pad_id
    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def end_id(self) -> int:
        return self._end_id

    @property
    def start_id(self) -> int:
        return self._start_id

    def encode(self, text: str, pad: bool = True) -> list:
        
        raw_tokens = self.encoder.encode(text).ids
        
        tokens = [self.start_id] + raw_tokens + [self.end_id]

        #Truncado dejando espacio para el END
        if len(tokens) > self.context_length:
            tokens = tokens[:(self.context_length - 1)] + [self.end_id]
        
        #Padding 
        if pad and len(tokens) < self.context_length:
            pad_count = self.context_length - len(tokens)
            tokens.extend([self.pad_id] * pad_count)

        return tokens

    def encode_batch(self, texts: list, pad: bool = True) -> torch.Tensor:
        
        batch_tokens = [self.encode(text, pad=pad) for text in texts]
        
        return torch.tensor(np.array(batch_tokens), dtype=torch.long)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """Convierte tensores/listas de IDs de vuelta a texto legible"""
        if isinstance(ids, (torch.Tensor, np.ndarray)):
            ids = ids.tolist()
        return self.encoder.decode(ids, skip_special_tokens=skip_special_tokens)
    
    
class TranslationDataset(Dataset):
    """
    DATASET 
    Recibe un DataFrame de Pandas, extrae las columnas y las tokeniza en memoria 
    una sola vez para que el entrenamiento con GPU sea más rapido.
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer: TranslatorTokenizer, split_name: str = "Dataset"):
        
        textos_en = df['en'].astype(str).tolist()
        textos_es = df['es'].astype(str).tolist()

        self.X_tensors = tokenizer.encode_batch(textos_en)
        self.Y_tensors = tokenizer.encode_batch(textos_es)

    def __len__(self) -> int:
        return len(self.X_tensors)

    def __getitem__(self, idx: int):
        
        return self.X_tensors[idx], self.Y_tensors[idx]