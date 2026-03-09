import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


CONTEXT_LENGTH = 300
D_EMBEDDING = 512
ATTENTION_HEADS = 8

gpt2 = tiktoken.get_encoding("gpt2")
dataset_json = "dataset.json" #Ruta al json con el dataset

encoder = tiktoken.Encoding(
    name = "encoder",
    pat_str = gpt2._pat_str,
    mergeable_ranks = gpt2._mergeable_ranks,
    special_tokens = {
        **gpt2._special_tokens,
        "<START>": len(gpt2._mergeable_ranks) +1,
        "<END>": len(gpt2._mergeable_ranks) +2,
        "<PAD>": len(gpt2._mergeable_ranks) +3
    }
)









"""
ARQUITECTURA DEL TRANSFORMER
"""
class Encoder(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        """
        Embedding: Tabla E de dimension VOCAB_SIZE x D_EMBEDDING
        x_n tendrá dimension D_EMBEDDING
        Pytorch no multiplica matrices aquí, solo indica, dado un vector de indices de tokens, que filas de la matriz E extraer
        """
       
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=D_EMBEDDING, padding_idx= encoder._special_tokens["<PAD>"])

        """
        Positional Encoding: x_n = x_n + r_n -> r_ tiene dimensión D_EMBEDDING
        El objetivo es que permita determinar en que posición se encuentra cada palabra, por eso el numero de embeddings es CONTEXT_LENGT
        """
        self.position_embedding_table = nn.Embedding(num_embeddings= CONTEXT_LENGTH, embedding_dim=D_EMBEDDING)






"""  
DATALOADER 
"""
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.X = self.data['en']
        self.y = self.data['es']
        

    def __len__(self):
        return len(self.data)
    
    def adjust_length(self,text):
        pad = [encoder._special_tokens["<PAD>"] for _ in range(CONTEXT_LENGTH-len(text))]
        return np.concatenate((text,pad))
    

    def __getitem__(self, idx):

        en_text = encoder.encode(self.X.iloc[idx])
        es_text = encoder.encode(self.y.iloc[idx])


        #Truncar entrada en caso de que se supere dimension entrada x_n
        if len(en_text) > CONTEXT_LENGTH:
            en_text =  en_text[:300]
        
        if len(es_text) > CONTEXT_LENGTH:
            es_text = es_text[:(CONTEXT_LENGTH-2)]


        en_text = self.adjust_length(en_text)
        es_text = self.adjust_length(es_text)

       
        #Añadimos tokens especiales al targe
        es_text = np.concatenate(([encoder._special_tokens["<START>"]],
                                  es_text[:(CONTEXT_LENGTH-2)],
                                  [encoder._special_tokens["<END>"]]),
                                  axis=0)


        #COnvertimos a tensores
        es_text =  torch.tensor(es_text,dtype=torch.long)
        en_text = torch.tensor(en_text,dtype= torch.long)

        return en_text, es_text