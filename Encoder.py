import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn


CONTEXT_LENGTH = 300
D_EMBEDDING = 512
ATTENTION_HEADS = 8
DROPOUT = 0.0


#Attention Head
"""ARQUITECTURA TRANSFORMER"""

class AttentionHead(nn.Module):
    """Una sola capa de self-attention"""

    def __init__(self, dimension, attention_mask = None):
        super().__init__()
        self.key = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias=False)
        self.query = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias=False)
        self.value = nn.Linear(in_features=D_EMBEDDING,out_features=dimension, bias = False)

        self.mask = attention_mask 


        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self,x):

        #B = numero de Batch; T = numero de "tokens"; C = dimension de cada token
        B, T, C = x.shape

        K = self.key(x) # (B, T , C)
        Q = self.key(x) #(B,T,C)
        V = self.key(x) # (B,T,C)

        #Calculamos la Atención QxK^T
        """
        Calculamos la "afinidad" Q x K^t
        Como tenemos batches, usamos el operador @ que aplica la multiplicacion en las dos ultimas dimensiones B veces
        K.transpose significa transponer la penultima dimensino T con la pultima dimension C. 
        """
        scores = Q @ K.transpose(-2,-1) #(B,T,C) x (B,C,T) = (B,T,T)
        scores = scores /(C ** 0.5)
        scores = scores + self.mask  #Sumamos la mascara para evitar que se fije en tokens <PAD>

        scores = F.softmax(scores, dim=-1) #Aplicamos softmax sobre las filas, es decir, sobre la ultima dimension 

        attention = scores @ V

        return attention
    


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


