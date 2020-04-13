'''
JMJPFU
6-April-2020
This is the files for all the model related code

Lord bless this attempt of yours
'''

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

class Model:
    # THis is the class for all the modelling methods

    def __init__(self,bertModel,bertTokenizer):
        self.tokenizer = bertTokenizer
        self.model = bertModel

    def genMeanvec(self,string):
        marked_text = "[CLS] " + string + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # Generating the indexed tokens
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Mark each of the token as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Taking the mean of layers from 1 to last layer
        avg_embedding = torch.mean(token_embeddings[1:], dim=0)
        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(avg_embedding, dim=0)
        # Convert into a numpy array
        numpy_embedding = sentence_embedding.numpy()
        return numpy_embedding