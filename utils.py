import os
import pandas as pd
import tensorflow as tf
import numpy as np

from typing import List
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
from cddd.hyperparameters import DEFAULT_DATA_DIR

_default_model_dir = os.path.join(DEFAULT_DATA_DIR, 'default_model')


def get_model(batch_size=512, use_gpu=True):
    """
    Get CDDD model using CDDD repo logic with 'InferenceModel',
    default model location, and use GPU (when available)
    """

    infer_model = InferenceModel(model_dir=_default_model_dir,
                                 use_gpu=use_gpu,
                                 batch_size=batch_size)

    return infer_model


def smiles_to_embedding(smiles: List[str], infer_model: InferenceModel) -> pd.DataFrame:
    """
    Utilize CDDD model functionality to produce CDDD embeddings given a model,
    and a list of smiles.Returns a pandas dataframe with columns:
    'provided_smiles', 'processed_smiles', processed in such a way as to match
    the model training data (removing salts/stereochemistry), and columns
    "cddd_1-512" corresponding to embeddings.

    If a given smiles in uncrecognized, nan entries are returned.
    """

    smiles = pd.DataFrame({'provided_smiles': smiles})

    # contains nan for unrecognized smiles
    smiles['processed_smiles'] = smiles['provided_smiles'].map(preprocess_smiles)

    sml_list = smiles[~smiles.processed_smiles.isna()].processed_smiles.tolist()

    # embedding is of size 512
    embeddings_colnames = ["cddd_" + str(i+1) for i in range(512)] 

    if len(sml_list) > 0:
        descriptors = infer_model.seq_to_emb(sml_list)
        smiles = smiles.join(pd.DataFrame(
            descriptors,
            index=smiles[~smiles['processed_smiles'].isna()].index,
            columns=embeddings_colnames
            ))
    else:
        # no provided smiles are recognized as ok, return empty pandas DF
        smiles = smiles.reindex(
            columns=list(smiles.columns)+embeddings_colnames)

    return smiles
