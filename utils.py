import os
import pandas as pd
import tensorflow as tf

from typing import List
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
from cddd.hyperparameters import DEFAULT_DATA_DIR

_default_model_dir = os.path.join(DEFAULT_DATA_DIR, 'default_model')



def smiles_to_embedding(smiles : List[str]) -> pd.DataFrame:
    """
    # TODO docstring, typing
    """
    smiles = pd.DataFrame({'provided_smiles': smiles})


    # TODO: parametrize
    infer_model = InferenceModel(model_dir=_default_model_dir,
                                    use_gpu=False,
                                    batch_size=512,
                                    cpu_threads=2)

    smiles['processed_smiles'] = smiles['provided_smiles'].map(preprocess_smiles)  # contains nan for unrecognized smiles

    sml_list = smiles[~smiles.processed_smiles.isna()].processed_smiles.tolist()


    descriptors = infer_model.seq_to_emb(sml_list)




    smiles = smiles.join(pd.DataFrame(descriptors,
                                index=smiles[~smiles['processed_smiles'].isna()].index,
                                columns=["cddd_" + str(i+1) for i in range(512)]))


    return smiles

