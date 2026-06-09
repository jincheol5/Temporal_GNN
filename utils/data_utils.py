import os
import pandas as pd

class DataUtils:
    base_path=os.path.join("..","data","temporal_gnn")
    @staticmethod
    def preprocess_dataset(dataset_name:str):
        """
        Return
            pd.DataFrame
        """
        dataset_path=os.path.join("dataset",dataset_name,f"{dataset_name}.txt") # ex
