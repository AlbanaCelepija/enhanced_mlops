import pandas as pd
import digitalhub as dh
from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation

class DataPlatform(Data):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def load_dataset(self, product_name):
        self.project = dh.get_or_create_project(product_name)
        dataset = self.project.get_dataitem(self.filepath)
        return dataset
    
    def log_dataset(self, dataset, product_name):
        self.project = dh.get_or_create_project(product_name)
        self.project.log_dataitem(data=dataset, name=self.filepath)
        

