import os
import yaml
import json
import pandas as pd


class Data:
    """
    The primary artifact fed into the training algorithm to fit the best model
    """

    def __init__(self, filepath=None, dataset=None):
        self.filepath = filepath
        self.dataset = dataset
        self.load_dataset()

    def load_dataset(self):
        self.dataset = pd.read_parquet(self.filepath)
        return self.dataset

    def get_dataset(self):
        return self.dataset
    
    def output_dataset(self, filepath):
        self.dataset.to_parquet(filepath)


class Report:
    """
    Structured information about the results obtained after applying a function
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.filetype = filepath.split(".")[1]
        self.load_report()

    def load_report(self):
        if os.path.isfile(self.filepath):
            if self.filetype == "csv":
                return pd.read_csv(self.filepath)
            elif self.filetype == "json":
                return pd.read_json(self.filepath)
        return None


class Model:
    """
    Either a single file or multiple files constituting the model
    """

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name


class Configuration:
    """
    Declarative specifications used to orchestrate the execution of individual components or pipelines
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)

    def load_config(self):
        with open(self.filepath, "r") as file:
            config = yaml.safe_load(file)
        return config

    def save_config(self, configs):
        with open(self.filepath, "w") as file:
            yaml.dump(configs, file, default_flow_style=False)


class Status:
    """
    A blocking artifact that defines the pipeline execution
    """

    def __init__(self, status_key, status_file):
        # TODO load status file
        self.status_key = status_key
        self.status_file = status_file

    def change_status(self, new_status):
        self.status_key = status_key
        value = open(self.status_file, "r")
        data = eval(value)

        data[self.status_key] = new_status
        with open(self.status_file, "w") as file:
            file.write("status = ")
            json.dump(data, file, indent=4)


class Documentation:
    """
    Files that contain human-readable content
    """

    def __init__(self, type, filepath, content):
        self.type = type
        self.filepath = filepath
        self.content = content
    
    def get_content(self):
        return self.content
    
    def save_content(self):
        with open(self.filepath, "w") as file:
            file.write("\n\n ".join(self.type, self.content))


class Function:
    """
    An implementation function
    """

    def __init__(self, name, file):
        self.name = name
        self.file = file


class Service:
    """
    Service encapsulating the model and parameters for serving it, matching evaluation parameters
    """

    def __init__(self, name):
        self.name = name
        
    


class Logs:
    """
    Generated during each phase of the AI lifecycle. Logging of events or Observability services
    """

    def __init__(self, path, level="INFO"):
        self.path = path
        self.level = level
