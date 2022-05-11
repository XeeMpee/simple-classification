import os
import shutil

import yaml


class Config:
    def __init__(self) -> None:
        if not os.path.exists("config.yaml"):
            shutil.copyfile("templates/config.yaml", "config.yaml")
        with open("config.yaml", "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            
        self.input = config["general"]["input"]
        self.class_tag = config["general"]["class-tag"] if config["general"]["class-tag"] is not None else "class"
        self.learn_model = config["general"]["learn-model"]
        self.generate_dataspec = config["general"]["generate-dataspec"]
        
        self.random_corruption = config["data-preprocessing"]["random-corruption"]
        self.malformed_policy = config["data-preprocessing"]["malformed-policy"]
        self.save_dataframe = config["data-preprocessing"]["save-dataframe"]
        self.pca_analyse = config["data-preprocessing"]["pca"]
        self.redundant_columns = config["data-preprocessing"]["redundant-columns"] if config["data-preprocessing"]["redundant-columns"] is not None else []
        self.malformed_values = config["data-preprocessing"]["malformed-values"] if config["data-preprocessing"]["malformed-values"] is not None else []