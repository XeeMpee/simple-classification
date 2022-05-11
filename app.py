from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.config.config import Config
from src.common.data_restorer import DataRestorer, RestorationType
from src.common.class_tags_converter import ClassLabelsConverter
import src.dataspec.gen_data_spec as dataspec
from src.utils.malformer import Malformer

from src.models.svm_model import SvmModel


def column_purification(df, redundant_columns):
    for label in redundant_columns:
        if(label in df.columns):
            df.pop(label)
    print("columns purified...")
    return df


def rows_purification(df, class_tag, malformed_rows_tags):
    malformed_rows = []
    for column in df.columns:
        for malform_value in malformed_rows_tags:
            malformed_rows += df.index[df[column] == malform_value].tolist()
    for row in malformed_rows:
        try:
            df.drop(row, axis=0, inplace=True)
        except:
            pass
    print("rows purified...")
    return df


if __name__ == "__main__":
    config = Config()

    df = pd.read_csv(config.input)

    if config.redundant_columns:
        df = column_purification(df, config.redundant_columns)
    
    if config.random_corruption:
        df = Malformer(config.random_corruption).malform(df, config.class_tag)

    if config.malformed_policy and config.malformed_policy == "remove":
        df = rows_purification(df, config.class_tag, config.malformed_values)

    if config.malformed_policy and config.malformed_policy == "restore":
        df = DataRestorer(RestorationType.MEAN, config.malformed_values).restore(df, config.class_tag)

    if config.pca_analyse:
        pca = PCA(n_components=config.pca_analyse)
        dfx = df.drop(config.class_tag, axis=1)
        dfy = df[config.class_tag].to_numpy().astype(np.int)
        pca.fit(dfx.to_numpy())
        data = pca.transform(dfx)
        dfx = pd.DataFrame(data=data,
                          columns=[[f'data[{i}]' for i in range(0, config.pca_analyse)]]
                          )
        df = dfx.assign(**{config.class_tag : dfy})
        
    if config.save_dataframe:
        now = datetime.now()
        current_time_str = now.strftime("%Y%d%m%H%M%S")
        df.to_csv(f"data/{current_time_str}.csv")

    if config.learn_model and config.learn_model == "svm":
        model = SvmModel(config.class_tag, df)
        model.train()

    if config.generate_dataspec:
        dataspec.run(df, config.class_tag)
