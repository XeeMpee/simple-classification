from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.ensemble import StackingClassifier, VotingClassifier
from src.models.learn_model_factory import LearnModelFactory
from src.models.learn_model import LearnModel
from src.config.config import Config
from src.common.data_restorer import DataRestorer, RestorationType
import src.dataspec.gen_data_spec as dataspec
from src.utils.data_processing_utils import DataProcessingUtils

from src.models.strategies.svm_model_strategy import SvmModelStrategy
from src.models.strategies.random_forest_model_strategy import RandomForestModelStrategy


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
        df = DataProcessingUtils().malform(df, config.class_tag, config.random_corruption)

    if config.malformed_policy and config.malformed_policy == "remove":
        df = rows_purification(df, config.class_tag, config.malformed_values)

    if config.malformed_policy and config.malformed_policy == "restore":
        df = DataRestorer(RestorationType.MEAN, config.malformed_values).restore(
            df, config.class_tag)

    if config.pca_analyse:
        pca = PCA(n_components=config.pca_analyse)
        dfx = df.drop(config.class_tag, axis=1).to_numpy().astype(np.int64)
        dfy = df[config.class_tag].to_numpy().astype(np.int64)
        pca.fit(dfx)
        data = pca.transform(dfx)
        dfx = pd.DataFrame(data=data,
                           columns=[
                               [f'data[{i}]' for i in range(0, config.pca_analyse)]]
                           )
        df = dfx.assign(**{config.class_tag: dfy})

    if config.save_dataframe:
        now = datetime.now()
        current_time_str = now.strftime("%Y%d%m%H%M%S")
        df.to_csv(f"data/{current_time_str}.csv")

    if config.learn_models:
        dataset = DataProcessingUtils().divide(df, config.class_tag, 0.2)
        models = []
        for model_name in config.learn_models:
            models.append(LearnModelFactory().create(model_name))

        if config.ensemble_learning is False:
            for indx, model in enumerate(models):
                model.fit(dataset.X_train, dataset.y_train.ravel())
                y_predicted = model.predict(dataset.X_test)
                accuracy = metrics.accuracy_score(dataset.y_test, y_predicted)
                print(f"{config.learn_models[indx]} accuracy: {accuracy}")
        else:
            if len(models) < 2:
                raise RuntimeError(
                    "Insufficient number of models for ensemble learning")
            else:
                estimators = []
                for indx, model in enumerate(models):
                    estimators.append((config.learn_models[indx], model.raw()))

            voting_classifier_hard = VotingClassifier(
                estimators=estimators, 
                voting='hard'
            )
            
            voting_classifier_soft = VotingClassifier(
                estimators=estimators, 
                voting='soft'
            )
            
            stacking_classifier =StackingClassifier(
                estimators=estimators, 
            )
            
            voting_classifier_hard.fit(dataset.X_train,dataset.y_train.ravel())
            y_predicted = voting_classifier_hard.predict(dataset.X_test)
            accuracy = metrics.accuracy_score(dataset.y_test, y_predicted)
            print(f"voting classifier (hard voting) accuracy: {accuracy}")
            
            
            voting_classifier_soft.fit(dataset.X_train,dataset.y_train.ravel())
            y_predicted = voting_classifier_soft.predict(dataset.X_test)
            accuracy = metrics.accuracy_score(dataset.y_test, y_predicted)
            print(f"voting classifier (soft voting) accuracy: {accuracy}")
            
            
            stacking_classifier.fit(dataset.X_train,dataset.y_train.ravel())
            y_predicted = stacking_classifier.predict(dataset.X_test)
            accuracy = metrics.accuracy_score(dataset.y_test, y_predicted)
            print(f"stacking classifier accuracy: {accuracy}")
            

    if config.generate_dataspec:
        dataspec.run(df, config.class_tag)
