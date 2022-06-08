from datetime import datetime
import itertools
from django import conf
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from src.metrics.model_metrics import ModelMetrics
from src.models.ensemble_learn_model_factory import EnsembleLearnModelFactory
from src.models.learn_model_factory import LearnModelFactory
from src.config.config import Config
from src.common.data_restorer import DataRestorer, RestorationType
import src.dataspec.gen_data_spec as dataspec
from src.utils.data_processing_utils import DataProcessingUtils
import sys

np.set_printoptions(threshold=sys.maxsize)


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

    if config.generate_dataspec:
        dfx = df.drop(config.class_tag, axis=1)
        dfy = df[config.class_tag].to_numpy().astype(np.int64)
        dataspec.run(dfx, dfy, config.class_tag)

    if config.save_dataframe:
        now = datetime.now()
        current_time_str = now.strftime("%Y%d%m%H%M%S")
        df.to_csv(f"data/{current_time_str}.csv")

    if config.learn_models:
        models = []

        partial_models = []
        for model_name in config.learn_models:
            partial_models.append(LearnModelFactory().create(model_name))

        if config.ensemble_learning_classifiers is None:
            models = partial_models
        else:
            if len(partial_models) < 2:
                raise RuntimeError(
                    "Insufficient number of models for ensemble learning")

            ensemble_classifiers = []
            for ensemble_classifier_name in config.ensemble_learning_classifiers:
                models.append(EnsembleLearnModelFactory().create(
                    ensemble_classifier_name, partial_models))

        # training models:
        dataset = DataProcessingUtils().divide(df, config.class_tag, 0.2)
        for model in models:
            # model.fit(
            #     dataset.X_train,
            #     dataset.y_train.ravel()
            # )
            # y_predicted = model.predict(dataset.X_test)

            # if(config.print_predicted):
            #     print(y_predicted)

            # # printing metrics
            # if config.metrics:
            #     print(f"{model.name()} metrics: ")
            #     for metrics in ModelMetrics(config, dataset.y_test, y_predicted).all_metrics():
            #         print(metrics)

            # TEMPORARY
            print("Default run")
            model.fit(dataset.X_train, dataset.y_train.ravel())
            y_predicted = model.predict(dataset.X_test)
            print(ModelMetrics(config, dataset.y_test, y_predicted).all_metrics())
            
            print("Optimized run")
            grid_search_cv = GridSearchCV(
                estimator=model.raw(),
                param_grid=config.search_grid,
                scoring=config.metrics[0],
                n_jobs=8,
            )
            grid_search_cv.fit(dataset.X_train, dataset.y_train.ravel())
            y_predicted = grid_search_cv.predict(dataset.X_test)
            print(ModelMetrics(config, dataset.y_test, y_predicted).all_metrics())
            print(f"Best params: {grid_search_cv.best_params_}")
            
