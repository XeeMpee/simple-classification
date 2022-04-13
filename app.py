import argparse
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA
from src.common.data_restorer import DataRestorer, RestorationType
from src.common.class_tags_converter import ClassLabelsConverter
import src.dataspec.gen_data_spec as dataspec
from src.utils.malformer import Malformer


def column_purification(df, redundant_columns):
    for label in redundant_columns:
        if(label in df.columns):
            df.pop(label)
    print("columns purified...")


def rows_purification(df, malformed_rows_tags):
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


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description='Process some integers.') 
    parser.add_argument('-m','--random-malform', help='randomly malform some values', type=float, default=None)
    parser.add_argument('-r','--remove-malformed', help='removes malformed values', action="store_true")
    parser.add_argument('-c','--restoration', help='restores blank values with mean', action="store_true")
    parser.add_argument('-s','--save-dataframe', help='saves dataframe after operations', action="store_true")
    parser.add_argument('-t', '--class-tag', help='defines class tag', type=str)
    parser.add_argument('-p', '--pca', help='defines class tag', type=int)
    parser.add_argument('--dataspec', help='generates data specification', action="store_true")
    args = parser.parse_args()
    # fmt: on

    redundant_columns_tags = ["obj_ID"]
    malformed_rows_tags = ["#NUM!"]
    class_tag = "class" if args.class_tag is None else args.class_tag

    df = pd.read_csv("data/waterQuality1.csv")
    df = ClassLabelsConverter(df, "is_safe").convert()
    column_purification(df, redundant_columns_tags)

    if(args.remove_malformed is not None):
        rows_purification(df, malformed_rows_tags)

    if args.random_malform is not None:
        print("Malforming..")
        Malformer(args.random_malform).malform(df)
        print("Malfroming done!")

    if args.restoration is not None:
        DataRestorer(RestorationType.MEAN).restore(df)

    if args.pca is not None:
        pca = PCA(n_components=args.pca)
        pca.fit(df)
        data = pca.transform(df)
        df = pd.DataFrame(data = data
             , columns = [[f'data[{i}]' for i in range(0,args.pca)]])

    if args.save_dataframe is not None:
        now = datetime.now()
        current_time_str = now.strftime("%Y%d%m%H%M%S")
        df.to_csv(f"data/{current_time_str}.csv")

    if args.dataspec:
        print("Data spec generating...")
        dataspec.run(df, class_tag)
        print("Dataspec generating done")
