import os

import pandas as pd
import swifter

def parser(x):
    return ":".join(x.split(":")[:-1] + ["00"])

def agg(filename="train"):  # w/o grouping agg methods
    if not os.path.exists(f"./data/{filename}_need_aggregate.csv"):
        raise FileNotFoundError

    df = pd.read_csv(f"./data/{filename}_need_aggregate.csv")

    result = {}
    for x in df.itertuples():
        dt = parser(x[1])
        if dt in result:
            result[dt].append(x[2])
        else:
            result[dt] = [x[2]]

    pd.DataFrame(result.items(), columns=df.columns).to_csv(
        f"./result/{filename}.csv", index=False
    )


def time_idx_resample(filename="train"):  # time index resample
    if not os.path.exists(f"./data/{filename}_need_aggregate.csv"):
        raise FileNotFoundError

    (
        pd.read_csv(
            f"./data/{filename}_need_aggregate.csv",
            index_col="datetime",
            parse_dates=True,
        )
        .resample("T")
        .agg({"EventId": lambda x: x.tolist()})
        .to_csv(f"./result/{filename}.csv")
    )


if __name__ == "__main__":
    if not os.path.exists("result"):
        os.mkdir("result")

    # w/o grouping agg methods
    # agg("train")
    # agg("test")

    # time index resample
    time_idx_resample("train")
    time_idx_resample("test")

    # w/ grouping method
    # df_tr = pd.read_csv("./data/train_need_aggregate.csv")
    # df_te = pd.read_csv("./data/test_need_aggregate.csv")

    # df_tr["datetime"] = df_tr["datetime"].swifter.apply(parser)

    # df_te["datetime"] = df_te["datetime"].swifter.apply(parser)

    # df_tr_agg = (
    #     df_tr.groupby("datetime")["EventId"]
    #     .apply(list)
    #     .reset_index(name="EventId")
    #     .to_csv("./result/train.csv", index=False)
    # )

    # df_te_agg = (
    #     df_te.groupby("datetime")["EventId"]
    #     .apply(list)
    #     .reset_index(name="EventId")
    #     .to_csv("./result/test.csv", index=False)
    # )
