import os

import numpy as np
import pandas as pd
import pendulum as pdl
import swifter
import torch
from torch.utils.data import DataLoader

from model import Model, dataset

if __name__ == "__main__":
    if not os.path.exists("result"):
        os.mkdir("result")

    win = 22

    dataset_te = dataset(task="test", window_size=win)
    model = Model(dataset_te.features())

    if os.path.exists("./model/model_best"):
        model.load_state_dict(
            torch.load("./model/model_best", map_location=torch.device("cpu"))
        )
    elif os.path.exists("./model/model"):
        model.load_state_dict(
            torch.load("./model/model", map_location=torch.device("cpu"))
        )
    else:
        print("no trained model to load")

    loader_te = DataLoader(
        dataset=dataset_te, shuffle=False, num_workers=1, batch_size=1024
    )

    report = pd.read_csv("../exam2/data/test_need_aggregate.csv")

    with torch.no_grad():
        model.eval()
        pred = []
        for s, batch in enumerate(loader_te):
            pred.append(model(batch["x"]))

        prob = torch.cat(pred, dim=0)

    anomaly_prob = []
    confidence = []

    for record, p in zip(list(report.EventId[win:] - 1), prob.numpy()):
        anomaly_prob.append((1 + max(p) - 2 * p[record]) / 2)
        confidence.append((1 - max(p)))

    report["conf"] = [1 for i in range(win)] + confidence
    report["anomaly_prob"] = [0 for i in range(win)] + anomaly_prob

    report["anomaly"] = [0 for x in report.index]
    report.loc[
        (
            (report["anomaly_prob"] > np.quantile(report["anomaly_prob"], 0.997))
            & (report["anomaly_prob"] > 0.7)
        ),
        "anomaly",
    ] = 1

    report["datetime"] = report["datetime"].swifter.apply(
        lambda x: pdl.parse(x).format("YYYY-MM-DD HH:mm:00")
    )

    report_agg = (
        report.groupby("datetime")["anomaly"]
        .apply(list)
        .apply(max)
        .reset_index(name="anomaly")
    )

    pd.merge(
        pd.read_csv("../exam2/result/test.csv"), report_agg, how="left", on=["datetime"]
    ).to_csv("./result/predict.csv", index=False)

