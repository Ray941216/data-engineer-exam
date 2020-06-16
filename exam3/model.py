import math
import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.lstm_in = nn.LSTM(
            input_size=features, hidden_size=3, batch_first=True
        )
        self.dnn_out = nn.Linear(
            in_features=3, out_features=features
        )

    def forward(self, x):
        output, (ht, ct) = self.lstm_in(x)
        return F.softmax(self.dnn_out(F.relu(ht.view(-1, ht.size(-1)))), dim=-1)
class dataset(Dataset):
    def __init__(self, window_size=5, step_size=1, task="train", evaluate_ratio=0.2):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size

        if os.path.exists("./onehotencoder.pkl"):
            with open("./onehotencoder.pkl", "rb") as f:
                self.onehot = pickle.load(f)
        else:
            self.onehot = OneHotEncoder(handle_unknown="ignore")
            self.onehot.fit(
                pd.read_csv("../exam2/data/train_need_aggregate.csv")["EventId"]
                .unique()
                .reshape(-1, 1)
            )
            with open("./onehotencoder.pkl", "wb") as f:
                pickle.dump(self.onehot, f)

        data_to_load = "test" if task == "test" else "train"
        self.data = self.onehot.transform(
            pd.read_csv(f"../exam2/data/{data_to_load}_need_aggregate.csv")[
                "EventId"
            ].values.reshape(-1, 1)
        ).toarray()

        if task == "train":
            self.start_idx = 0
            self.end_idx = math.floor(self.data.shape[0] * (1 - evaluate_ratio))
        elif task == "evaluate":
            self.start_idx = (
                math.floor(self.data.shape[0] * (1 - evaluate_ratio)) - window_size
            )
            self.end_idx = self.data.shape[0]
        elif task == "test":
            self.start_idx = 0
            self.end_idx = self.data.shape[0]

    def __getitem__(self, n):
        if n < 0:
            n += self.__len__()

        x = self.start_idx + n * self.step_size
        return {
            "x": torch.tensor(self.data[x : x + self.window_size], dtype=torch.float),
            # "y": torch.tensor(self.data[x + self.window_size], dtype=torch.float),
            "y": torch.tensor((self.onehot.inverse_transform(self.data[x + self.window_size].reshape(1, -1)) - 1), dtype=torch.long),
        }

    def __len__(self):
        return (self.end_idx - self.start_idx - self.window_size) // self.step_size

    def features(self):
        return self.__getitem__(0)["x"].size(-1)
