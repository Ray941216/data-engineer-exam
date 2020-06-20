import math
import os
import random

import numpy as np
from scipy.stats import iqr
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import OneClassSVM


class Detector:
    def __init__(self, wait=61*2, sensitive=3, ignore_continuous=10, max_window=61 * 5):
        super().__init__()
        self.data = []  # store data in [n_samples, n_features]
        self.inputs = []
        self.wait = int(wait)  # cold start waiting
        self.max_window = int(max_window)  # max data for training
        self.retrain = 16  # retrain delay time-step
        self.sensitive = sensitive  # every N anomaly should be retrained
        self.Anomaly = 0  # anomaly count
        self.sigRetrain = True  # signal of retrain

        self.ignore_continuous = ignore_continuous  # anomaly alert every N ticks
        self.continuous = 0  # counter for counting alert delay
        self.cont = False  # Anomaly continue state
        self.anomaly_cont_acc = 0  # Anomaly continue counter

        self.ma = MA(list(range(3, 32, 2)) + [61, 121])
        self.madiff = MADIFF(self.ma)
        self.ewma = EWMA([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.ewmadiff = EWMADIFF(self.ewma)
        self.dif = DIF()

    def fit_predict(self, ptr):
        self.continuous = (
            self.continuous
            if self.continuous == 0
            else 0
            if self.continuous + 1 > self.ignore_continuous
            else self.continuous + 1
        )
        self.inputs.append(float(ptr))

        ptr = self.preprocess(float(ptr))
        if self.data and len(self.data) >= self.wait:
            ans = self.vote(ptr)

            try:
                if ans == 1 and self.continuous == 0:
                    self.continuous += 1
                    self.anomaly_cont_acc += 1
                    return ans

                elif ans == 1 and self.continuous > 0:
                    self.cont = True
                    self.anomaly_cont_acc += 1
                    return 0

                else:
                    if self.cont:
                        self.cont = False
                        self.ignore_continuous = math.ceil(
                            self.ignore_continuous * 0.6 + self.anomaly_cont_acc * 0.4
                        )
                        self.sensitive = math.ceil(
                            self.sensitive * 0.8 + self.anomaly_cont_acc * 0.2
                        )
                        self.anomaly_cont_acc = 0
                        self.continuous = 0

                    return ans
            except:
                pass
            finally:
                if len(self.data) == self.max_window:
                    _ = self.data.pop(0)

                self.Anomaly += ans
                if self.Anomaly >= self.sensitive:
                    self.sigRetrain = True

                if self.sigRetrain:
                    self.train_model()

        else:
            self.data.append(ptr)
            return 0

    def train_model(self):
        # reset signal and counter
        self.sigRetrain, self.Anomaly = False, 0

        self.iforest = IsolationForest(
            n_estimators=math.ceil(np.mean(self.ma.periods)) * len(self.data[-1]) // 10
            + 120,
            # n_jobs=os.cpu_count() - 1,
        )
        self.ocsvm = OneClassSVM(kernel="rbf")

        # num = len(self.data) - 1 if len(self.data) < 31 else 30
        self.lof = LocalOutlierFactor(
            n_neighbors=math.ceil(np.mean(self.ma.periods)),
            novelty=True,
            # n_jobs=os.cpu_count() - 1,
        )

        self.ee = EllipticEnvelope(support_fraction=1.0, contamination=0.25)

        # self.sscalar = StandardScaler().fit(np.array(self.data))
        # tmp = self.sscalar.transform(np.array(self.data))

        tmp = np.array(self.data)

        self.ee.fit(tmp)
        self.ocsvm.fit(tmp)
        self.lof.fit(tmp)
        self.iforest.fit(tmp)

    def vote(self, val):
        if self.sigRetrain:
            self.train_model()

        # tmp = self.sscalar.transform([val])
        tmp = [val]
        ans = (  # -1 is anomaly and 1 is normal
            self.ee.predict(tmp)
            + self.ocsvm.predict(tmp)
            + self.lof.predict(tmp)
            + self.iforest.predict(tmp)
        )

        for i in range(len(self.ma.data.keys()) + len(self.ewma.data.keys()) + 1):
            ans += self.Boxplot_Anatomy(val, idx=i)

        self.data.append(val)
        if len(self.data) % self.retrain == 0:
            self.sigRetrain = True
            self.retrain = int(len(self.data) ** 0.5) - 1

        return 1 if ans[0] < 0 else 0

    def Boxplot_Anatomy(self, vals, idx=0):
        upper_bound = np.quantile(np.array(self.data).T[idx], 0.75) + 1.5 * iqr(
            np.array(self.data).T[idx]
        )
        lower_bound = np.quantile(np.array(self.data).T[idx], 0.25) - 1.5 * iqr(
            np.array(self.data).T[idx]
        )
        return -1 if vals[idx] > upper_bound or vals[idx] < lower_bound else 1

    def preprocess(self, val):
        ma = self.ma.get(val)
        ewma = self.ewma.get(val)
        return (
            [val]
            + ma
            + ewma
            + self.dif.get(val)
            + self.madiff.get(ma)
            + self.ewmadiff.get(ewma)
        )


class MA:
    def __init__(self, period):
        super().__init__()
        if period:
            if isinstance(period, int):
                self.periods = period
                self.data = {period: []}
            elif isinstance(period, list):
                self.periods = period
                self.data = {x: [] for x in self.periods}
            else:
                raise TypeError()
        else:
            raise TypeError()

    def get(self, val):
        rt = []
        for k, v in self.data.items():
            if len(v) == k:
                _ = v.pop(0)

            self.data[k].append(val)
            rt.append(np.mean(self.data[k]))

        return rt


class MADIFF:
    def __init__(self, ma: MA):
        super().__init__()
        self.difs = [DIF() for k in ma.data.keys()]

    def get(self, ma):
        ans = []
        for m, d in zip(ma, self.difs):
            ans += d.get(m)
        return ans


class EWMA:
    def __init__(self, alpha):
        super().__init__()
        if alpha:
            if isinstance(alpha, int):
                self.alphas = alpha
                self.data = {alpha: None}
            elif isinstance(alpha, list):
                self.alphas = alpha
                self.data = {x: None for x in self.alphas}
        else:
            raise TypeError()

    def get(self, val):
        rt = []
        for k, v in self.data.items():
            if v is None:
                v = val

            rt.append(val * k + (1 - k) * v)
            self.data[k] = rt[-1]

        return rt


class EWMADIFF:
    def __init__(self, ewma: EWMA):
        super().__init__()
        self.difs = [DIF() for k in ewma.data.keys()]

    def get(self, ewma):
        ans = []
        for m, d in zip(ewma, self.difs):
            ans += d.get(m)
        return ans


class DIF:
    def __init__(self):
        super().__init__()
        self.last = 0

    def get(self, val):
        try:
            return [val - self.last]
        except:
            pass
        finally:
            self.last = val
