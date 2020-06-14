import random
import numpy as np
import os
from scipy.stats import iqr

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class Detector:
    def __init__(self, wait=61, sensitive=3):
        super().__init__()
        self.data = []  # store data in [n_samples, n_features]
        self.inputs = []
        self.wait = wait  # cold start waiting
        self.retrain = 16  # retrain delay time-step
        self.sensitive = sensitive  # every N anomaly should be retrained
        self.Anomaly = 0  # anomaly count
        self.sigRetrain = True  # signal of retrain

        self.ma = MA([5, 30, 60])
        self.arima = ARIMA(self.ma)
        self.ewma = EWMA([.1, .3, .5 , .7, .9])
        self.dif = DIF()

    def fit_predict(self, ptr):
        # pred = random.choices([0, 1], weights=[0.99, 0.01])[0]
        # return pred
        ptr = self.preprocess(float(ptr))
        if self.data and len(self.data) >= self.wait:
            ans = self.vote(ptr)

            self.Anomaly += ans
            if self.Anomaly >= self.sensitive:
                self.sigRetrain = True

            try:
                return ans
            except:
                pass
            finally:
                if self.sigRetrain:
                    self.train_model()

        else:
            self.data.append(ptr)
            return 0

    def train_model(self):
        # reset signal and counter
        self.sigRetrain, self.Anomaly = False, 0

        self.iforest = IsolationForest(n_estimators=len(self.data) // 10 + 120, n_jobs=os.cpu_count() - 1)
        self.ocsvm = OneClassSVM(kernel='rbf')

        
        num = len(self.data) - 1 if len(self.data) < 31 else 30
        self.lof30 = LocalOutlierFactor(n_neighbors=num, novelty=True, n_jobs=os.cpu_count() - 1)

        self.ee = EllipticEnvelope(support_fraction=1., contamination=0.25)


        # self.sscalar = StandardScaler().fit(np.array(self.data))

        # tmp = self.sscalar.transform(np.array(self.data))
        tmp = np.array(self.data)

        self.ee.fit(tmp)
        self.ocsvm.fit(tmp)

        self.lof30.fit(tmp)

        self.iforest.fit(tmp)

    def vote(self, val):
        if self.sigRetrain:
            self.train_model()

        # tmp = self.sscalar.transform([val])
        tmp = [val]
        ans = (  # -1 is anomaly and 1 is normal
                self.ee.predict(tmp) +
                self.ocsvm.predict(tmp) +
                self.lof30.predict(tmp) +
                self.iforest.predict(tmp)
        )

        for i in range(5):
            ans += self.Boxplot_Anatomy(val, idx=i)

        self.data.append(val)
        if len(self.data) % self.retrain == 0:
            self.sigRetrain = True
            self.retrain = int(len(self.data)**0.5) - 1

        return 1 if ans[0] < 0 else 0

    def Boxplot_Anatomy(self, vals, idx=0):
        upper_bound = (np.quantile(np.array(self.data).T[idx], 0.75) + 1.5 * iqr(np.array(self.data).T[idx]))
        lower_bound = (np.quantile(np.array(self.data).T[idx], 0.25) - 1.5 * iqr(np.array(self.data).T[idx]))
        return -1 if vals[idx] > upper_bound or vals[idx] < lower_bound else 1

    def preprocess(self, val):
        return [val] + self.arima.get(self.ma.get(val)) + self.dif.get(val) + self.ewma.get(val)

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

class ARIMA:
    def __init__(self, ma:MA):
        super().__init__()
        self.last = [0 for k in ma.data.keys()]

    def get(self, ma):
        ans = [m - l for m, l in zip(ma, self.last)]
        self.last = ma
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

