import csv
import os
import shutil
import zipfile
from multiprocessing import Pool

import numpy as np
import pendulum as pdl

from algorithm import Detector
from plot import plot


def detect(path, detector):
    pred = []
    vals = []
    with open(path, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            val = row.get('value')
            ret = detector.fit_predict(val)
            if ret > 0:
                print(f"{pdl.now().to_datetime_string()}: File: {path} @ #{row.get('timestamp')} is anomaly!\n")
            pred.append(float(ret))
            vals.append(float(val))
    return vals, pred


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.mkdir('result')

    def f(i):
        zip_path = 'data/time-series.zip'
        archive = zipfile.ZipFile(zip_path, 'r')

        detector = Detector()

        name = f'time-series/real_{i}.csv'
        path = archive.extract(member=name)
        print(f'{pdl.now().to_datetime_string()}: working on {name}...')
        vals, pred = detect(path, detector)
        plot(name, vals, pred)
        print(f'{pdl.now().to_datetime_string()}: {name} has done!')

    with Pool(os.cpu_count() - 1) as p:
        p.map(f, [i for i in range(1, 68)])

    shutil.rmtree('time-series')
