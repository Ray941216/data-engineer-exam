import os
import csv
import shutil
import zipfile
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
            # print(val, ret)
            pred.append(float(ret))
            vals.append(float(val))
    return vals, pred


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.mkdir('result')

    zip_path = 'data/time-series.zip'
    archive = zipfile.ZipFile(zip_path, 'r')

    for i in range(1, 68):
        detector = Detector()

        name = f'time-series/real_{i}.csv'
        path = archive.extract(member=name)
        print(f'{pdl.now().to_datetime_string()}: working on {name}...')
        vals, pred = detect(path, detector)
        plot(name, vals, pred)
        print(f'{pdl.now().to_datetime_string()}: {name} has done!')

    shutil.rmtree('time-series')
