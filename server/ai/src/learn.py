#!/usr/bin/python3

import json
import csv
from random import shuffle
import warnings
import pickle
import gzip
import operator
import time
import logging
import math
from threading import Thread
import functools
import multiprocessing
import os  # [Abi, 2025-09-03] For config file path handling

# create logger with 'spam_application'
logger = logging.getLogger('learn')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('learn.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - [%(name)s/%(funcName)s] - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from naive_bayes import ExtendedNaiveBayes
from naive_bayes2 import ExtendedNaiveBayes2


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (
                func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


class AI(object):

    def __init__(self, family=None, path_to_data=None, config=None):
        """
        [Abi, 2025-09-03] Config for:
        - AP whitelist/blacklist
        - Max missing threshold
        - Default RSSI value
        - Mean imputation toggle
        - Model filtering
        """
        self.logger = logging.getLogger('learn.AI')
        self.naming = {'from': {}, 'to': {}}
        self.family = family
        self.path_to_data = path_to_data

        self.config = {
            "ap_whitelist": None,
            "ap_blacklist": None,
            "max_missing": 3,
            "default_rssi": -100.0,
            "use_mean_imputation": True,
            "models_enabled": None
        }

        file_cfg = {}
        if self.path_to_data:
            cfg_path = os.path.join(self.path_to_data, "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        file_cfg = json.load(f)
                except Exception as e:
                    self.logger.error("Failed to load config.json: {}".format(e))
        if file_cfg:
            self.config.update({k: v for k, v in file_cfg.items() if v is not None})
        if config:
            self.config.update({k: v for k, v in config.items() if v is not None})

        if self.config["ap_whitelist"] is not None:
            self.config["ap_whitelist"] = set(self.config["ap_whitelist"])
        if self.config["ap_blacklist"] is not None:
            self.config["ap_blacklist"] = set(self.config["ap_blacklist"])

        self.mean_per_ap = None  # [Abi]

    def classify(self, sensor_data):
        header = self.header[1:]
        is_unknown = True
        if self.config["use_mean_imputation"] and isinstance(self.mean_per_ap, numpy.ndarray):
            csv_data = self.mean_per_ap.copy()
        else:
            csv_data = numpy.full(len(header), self.config["default_rssi"], dtype=float)

        seen = 0
        for sensorType in sensor_data.get('s', {}):
            for sensor in sensor_data['s'][sensorType]:
                sensorName = sensorType + "-" + sensor
                if sensorName in header:
                    is_unknown = False
                    try:
                        csv_data[header.index(sensorName)] = float(sensor_data['s'][sensorType][sensor])
                        seen += 1
                    except Exception:
                        self.logger.debug("Bad value for {}".format(sensorName))

        self.headerClassify = header
        self.csv_dataClassify = csv_data.reshape(1, -1)
        payload = {'location_names': self.naming['to'], 'predictions': []}

        if not getattr(self, "algorithms", None):
            self.logger.warning("[Abi] classify() called but no models are trained.")
            self.logger.info("CLASSIFY_SUMMARY seen=%d, models=0, unknown=%s, top1=%s",
                             seen, is_unknown, [])
            payload['is_unknown'] = is_unknown
            return payload

        threads = [None]*len(self.algorithms)
        self.results = [None]*len(self.algorithms)
        for i, alg in enumerate(self.algorithms.keys()):
            threads[i] = Thread(target=self.do_classification, args=(i, alg))
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()
        for result in self.results:
            if result is not None:
                payload['predictions'].append(result)

        payload['is_unknown'] = is_unknown
        top1 = []
        for r in payload['predictions']:
            if r['locations'] and r['probabilities']:
                top1.append((r['name'], r['locations'][0], r['probabilities'][0]))
        self.logger.info("CLASSIFY_SUMMARY seen=%d, models=%d, unknown=%s, top1=%s",
                         seen, len(top1), is_unknown, top1)
        return payload

    def do_classification(self, index, name):
        try:
            prediction = self.algorithms[name].predict_proba(self.csv_dataClassify)
        except Exception as e:
            logger.error(str(e))
            return
        predict_payload = {'name': name, 'locations': [], 'probabilities': []}
        for i, pred in sorted(enumerate(prediction[0]), key=lambda x: x[1], reverse=True):
            predict_payload['locations'].append(str(i))
            predict_payload['probabilities'].append(round(float(pred), 2))
        self.results[index] = predict_payload

    @timeout(10)
    def train(self, clf, x, y):
        return clf.fit(x, y)

    def _filter_header_by_ap(self, orig_header):
        ap_whitelist = self.config["ap_whitelist"]
        ap_blacklist = self.config["ap_blacklist"]
        if ap_whitelist is None and ap_blacklist is None:
            return orig_header
        return [ap for ap in orig_header
                if (ap_whitelist is None or ap in ap_whitelist) and
                   (ap_blacklist is None or ap not in ap_blacklist)]

    def learn(self, fname):
        t0 = time.time()
        self.header = []
        raw_rows = []
        naming_num = 0

        with open(fname, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    feature_header = row[1:]
                    filtered_features = self._filter_header_by_ap(feature_header)
                    keep_indices = [1 + feature_header.index(h) for h in filtered_features]
                    self.header = [row[0]] + filtered_features
                    keep_index_set = set(keep_indices)
                else:
                    new_row = [None] * (1 + len(self.header[1:]))
                    val = row[0]
                    if val not in self.naming['from']:
                        self.naming['from'][val] = naming_num
                        self.naming['to'][naming_num] = val
                        naming_num += 1
                    new_row[0] = self.naming['from'][val]
                    dst_j = 1
                    for j in range(1, len(row)):
                        if j not in keep_index_set:
                            continue
                        try:
                            new_row[dst_j] = float(row[j]) if row[j] else None
                        except:
                            new_row[dst_j] = None
                        dst_j += 1
                    raw_rows.append(new_row)

        num_features = len(self.header) - 1
        sums = numpy.zeros(num_features)
        counts = numpy.zeros(num_features)
        for r in raw_rows:
            for k, v in enumerate(r[1:]):
                if v is not None:
                    sums[k] += v
                    counts[k] += 1
        mean_per_ap = numpy.where(counts > 0, sums / counts, self.config["default_rssi"])
        self.mean_per_ap = mean_per_ap

        max_missing = int(self.config["max_missing"])
        filtered_rows, dropped_rows = [], 0
        for r in raw_rows:
            feats = r[1:]
            if sum(1 for v in feats if v is None) > max_missing:
                dropped_rows += 1
                continue
            imputed = [mean_per_ap[k] if v is None and self.config["use_mean_imputation"]
                       else (self.config["default_rssi"] if v is None else v)
                       for k, v in enumerate(feats)]
            filtered_rows.append([r[0]] + imputed)

        if not filtered_rows:
            self.algorithms = {}
            elapsed_ms = int(1000 * (time.time() - t0))
            self.logger.warning("[Abi] No rows remained after filtering. Training skipped.")
            self.logger.info("LEARN_SUMMARY rows_total=%d, rows_used=0, rows_dropped=%d, features=%d, models_trained=0, models=[], elapsed_ms=%d",
                             len(raw_rows), dropped_rows, num_features, elapsed_ms)
            return

        y = numpy.array([r[0] for r in filtered_rows])
        x = numpy.array([r[1:] for r in filtered_rows])

        names = ["Nearest Neighbors","Linear SVM","RBF SVM","Decision Tree",
                 "Random Forest","Neural Net","AdaBoost","Naive Bayes","QDA"]
        classifiers = [KNeighborsClassifier(3),
                       SVC(kernel="linear", C=0.025, probability=True),
                       SVC(gamma=2, C=1, probability=True),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                       MLPClassifier(alpha=1),
                       AdaBoostClassifier(),
                       GaussianNB(),
                       QuadraticDiscriminantAnalysis()]

        if self.config["models_enabled"] is not None:
            mask = [n in self.config["models_enabled"] for n in names]
            names = [n for n, m in zip(names, mask) if m]
            classifiers = [c for c, m in zip(classifiers, mask) if m]

        self.algorithms, trained = {}, []
        for name, clf in zip(names, classifiers):
            try:
                self.algorithms[name] = self.train(clf, x, y)
                trained.append(name)
            except Exception as e:
                self.logger.error("{} {}".format(name, str(e)))

        elapsed_ms = int(1000 * (time.time() - t0))
        self.logger.info("LEARN_SUMMARY rows_total=%d, rows_used=%d, rows_dropped=%d, features=%d, models_trained=%d, models=%s, elapsed_ms=%d",
                         len(raw_rows), len(filtered_rows), dropped_rows, num_features,
                         len(trained), trained, elapsed_ms)

    def save(self, save_file):
        f = gzip.open(save_file, 'wb')
        pickle.dump(self.header, f)
        pickle.dump(self.naming, f)
        pickle.dump(self.algorithms, f)
        pickle.dump(self.family, f)
        pickle.dump(self.mean_per_ap, f)
        pickle.dump(self.config, f)
        f.close()
        self.logger.info("SAVE_SUMMARY file=%s, features=%d, models=%d",
                         save_file, len(self.header)-1, len(self.algorithms))

    def load(self, save_file):
        f = gzip.open(save_file, 'rb')
        self.header = pickle.load(f)
        self.naming = pickle.load(f)
        self.algorithms = pickle.load(f)
        self.family = pickle.load(f)
        try:
            self.mean_per_ap = pickle.load(f)
            self.config = pickle.load(f)
        except Exception:
            self.mean_per_ap = None
        f.close()
        self.logger.info("LOAD_SUMMARY file=%s, features=%d, models=%d",
                         save_file, len(self.header)-1, len(self.algorithms))


def do():
    ai = AI()
    ai.load()
    # unchanged clustering demo
