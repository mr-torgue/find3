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
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
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
        [Abi, 2025-09-03] Added optional config enabling:
        - AP whitelist/blacklist filtering
        - Max missing threshold for row filtering
        - Default RSSI value for missing features
        - Mean imputation toggle (stores means for classify)
        - Model filtering (select which classifiers train)
        """
        self.logger = logging.getLogger('learn.AI')
        self.naming = {'from': {}, 'to': {}}
        self.family = family
        self.path_to_data = path_to_data

        # [Abi, 2025-09-03] Config defaults (can be overridden by file or param)
        self.config = {
            "ap_whitelist": None,
            "ap_blacklist": None,
            "max_missing": 3,          # Skip rows with >3 missing RSSIs
            "default_rssi": -100.0,    # Default changed from 0 -> -100 dBm
            "use_mean_imputation": True,
            "models_enabled": None
        }

        # [Abi, 2025-09-03] Load config.json if present
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

        # Normalize lists to sets for fast checks
        if self.config["ap_whitelist"] is not None:
            self.config["ap_whitelist"] = set(self.config["ap_whitelist"])
        if self.config["ap_blacklist"] is not None:
            self.config["ap_blacklist"] = set(self.config["ap_blacklist"])

        self.mean_per_ap = None  # [Abi, 2025-09-03] Computed in learn(), reused in classify()

    def classify(self, sensor_data):
        header = self.header[1:]
        is_unknown = True

        # [Abi, 2025-09-03] Start from imputation baseline
        if self.config["use_mean_imputation"] and isinstance(self.mean_per_ap, numpy.ndarray):
            csv_data = self.mean_per_ap.copy()
        else:
            csv_data = numpy.full(len(header), self.config["default_rssi"], dtype=float)

        # Apply incoming readings
        seen = 0  # [Abi, 2025-09-03] Track how many features were actually provided
        for sensorType in sensor_data.get('s', {}):
            for sensor in sensor_data['s'][sensorType]:
                sensorName = sensorType + "-" + sensor
                if sensorName in header:
                    is_unknown = False
                    val = sensor_data['s'][sensorType][sensor]
                    try:
                        csv_data[header.index(sensorName)] = float(val)
                        seen += 1
                    except Exception:
                        self.logger.debug("Non-float sensor value {} for {}".format(val, sensorName))

        self.headerClassify = header
        self.csv_dataClassify = csv_data.reshape(1, -1)
        payload = {'location_names': self.naming['to'], 'predictions': []}

        threads = [None]*len(self.algorithms)
        self.results = [None]*len(self.algorithms)

        for i, alg in enumerate(self.algorithms.keys()):
            threads[i] = Thread(target=self.do_classification, args=(i, alg))
            threads[i].start()

        for i, _ in enumerate(self.algorithms.keys()):
            threads[i].join()

        for result in self.results:
            if result != None:
                payload['predictions'].append(result)
        payload['is_unknown'] = is_unknown

        # [Abi, 2025-09-03] Small result summary to INFO log
        # Build a compact top-1 view per model
        top1 = []
        for r in payload['predictions']:
            if r['locations'] and r['probabilities']:
                top1.append((r['name'], r['locations'][0], r['probabilities'][0]))
        self.logger.info(
            "CLASSIFY_SUMMARY seen=%d, models=%d, unknown=%s, top1=%s",
            seen, len(top1), is_unknown, top1
        )

        return payload

    def do_classification(self, index, name):
        """
        header = ['wifi-a', 'wifi-b']
        csv_data = [-67 0]
        """
        if name == 'Gaussian Process':
            return

        t = time.time()
        try:
            prediction = self.algorithms[name].predict_proba(self.csv_dataClassify)
        except Exception as e:
            logger.error(self.csv_dataClassify)
            logger.error(str(e))
            return
        predict = {}
        for i, pred in enumerate(prediction[0]):
            predict[i] = pred
        predict_payload = {'name': name,
                           'locations': [], 'probabilities': []}
        badValue = False
        for tup in sorted(predict.items(), key=operator.itemgetter(1), reverse=True):
            predict_payload['locations'].append(str(tup[0]))
            predict_payload['probabilities'].append(round(float(tup[1]), 2))
            if math.isnan(tup[1]):
                badValue = True
                break
        if badValue:
            return

        self.results[index] = predict_payload

    @timeout(10)
    def train(self, clf, x, y):
        return clf.fit(x, y)

    def _filter_header_by_ap(self, orig_header):
        """
        [Abi, 2025-09-03] Apply AP whitelist/blacklist to feature header.
        """
        ap_whitelist = self.config["ap_whitelist"]
        ap_blacklist = self.config["ap_blacklist"]

        if ap_whitelist is None and ap_blacklist is None:
            return orig_header

        filtered = []
        for ap in orig_header:
            if ap_whitelist is not None and ap not in ap_whitelist:
                continue
            if ap_blacklist is not None and ap in ap_blacklist:
                continue
            filtered.append(ap)
        return filtered

    def learn(self, fname):
        t0 = time.time()
        # load CSV file
        self.header = []
        raw_rows = []
        naming_num = 0

        # [Abi, 2025-09-03] Steps:
        # 1) Read header; filter APs via whitelist/blacklist.
        # 2) Parse rows; keep only filtered columns; mark missing as None.
        # 3) Compute per-AP mean (exclude None).
        # 4) Drop rows with too many missing features (> max_missing).
        # 5) Impute remaining missings (mean or default_rssi).
        with open(fname, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                self.logger.debug(row)
                if i == 0:
                    original_header = row  # includes class label at index 0
                    feature_header = original_header[1:]
                    filtered_features = self._filter_header_by_ap(feature_header)
                    keep_indices = [1 + feature_header.index(h) for h in filtered_features]
                    self.header = [original_header[0]] + filtered_features
                    keep_index_set = set(keep_indices)
                    self._keep_indices = keep_indices  # might be useful later
                else:
                    new_row = [None] * (1 + len(self.header[1:]))
                    # class/label
                    val = row[0]
                    if val not in self.naming['from']:
                        self.naming['from'][val] = naming_num
                        self.naming['to'][naming_num] = val
                        naming_num += 1
                    new_row[0] = self.naming['from'][val]

                    # features
                    dst_j = 1
                    for j in range(1, len(row)):
                        if j not in keep_index_set:
                            continue
                        v = row[j]
                        if v == '' or v is None:
                            new_row[dst_j] = None
                        else:
                            try:
                                new_row[dst_j] = float(v)
                            except:
                                self.logger.error("problem parsing value " + str(v))
                                new_row[dst_j] = None
                        dst_j += 1

                    raw_rows.append(new_row)

        # Compute per-AP mean (excluding None)
        num_features = len(self.header) - 1
        sums = numpy.zeros(num_features, dtype=float)
        counts = numpy.zeros(num_features, dtype=float)
        for r in raw_rows:
            feats = r[1:]
            for k, v in enumerate(feats):
                if v is not None:
                    sums[k] += v
                    counts[k] += 1.0
        mean_per_ap = numpy.zeros(num_features, dtype=float)
        default_rssi = float(self.config["default_rssi"])
        for k in range(num_features):
            if counts[k] > 0:
                mean_per_ap[k] = sums[k] / counts[k]
            else:
                mean_per_ap[k] = default_rssi  # fallback if an AP never observed

        self.mean_per_ap = mean_per_ap  # [Abi, 2025-09-03] Persist for classify()

        # Build x, y with row filtering and imputation
        max_missing = int(self.config["max_missing"])
        filtered_rows = []
        dropped_rows = 0  # [Abi, 2025-09-03] For summary
        for r in raw_rows:
            feats = r[1:]
            missing_count = sum(1 for v in feats if v is None)
            if missing_count > max_missing:
                dropped_rows += 1
                continue
            imputed = []
            for k, v in enumerate(feats):
                if v is None:
                    if self.config["use_mean_imputation"]:
                        imputed.append(mean_per_ap[k])
                    else:
                        imputed.append(default_rssi)
                else:
                    imputed.append(v)
            filtered_rows.append([r[0]] + imputed)

        if not filtered_rows:
            raise RuntimeError("After filtering, no rows remain to train on. "
                               "Consider relaxing max_missing or whitelist/blacklist.")

        # first column in row is the classification, Y
        y = numpy.zeros(len(filtered_rows))
        x = numpy.zeros((len(filtered_rows), num_features))

        # shuffle it up for training
        record_range = list(range(len(filtered_rows)))
        shuffle(record_range)
        for i in record_range:
            y[i] = filtered_rows[i][0]
            x[i, :] = numpy.array(filtered_rows[i][1:])

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            # "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA"]
        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, probability=True),
            SVC(gamma=2, C=1, probability=True),
            # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        # [Abi, 2025-09-03] Optional model filtering via config
        models_enabled = self.config["models_enabled"]
        if models_enabled is not None:
            mask = [n in models_enabled for n in names]
            names = [n for n, m in zip(names, mask) if m]
            classifiers = [c for c, m in zip(classifiers, mask) if m]

        self.algorithms = {}
        trained = []  # [Abi, 2025-09-03] For summary
        for name, clf in zip(names, classifiers):
            t2 = time.time()
            self.logger.debug("learning {}".format(name))
            try:
                self.algorithms[name] = self.train(clf, x, y)
                self.logger.debug("learned {}, {:d} ms".format(
                    name, int(1000 * (t2 - time.time()))))
                trained.append(name)
            except Exception as e:
                self.logger.error("{} {}".format(name, str(e)))

        elapsed_ms = int(1000 * (time.time() - t0))
        self.logger.debug("{:d} ms".format(elapsed_ms))

        # [Abi, 2025-09-03] Small result summary to INFO log
        self.logger.info(
            "LEARN_SUMMARY rows_total=%d, rows_used=%d, rows_dropped=%d, features=%d, models_trained=%d, models=%s, max_missing=%d, default_rssi=%.1f, mean_impute=%s, elapsed_ms=%d",
            len(raw_rows), len(filtered_rows), dropped_rows, num_features,
            len(trained), trained, max_missing, default_rssi, self.config['use_mean_imputation'],
            elapsed_ms
        )

        # [Abi, 2025-09-03] Legacy commented blocks retained intentionally.

    def save(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'wb')
        # Original order (retain for backward compatibility)
        pickle.dump(self.header, f)
        pickle.dump(self.naming, f)
        pickle.dump(self.algorithms, f)
        pickle.dump(self.family, f)
        # [Abi, 2025-09-03] Persist imputation means and config
        try:
            pickle.dump(self.mean_per_ap, f)
            pickle.dump(self.config, f)
        except Exception as e:
            self.logger.error("Optional extras not saved: {}".format(e))
        f.close()
        elapsed_ms = int(1000 * (time.time() - t))
        self.logger.debug("{:d} ms".format(elapsed_ms))
        # [Abi, 2025-09-03] Small result summary
        self.logger.info("SAVE_SUMMARY file=%s, features=%d, models=%d, elapsed_ms=%d",
                         save_file, len(self.header) - 1 if self.header else -1,
                         len(self.algorithms) if hasattr(self, 'algorithms') else -1,
                         elapsed_ms)

    def load(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'rb')
        self.header = pickle.load(f)
        self.naming = pickle.load(f)
        self.algorithms = pickle.load(f)
        self.family = pickle.load(f)
        # [Abi, 2025-09-03] Try to load new fields if present
        try:
            self.mean_per_ap = pickle.load(f)
            self.config = pickle.load(f)
            if self.config.get("ap_whitelist") is not None and not isinstance(self.config["ap_whitelist"], set):
                self.config["ap_whitelist"] = set(self.config["ap_whitelist"])
            if self.config.get("ap_blacklist") is not None and not isinstance(self.config["ap_blacklist"], set):
                self.config["ap_blacklist"] = set(self.config["ap_blacklist"])
        except Exception:
            if not hasattr(self, "mean_per_ap"):
                self.mean_per_ap = None
        f.close()
        elapsed_ms = int(1000 * (time.time() - t))
        self.logger.debug("{:d} ms".format(elapsed_ms))
        # [Abi, 2025-09-03] Small result summary
        self.logger.info("LOAD_SUMMARY file=%s, features=%d, models=%d, elapsed_ms=%d",
                         save_file, len(self.header) - 1 if self.header else -1,
                         len(self.algorithms) if hasattr(self, 'algorithms') else -1,
                         elapsed_ms)


def do():
    # NOTE: This demo function is unchanged to preserve original structure.
    ai = AI()
    ai.load()
    # ai.learn()
    params = {'quantile': .3,
              'eps': .3,
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': 3}
    bandwidth = cluster.estimate_bandwidth(ai.x, quantile=params['quantile'])
    connectivity = kneighbors_graph(
        ai.x, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')
    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            try:
                algorithm.fit(ai.x)
            except:
                continue

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(numpy.int)
        else:
            y_pred = algorithm.predict(ai.x)
        if max(y_pred) > 3:
            continue
        known_groups = {}
        for i, group in enumerate(ai.y):
            group = int(group)
            if group not in known_groups:
                known_groups[group] = []
            known_groups[group].append(i)
        guessed_groups = {}
        for i, group in enumerate(y_pred):
            if group not in guessed_groups:
                guessed_groups[group] = []
            guessed_groups[group].append(i)
        for k in known_groups:
            for g in guessed_groups:
                print(
                    k, g, len(set(known_groups[k]).intersection(guessed_groups[g])))
