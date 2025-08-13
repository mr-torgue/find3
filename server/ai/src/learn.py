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
import os

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

# =============================================================================
# RSSI constants & helpers
# -----------------------------------------------------------------------------
# [OLD] (none)
# [Abhi | 2025-08-13] Add consistent RSSI handling constants + scaler
ABSENT_RSSI = -100.0
CLIP_MIN, CLIP_MAX = -95.0, -30.0

def clip_scale_array(a):
    """Clip RSSI to [CLIP_MIN, CLIP_MAX] and scale to 0..1."""
    a = numpy.clip(a, CLIP_MIN, CLIP_MAX)
    return (a - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
# =============================================================================


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

    def __init__(self, family=None, path_to_data=None):  # [Abhi | 2025-08-13] keep defaults to avoid caller errors
        self.logger = logging.getLogger('learn.AI')
        self.naming = {'from': {}, 'to': {}}
        self.family = family
        self.path_to_data = path_to_data
        # [Abhi | 2025-08-13] will be set during learn()
        self.global_means = None


    ''' changes Folmer 14-05-2025
    Includes the following:
    1. default value  (specified by self.default_value)
    2. ignore data that see less than x percent of the access points (specified by self.threshold)
    '''
    def classify(self, sensor_data):

        # try to load, set to default values otherwise
        try:
            default_value = self.default_value
            threshold = self.threshold
        except:
            # default_value = 0                # [OLD]
            # [Abhi | 2025-08-13] Make sane fallbacks
            default_value = ABSENT_RSSI
            threshold = 0.6

        # header = self.header[1:]             # [OLD]
        header = self.header[1:]  # [Abhi | 2025-08-13] same

        # is_unknown = True                     # [OLD]
        is_unknown = True                      # [Abhi | 2025-08-13] same

        # csv_data = numpy.full(len(header), default_value)     # [OLD]
        # for sensorType in sensor_data['s']:
        #     for sensor in sensor_data['s'][sensorType]:
        #         sensorName = sensorType + "-" + sensor
        #         if sensorName in header:
        #             is_unknown = False
        #             csv_data[header.index(sensorName)] = sensor_data['s'][sensorType][sensor]
        # self.headerClassify = header
        # self.csv_dataClassify = csv_data.reshape(1, -1)

        # [Abhi | 2025-08-13] Consistent inference vector: start at ABSENT_RSSI, fill, gate, impute, normalize
        x_vec = numpy.full(len(header), ABSENT_RSSI, dtype=float)
        for sensorType in sensor_data.get('s', {}):
            for sensor in sensor_data['s'][sensorType]:
                sensorName = sensorType + "-" + sensor
                if sensorName in header:
                    is_unknown = False
                    try:
                        x_vec[header.index(sensorName)] = float(sensor_data['s'][sensorType][sensor])
                    except Exception:
                        pass

        # [Abhi | 2025-08-13] Hard missing gate: skip if too many missing
        num_absent = int(numpy.sum(x_vec <= ABSENT_RSSI))

        # [Abhi | 2025-08-13 v2] Debug feature/missing info
        self.logger.debug(
            "Classify: features=%d, num_absent=%d (threshold=%.2f), saw_any=%s",
            len(header), num_absent, threshold, str(not is_unknown)
        )

        if num_absent > 3:
            payload = {'location_names': self.naming['to'], 'predictions': [], 'is_unknown': True}
            return payload

        # [Abhi | 2025-08-13] Impute remaining missings with global means learned during train()
        if getattr(self, 'global_means', None) is not None:
            mask_absent = (x_vec <= ABSENT_RSSI)
            x_vec[mask_absent] = self.global_means[mask_absent]

        # [Abhi | 2025-08-13] Clip + normalize to 0..1
        x_vec = clip_scale_array(x_vec)

        self.headerClassify = header
        self.csv_dataClassify = x_vec.reshape(1, -1)

        self.logger.debug("Using %d features to classify!", len(header))
        payload = {'location_names': self.naming['to'], 'predictions': []}

        # check if most values have been set (is it len(header) or len(header) - 1)
        # if(sum(d == default_value for d in csv_data) / len(header) < threshold):   # [OLD]
        # [Abhi | 2025-08-13] We already gated on missing count; keep threshold logic using ABSENT_RSSI basis
        if (num_absent / max(1, len(header))) < threshold:
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
            prediction = self.algorithms[
                name].predict_proba(self.csv_dataClassify)
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
            predict_payload['probabilities'].append(
                round(float(tup[1]), 2))
            if math.isnan(tup[1]):
                badValue = True
                break
        if badValue:
            return

        self.results[index] = predict_payload

    @timeout(10)
    def train(self, clf, x, y):
        return clf.fit(x, y)

    '''
    modifications Folmer 13-09-2024:
    Added support for whitelists and blacklists.
    Allows the end-user to select or exclude certain access points based on MAC address.
    Whitelist takes precedence over blacklist (usually you only select one of them).

    Modifications 13-05-2025:
    Added some more settings. 
    Allowed for model selection. 
    Also added a default value. Default value of 0 might be problematic, since it indicates a really strong connection (RSSI)
    Threshold can be specified, so that certain requests with insufficient data are not classified (prevents weird classifications)

    Format:
        {
            "whitelist": 
            [
                "wifi-38:91:b7:1a:22:ec",
                "wifi-38:91:b7:1a:22:e2",
                ...
            ],
            "blacklist":
            [
                "wifi-38:91:b7:1a:22:ec",
                ...
            ],
            "models":
            [
                "Nearest Neighbors",
                "Linear SVM",
                "RBF SVM",
                # "Gaussian Process",
                "Decision Tree",
                "Random Forest",
                "Neural Net",
                "AdaBoost",
                "Naive Bayes",
                "QDA"
            ],
            "default": 100,
            "threshold": 0.6
        }
    '''
    def learn(self, fname):
        t = time.time()
        # load CSV file
        rows = []
        naming_num = 0

        with open(fname, 'r') as csvfile:
            # if file does not exist, simply ignore it
            jsonfname = 'settings.json'
            try:
                settings = json.load(open(jsonfname))
            except Exception as e:
                self.logger.error("Could not load json settings file: %s\nCurrent working directory: %s" % (e, os.getcwd()))
                settings = {}  # [Abhi | 2025-08-13] ensure dict

            # set default value
            # try:
            #     self.default_value = float(settings["default"])
            # except:
            #     self.default_value = 0
            # [Abhi | 2025-08-13] Use ABSENT_RSSI as default when not provided
            try:
                self.default_value = float(settings["default"])
            except:
                self.default_value = ABSENT_RSSI

            # set threshold
            try:
                self.threshold = float(settings["threshold"])
            except:
                self.threshold = 0.6

            # always include the location
            reader = csv.reader(csvfile, delimiter=',')
            fullheader = next(reader)
            columns = [0]
            self.header = ['location']

            # check which columns to include and build a new header
            # for i, column in enumerate(fullheader):
            #     # try if whitelist if available
            #     try:
            #         if column in settings["whitelist"]:
            #             columns.append(i)
            #             self.header.append(column)
            #             continue  # [Abhi | 2025-08-13] skip blacklist if whitelisted
            #     except:
            #         pass
            #     # if not, try to use blacklist
            #     try:
            #         if "blacklist" in settings and column in settings["blacklist"]:
            #             continue
            #     except:
            #         pass
            #     # no white- or blacklist include it
            #     columns.append(i)
            #     self.header.append(column)

            # [Abhi | 2025-08-13 v2] Fixed header duplication: never add the label col again
            for i, column in enumerate(fullheader):
                if i == 0:
                    continue  # never add 'location' as a feature
                # whitelist first (if provided and non-empty)
                try:
                    if "whitelist" in settings and settings["whitelist"]:
                        if column in settings["whitelist"]:
                            columns.append(i)
                            self.header.append(column)
                        # if whitelist active, skip non-listed columns
                        continue
                except Exception:
                    pass
                # blacklist (optional)
                try:
                    if "blacklist" in settings and column in settings["blacklist"]:
                        continue
                except Exception:
                    pass
                # include by default
                columns.append(i)
                self.header.append(column)

            self.logger.debug("Using %d features for the AI: %s" % (len(self.header), self.header))

            # [Abhi | 2025-08-13 v2] Safety fallback if no AP features selected
            if len(self.header) <= 1:
                self.logger.error("No AP features selected (whitelist/blacklist). Falling back to ALL non-location columns.")
                self.header = ['location'] + [c for c in fullheader if c != 'location']
                columns = [0] + [idx for idx, c in enumerate(fullheader) if c != 'location']
                self.logger.debug("Fallback header: %s", self.header)
            
            count_all = 0
            count_skipped = 0
            for i, row in enumerate(reader):
                count_all += 1
                new_row = []
                for j in columns:
                    val = row[j]
                    if j == 0:
                        # this is a name of the location
                        # if val not in self.naming['from']:                   # [OLD]
                        #     self.naming['from'][val] = naming_num
                        #     self.naming['to'][naming_num] = val
                        #     naming_num += 1
                        #     new_row.append(self.naming['from'][val])
                        #     continue
                        # [Abhi | 2025-08-13] normalize labels (e.g., a01->A01)
                        val = val.strip().upper()
                        if val not in self.naming['from']:
                            self.naming['from'][val] = naming_num
                            self.naming['to'][naming_num] = val
                            naming_num += 1
                        new_row.append(self.naming['from'][val])
                        continue
                    if val == '':
                        new_row.append(self.default_value)
                        continue
                    try:
                        new_row.append(float(val))
                    except:
                        self.logger.error(
                            "problem parsing value " + str(val))
                if(len(new_row) != len(self.header)):
                    self.logger.error("Row size(%d) should be the same as header size(%d)" % (len(new_row), len(self.header)))
                # if(sum(d == self.default_value for d in new_row) / (len(self.header) - 1) < self.threshold):  # [OLD]
                #     rows.append(new_row)
                # else:
                #     count_skipped += 1
                # [Abhi | 2025-08-13] Use ABSENT_RSSI gate; keep same semantics
                if (sum(d <= ABSENT_RSSI for d in new_row[1:]) / (len(self.header) - 1)) < self.threshold:
                    rows.append(new_row)
                else:
                    count_skipped += 1
                #self.logger.debug("row %d: %s" % (i, new_row))
        self.logger.debug("Total rows: %d, skipped %d" % (count_all, count_skipped))

        # first column in row is the classification, Y
        y = numpy.zeros(len(rows))
        x = numpy.zeros((len(rows), len(rows[0]) - 1))

        # shuffle it up for training
        record_range = list(range(len(rows)))
        shuffle(record_range)
        for i in record_range:
            y[i] = rows[i][0]
            x[i, :] = numpy.array(rows[i][1:])
        
        # try:
        #     if settings["mode"] == "mean":
        #         x = numpy.where(x == self.default_value, numpy.nan, x)
        #         
        #         rooms = numpy.unique(y)
        #         for room in rooms:
        #             mask = y == room
        #             x_room = x[mask]
        #             if numpy.isnan(x_room).all():   
        #                 x[mask] = self.default_value
        #             else:
        #                 room_means = numpy.nanmean(x_room, axis=0)
        #                 x[mask] = numpy.where(numpy.isnan(x_room), room_means, x_room)
        #                 self.logger.debug("Mean per column: %s for room %d" % (room_means, room))
        #
        # except Exception as e:
        #     print("An exception occurred: %s" % (e))
        # self.logger.debug("x: %s\ncontains nan: %s" % (x, numpy.isnan(x).any()))

        # [Abhi | 2025-08-13] Unified imputation + normalization (room means -> global means -> clip/scale)
        try:
            # 1) mark absents as NaN for stats
            x_nan = numpy.where(x <= ABSENT_RSSI, numpy.nan, x)

            # 2) optional per-room mean imputation
            use_room_mean = False
            try:
                use_room_mean = (settings.get("mode", "").lower() == "mean")
            except Exception:
                pass

            if use_room_mean:
                rooms = numpy.unique(y)
                for room in rooms:
                    mask = (y == room)
                    x_room = x_nan[mask]
                    if numpy.isnan(x_room).all():
                        continue
                    room_means = numpy.nanmean(x_room, axis=0)
                    x_nan[mask] = numpy.where(numpy.isnan(x_room), room_means, x_room)
                    self.logger.debug("Room %s mean imputed.", str(room))

            # 3) global means across all rows
            global_means = numpy.nanmean(x_nan, axis=0)
            global_means = numpy.where(numpy.isnan(global_means), ABSENT_RSSI, global_means)

            # 4) fill remaining NaNs with global means
            x_filled = numpy.where(numpy.isnan(x_nan), global_means, x_nan)

            # 5) clip + normalize to 0..1
            x_norm = clip_scale_array(x_filled)

            # 6) stash for inference
            self.global_means = global_means

            self.logger.debug("Training matrix normalized. Any NaN left? %s", numpy.isnan(x_norm).any())
            x = x_norm

        except Exception as e:
            self.logger.error("Imputation/normalization error: %s", e)
        
        names = []
        classifiers = []

        try:
            if "Nearest Neighbors" in settings.get("models", []):
                names.append("Nearest Neighbors")
                # classifiers.append(KNeighborsClassifier(3))  # [OLD]
                classifiers.append(KNeighborsClassifier(n_neighbors=5, weights='distance'))  # [Abhi | 2025-08-13]
            if "Linear SVM" in settings.get("models", []):
                names.append("Linear SVM")
                classifiers.append(SVC(kernel="linear", C=0.025, probability=True))
            if "RBF SVM" in settings.get("models", []):
                names.append("RBF SVM")
                classifiers.append(SVC(gamma=2, C=1, probability=True))
            if "Decision Tree" in settings.get("models", []):
                names.append("Decision Tree")
                classifiers.append(DecisionTreeClassifier(max_depth=5))
            if "Random Forest" in settings.get("models", []):
                names.append("Random Forest")
                # classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))  # [OLD]
                classifiers.append(RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=42))  # [Abhi | 2025-08-13]
            if "Neural Net" in settings.get("models", []):
                names.append("Neural Net")
                classifiers.append(MLPClassifier(alpha=1))
            if "AdaBoost" in settings.get("models", []):
                names.append("AdaBoost")
                classifiers.append(AdaBoostClassifier())
            if "Naive Bayes" in settings.get("models", []):
                names.append("Naive Bayes")
                classifiers.append(GaussianNB())
            if "QDA" in settings.get("models", []):
                names.append("QDA")
                classifiers.append(QuadraticDiscriminantAnalysis())
        except:
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
                # KNeighborsClassifier(3),  # [OLD]
                KNeighborsClassifier(n_neighbors=5, weights='distance'),  # [Abhi | 2025-08-13]
                SVC(kernel="linear", C=0.025, probability=True),
                SVC(gamma=2, C=1, probability=True),
                # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                DecisionTreeClassifier(max_depth=5),
                # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # [OLD]
                RandomForestClassifier(n_estimators=300, max_features='sqrt', min_samples_leaf=3, n_jobs=-1, random_state=42),  # [Abhi | 2025-08-13]
                MLPClassifier(alpha=1),
                AdaBoostClassifier(),
                GaussianNB(),
                QuadraticDiscriminantAnalysis()]

        self.logger.debug("Using %d models: %s" % (len(names), names))
        self.algorithms = {}
        # split_for_learning = int(0.70 * len(y))
        for name, clf in zip(names, classifiers):
            t2 = time.time()
            self.logger.debug("learning {}".format(name))
            try:
                self.algorithms[name] = self.train(clf, x, y)
                # score = self.algorithms[name].score(x,y)
                # logger.debug(name, score)
                self.logger.debug("learned {}, {:d} ms".format(
                    name, int(1000 * (t2 - time.time()))))
            except Exception as e:
                self.logger.error("{} {}".format(name, str(e)))

        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def save(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'wb')
        pickle.dump(self.header, f)
        pickle.dump(self.naming, f)
        pickle.dump(self.algorithms, f)
        pickle.dump(self.family, f)
        # [Abhi | 2025-08-13] persist global means for classify-time imputation
        try:
            pickle.dump(self.global_means, f)
        except Exception:
            pickle.dump(None, f)
        f.close()
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def load(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'rb')
        self.header = pickle.load(f)
        self.naming = pickle.load(f)
        self.algorithms = pickle.load(f)
        self.family = pickle.load(f)
        # [Abhi | 2025-08-13] load global means if present (backward compatible)
        try:
            self.global_means = pickle.load(f)
        except Exception:
            self.global_means = None
        f.close()
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))


def do():
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


# ai = AI()
# ai.learn("../testing/testdb.csv")
# ai.save("dGVzdGRi.find3.ai")
# ai.load("dGVzdGRi.find3.ai")
# a = json.load(open('../testing/testdb_single_rec.json'))
# classified = ai.classify(a)
# print(json.dumps(classified,indent=2))
