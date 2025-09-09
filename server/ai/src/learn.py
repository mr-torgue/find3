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

    def __init__(self, family, path_to_data):
        self.logger = logging.getLogger('learn.AI')
        self.naming = {'from': {}, 'to': {}}
        self.family = family
        self.path_to_data = path_to_data


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
            default_value = 0
            threshold = 0.6

        header = self.header[1:]
        is_unknown = True
        csv_data = numpy.full(len(header), default_value)
        for sensorType in sensor_data['s']:
            for sensor in sensor_data['s'][sensorType]:
                sensorName = sensorType + "-" + sensor
                if sensorName in header:
                    is_unknown = False
                    csv_data[header.index(sensorName)] = sensor_data['s'][sensorType][sensor]
        
        self.headerClassify = header
        self.csv_dataClassify = csv_data.reshape(1, -1)
         # self.csv_dataClassify = csv_data.reshape(1, -1)  # [Abhishek | 09-07-2025] original line commented
        
        '''
        # === [Abhishek | 09-07-2025] Apply missing value filter and mean imputation ===
        x_vec = csv_data
        num_defaults = numpy.count_nonzero(x_vec == default_value)
        if num_defaults > 3:
            self.logger.warning("Skipping classification: too many missing values (%d)" % num_defaults)
            payload['is_unknown'] = True
            return payload

        # Replace default (-100) values with mean per AP
        x_vec = numpy.where(x_vec == default_value, self.mean_per_ap, x_vec)
        self.csv_dataClassify = x_vec.reshape(1, -1)
        # === End of filter logic ===
        '''

        self.logger.debug("Using %d features to classify!" % len(header))
        payload = {'location_names': self.naming['to'], 'predictions': []}
        # check if most values have been set (is it len(header) or len(header) - 1)
        if(sum(d == default_value for d in csv_data) / len(header) < threshold):

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

        # try:
        #     t2 = time.time()
        #     name = "Extended Naive Bayes"
        #     clf = ExtendedNaiveBayes(self.family,path_to_data=self.path_to_data)
        #     predictions = clf.predict_proba(header,csv_data)
        #     predict_payload = {'name': name,'locations': [], 'probabilities': []}
        #     for tup in predictions:
        #         predict_payload['locations'].append(str(self.naming['from'][tup[0]]))
        #         predict_payload['probabilities'].append(round(tup[1],2))
        #     payload['predictions'].append(predict_payload)
        #     self.logger.debug("{} {:d} ms".format(name,int(1000 * (t2 - time.time()))))
        # except Exception as e:
        #     self.logger.error(str(e))

        # try:
        #     t2 = time.time()
        #     name = "Extended Naive Bayes2"
        #     clf = ExtendedNaiveBayes2(self.family, path_to_data=self.path_to_data)
        #     predictions = clf.predict_proba(header, csv_data)
        #     predict_payload = {'name': name, 'locations': [], 'probabilities': []}
        #     for tup in predictions:
        #         predict_payload['locations'].append(
        #             str(self.naming['from'][tup[0]]))
        #         predict_payload['probabilities'].append(round(tup[1], 2))
        #     payload['predictions'].append(predict_payload)
        #     self.logger.debug("{} {:d} ms".format(
        #         name, int(1000 * (t2 - time.time()))))
        # except Exception as e:
        #     self.logger.error(str(e))

        # self.logger.debug("{} {:d} ms".format(
        #     name, int(1000 * (t - time.time()))))
        self.results[index] = predict_payload

    @timeout(1000)
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
                pass

            # set default value
            try:
                self.default_value = float(settings["default"])
            except:
                self.default_value = 0
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
            for i, column in enumerate(fullheader):
                # try if whitelist if available
                try:
                    if column in settings["whitelist"]:
                        columns.append(i)
                        self.header.append(column)
                except:
                    # if not, try to use blacklist
                    try:
                        if column not in settings["blacklist"]:
                            columns.append(i)
                            self.header.append(column)
                    except:
                        # no white- or blacklist, just use it
                        columns.append(i)
                        self.header.append(column)
            self.logger.debug("Using %d features for the AI: %s" % (len(self.header), self.header))
            
            count_all = 0
            count_skipped = 0
            for i, row in enumerate(reader):
                count_all += 1
                new_row = []
                for j in columns:
                    val = row[j]
                    if j == 0:
                        # this is a name of the location
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
                if(sum(d == self.default_value for d in new_row) / (len(self.header) - 1) < self.threshold):
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
        
        try:
            if settings["mode"] == "mean":
                x = numpy.where(x == self.default_value, numpy.nan, x)
                
                rooms = numpy.unique(y)
                for room in rooms:
                    mask = y == room
                    x_room = x[mask]
                    if numpy.isnan(x_room).all():   
                        x[mask] = self.default_value
                    else:
                        room_means = numpy.nanmean(x_room, axis=0)
                        x[mask] = numpy.where(numpy.isnan(x_room), room_means, x_room)
                        self.logger.debug("Mean per column: %s for room %d" % (room_means, room))

        except Exception as e:
            print("An exception occurred: %s" % (e))
        self.logger.debug("x: %s\ncontains nan: %s" % (x, numpy.isnan(x).any()))
        
        names = []
        classifiers = []

        try:
            if "Nearest Neighbors" in settings["models"]:
                names.append("Nearest Neighbors")
                classifiers.append(KNeighborsClassifier(3))
            if "Linear SVM" in settings["models"]:
                names.append("Linear SVM")
                classifiers.append(SVC(kernel="linear", C=0.025, probability=True))
            if "RBF SVM" in settings["models"]:
                names.append("RBF SVM")
                classifiers.append(SVC(gamma=2, C=1, probability=True))
            if "Decision Tree" in settings["models"]:
                names.append("Decision Tree")
                classifiers.append(DecisionTreeClassifier(max_depth=5))
            if "Random Forest" in settings["models"]:
                names.append("Random Forest")
                classifiers.append(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
            if "Neural Net" in settings["models"]:
                names.append("Neural Net")
                classifiers.append(MLPClassifier(alpha=1))
            if "AdaBoost" in settings["models"]:
                names.append("AdaBoost")
                classifiers.append(AdaBoostClassifier())
            if "Naive Bayes" in settings["models"]:
                names.append("Naive Bayes")
                classifiers.append(GaussianNB())
            if "QDA" in settings["models"]:
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
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025, probability=True),
                SVC(gamma=2, C=1, probability=True),
                # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(
                    max_depth=5, n_estimators=10, max_features=1),
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


        # t2 = time.time()
        # name = "Extended Naive Bayes"
        # clf = ExtendedNaiveBayes(self.family, path_to_data=self.path_to_data)
        # try:
        #     clf.fit(fname)
        #     self.logger.debug("learned {}, {:d} ms".format(
        #         name, int(1000 * (t2 - time.time()))))
        # except Exception as e:
        #     self.logger.error(str(e))

        # t2 = time.time()
        # name = "Extended Naive Bayes2"
        # clf = ExtendedNaiveBayes2(self.family, path_to_data=self.path_to_data)
        # try:
        #     clf.fit(fname)
        #     self.logger.debug("learned {}, {:d} ms".format(
        #         name, int(1000 * (t2 - time.time()))))
        # except Exception as e:
        #     self.logger.error(str(e))
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def save(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'wb')
        pickle.dump(self.header, f)
        pickle.dump(self.naming, f)
        pickle.dump(self.algorithms, f)
        pickle.dump(self.family, f)
        f.close()
        self.logger.debug("{:d} ms".format(int(1000 * (t - time.time()))))

    def load(self, save_file):
        t = time.time()
        f = gzip.open(save_file, 'rb')
        self.header = pickle.load(f)
        self.naming = pickle.load(f)
        self.algorithms = pickle.load(f)
        self.family = pickle.load(f)
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
