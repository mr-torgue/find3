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
# [Abi, 2025-09-09] For environment + compact JSON logs
import sys, platform
import sklearn

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
            "max_missing": 3,          # Skip rows with >3 missing RSSIs (supports ratio too)
            "default_rssi": -100.0,    # (4) Default changed from 0 -> -100 dBm
            "use_mean_imputation": True,  # (5)
            "models_enabled": None      # (3)
        }

        # [Abi, 2025-09-09] Optional: read whitelist from env if config not provided
        env_whitelist = os.getenv("AP_WHITELIST", "")
        if env_whitelist and not self.config["ap_whitelist"]:
            try:
                self.config["ap_whitelist"] = [x.strip().lower()
                                               for x in env_whitelist.split(",") if x.strip()]
            except Exception:
                pass

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
            self.config["ap_whitelist"] = set([s.lower() for s in self.config["ap_whitelist"]])
        if self.config["ap_blacklist"] is not None:
            self.config["ap_blacklist"] = set([s.lower() for s in self.config["ap_blacklist"]])

        self.mean_per_ap = None  # [Abi, 2025-09-03] Computed in learn(), reused in classify()
        self._hdr_to_idx = None  # [Abi, 2025-09-09] Speed up lookups in classify
        self.x = None            # [Abi, 2025-09-09] keep training matrix for demo/do()
        self.y = None

        # [Abi, 2025-09-09] Log the feature flags once on init
        self.logger.info(
            "[FLAGS] max_missing=%s, default_rssi=%.1f, mean_impute=%s, "
            "models_enabled=%s, ap_whitelist=%d, ap_blacklist=%d",
            self.config["max_missing"], self.config["default_rssi"],
            self.config["use_mean_imputation"], bool(self.config["models_enabled"]),
            len(self.config["ap_whitelist"]) if self.config["ap_whitelist"] else 0,
            len(self.config["ap_blacklist"]) if self.config["ap_blacklist"] else 0
        )

        # [Abi, 2025-09-09] Print environment once
        self._env_summary_once()

    # [Abi, 2025-09-09] ------- Logging helpers -------

    def _env_summary_once(self):
        if getattr(self, "_env_logged", False):
            return
        try:
            self.logger.info(
                "[ENV] python=%s numpy=%s sklearn=%s platform=%s",
                sys.version.split()[0], numpy.__version__, sklearn.__version__, platform.platform()
            )
        finally:
            self._env_logged = True

    def _clip_list(self, items, limit=8):
        """Return a short preview list for logging."""
        items = list(items) if items is not None else []
        return items[:limit] + (["..."] if len(items) > limit else [])

    def _model_brief(self):
        """Compact model info without spamming logs."""
        brief = []
        for name, clf in getattr(self, "algorithms", {}).items():
            kind = type(clf).__name__
            try:
                pcount = len(clf.get_params())
            except Exception:
                pcount = None
            brief.append({"name": name, "type": kind, "n_params": pcount})
        return brief

    def _process_finished(self, stage, extra=None):
        """Unified FINISHED banner with compact details."""
        info = {
            "stage": stage,
            "n_models": len(getattr(self, "algorithms", {})),
            "n_features": (len(self.header) - 1) if getattr(self, "header", None) else 0,
            "n_classes": len(self.naming.get("to", {})),
            "config": {
                "max_missing": self.config.get("max_missing"),
                "default_rssi": self.config.get("default_rssi"),
                "use_mean_imputation": self.config.get("use_mean_imputation"),
                "models_enabled": bool(self.config.get("models_enabled")),
                "ap_whitelist_size": len(self.config["ap_whitelist"]) if self.config.get("ap_whitelist") else 0,
                "ap_blacklist_size": len(self.config["ap_blacklist"]) if self.config.get("ap_blacklist") else 0,
            },
            "models": self._model_brief(),
        }
        if extra:
            info.update(extra)

        # Keep it compact & readable in one line
        try:
            self.logger.info("PROCESS_FINISHED %s", json.dumps(info, default=str))
        except Exception:
            self.logger.info("PROCESS_FINISHED %s", str(info))

    # ---------- New helper: support ratio/int for max_missing ----------
    def _max_missing_threshold(self, num_features):
        """
        [Abi, 2025-09-09] Allow integer or ratio for max_missing.
        - int  : interpreted literally
        - 0<f<1: fraction of features (floored)
        """
        mm = self.config.get("max_missing", 3)
        try:
            if isinstance(mm, float) and 0 < mm < 1:
                return int(math.floor(mm * num_features))
            return int(mm)
        except Exception:
            return 3

    def classify(self, sensor_data):
        header = self.header[1:]
        is_unknown = True

        # [Abi, 2025-09-03] Start from imputation baseline
        if self.config["use_mean_imputation"] and isinstance(self.mean_per_ap, numpy.ndarray):
            csv_data = self.mean_per_ap.copy()
        else:
            csv_data = numpy.full(len(header), self.config["default_rssi"], dtype=float)

        # [Abi, 2025-09-09] Build/keep index map for speed
        if self._hdr_to_idx is None:
            self._hdr_to_idx = {h: i for i, h in enumerate(header)}

        # Apply incoming readings
        seen = 0  # [Abi, 2025-09-03] Track how many features were actually provided
        for sensorType in sensor_data.get('s', {}):
            for sensor in sensor_data['s'][sensorType]:
                sensorName = sensorType + "-" + sensor
                idx = self._hdr_to_idx.get(sensorName)
                if idx is not None:
                    is_unknown = False
                    val = sensor_data['s'][sensorType][sensor]
                    try:
                        csv_data[idx] = float(val)
                        seen += 1
                    except Exception:
                        self.logger.debug("Non-float sensor value {} for {}".format(val, sensorName))

        # Missing metrics
        missing_seen_gap = max(0, len(header) - seen)
        default_val = float(self.config["default_rssi"])
        missing_default_slots = int((csv_data == default_val).sum())
        if missing_seen_gap > self._max_missing_threshold(len(header)):
            self.logger.warning("[CLEAN] inference missing_seen_gap=%d (> %d); proceeding with imputed/defaults",
                                missing_seen_gap, self._max_missing_threshold(len(header)))

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
        top1 = []
        for r in payload['predictions']:
            if r['locations'] and r['probabilities']:
                top1.append((r['name'], r['locations'][0], r['probabilities'][0]))
        self.logger.info(
            "CLASSIFY_SUMMARY seen=%d, missing_seen_gap=%d, missing_default_slots=%d, models=%d, unknown=%s, top1=%s",
            seen, missing_seen_gap, missing_default_slots, len(top1), is_unknown, top1
        )

        # [Abi, 2025-09-09] Final banner for classify()
        self._process_finished(
            stage="classify",
            extra={
                "seen": seen,
                "missing_seen_gap": missing_seen_gap,
                "missing_default_slots": missing_default_slots,
                "unknown": is_unknown,
                "top1_preview": self._clip_list(top1, 5),
            }
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
        # [Abi, 2025-09-09] Debug elapsed per model
        self.logger.debug("classify %s took %d ms", name, int(1000*(time.time()-t)))

    @timeout(10)
    def train(self, clf, x, y):
        return clf.fit(x, y)

    def _filter_header_by_ap(self, orig_header):
        """
        [Abi, 2025-09-09] Apply AP whitelist/blacklist to feature header.
        Accepts entries with or without the 'wifi-' prefix (bare MACs ok).
        """
        ap_whitelist = self.config["ap_whitelist"]
        ap_blacklist = self.config["ap_blacklist"]

        # Build normalized MAC-only sets for comparison
        def mac_only_set(s):
            if not s:
                return None
            out = set()
            for item in s:
                it = item.lower()
                out.add(it[5:] if it.startswith("wifi-") else it)
            return out

        wl_macs = mac_only_set(ap_whitelist)
        bl_macs = mac_only_set(ap_blacklist)

        if wl_macs is None and bl_macs is None:
            return orig_header

        filtered = []
        for ap in orig_header:
            ap_l = ap.lower()
            mac = ap_l[5:] if ap_l.startswith("wifi-") else ap_l
            if wl_macs is not None and mac not in wl_macs:
                continue
            if bl_macs is not None and mac in bl_macs:
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

                    # [Abi, 2025-09-09] Log AP filter effect
                    self.logger.info(
                        "[AP] features_before=%d features_after=%d wl=%d bl=%d",
                        len(feature_header), len(filtered_features),
                        len(self.config["ap_whitelist"]) if self.config.get("ap_whitelist") else 0,
                        len(self.config["ap_blacklist"]) if self.config.get("ap_blacklist") else 0
                    )

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

        # [Abi, 2025-09-09] Missing-count distribution across raw rows
        if raw_rows:
            tentative_features = len(self.header) - 1
            mc = numpy.array([sum(1 for v in r[1:] if v is None) for r in raw_rows], dtype=int)
            p50 = int(numpy.percentile(mc, 50))
            p75 = int(numpy.percentile(mc, 75))
            p90 = int(numpy.percentile(mc, 90))
            p95 = int(numpy.percentile(mc, 95))
            self.logger.info(
                "[CLEAN] missing_per_row: min=%d p50=%d p75=%d p90=%d p95=%d max=%d (features=%d, rows=%d)",
                int(mc.min()), p50, p75, p90, p95, int(mc.max()), tentative_features, len(raw_rows)
            )

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
        max_missing_eff = self._max_missing_threshold(num_features)

        filtered_rows = []
        dropped_rows = 0  # [Abi, 2025-09-03] For summary
        for r in raw_rows:
            feats = r[1:]
            missing_count = sum(1 for v in feats if v is None)
            if missing_count > max_missing_eff:
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

        # Auto-relax if you’d otherwise train on nothing
        if not filtered_rows:
            self.logger.warning(
                "[CLEAN] 0 rows after filtering (features=%d, max_missing=%s, wl=%d, bl=%d). "
                "Relaxing filter: will keep ALL rows and only impute.",
                num_features,
                str(self.config.get('max_missing')),
                len(self.config['ap_whitelist']) if self.config.get('ap_whitelist') else 0,
                len(self.config['ap_blacklist']) if self.config.get('ap_blacklist') else 0,
            )
            for r in raw_rows:
                feats = r[1:]
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
            dropped_rows = 0  # since we kept all

        # Sanity: how many classes remain?
        classes_used = {int(r[0]) for r in filtered_rows}
        if len(classes_used) < 2:
            self.logger.warning(
                "[CLEAN] Only %d class present after filtering/imputation. "
                "Training may be unstable. Consider relaxing filters.",
                len(classes_used)
            )

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

        # [Abi, 2025-09-09] Keep for demo; make available to do()
        self.x = x
        self.y = y
        self._hdr_to_idx = {h: i for i, h in enumerate(self.header[1:])}  # for classify

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

        # [Abi, 2025-09-03] Optional model filtering via config (3)
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
                self.logger.debug("learned %s, %d ms", name, int(1000 * (time.time() - t2)))
                trained.append(name)
            except Exception as e:
                self.logger.error("{} {}".format(name, str(e)))

        elapsed_ms = int(1000 * (time.time() - t0))
        self.logger.debug("{:d} ms".format(elapsed_ms))

        # [Abi, 2025-09-03] Small result summary to INFO log
        self.logger.info(
            "LEARN_SUMMARY rows_total=%d, rows_used=%d, rows_dropped=%d, features=%d, "
            "models_trained=%d, models=%s, max_missing_eff=%d, default_rssi=%.1f, "
            "mean_impute=%s, elapsed_ms=%d",
            len(raw_rows), len(filtered_rows), dropped_rows, num_features,
            len(trained), trained, max_missing_eff, default_rssi,
            self.config['use_mean_imputation'], elapsed_ms
        )

        # [Abi, 2025-09-09] Final banner for learn()
        self._process_finished(
            stage="learn",
            extra={
                "elapsed_ms": elapsed_ms,
                "rows_total": len(raw_rows),
                "rows_used": len(filtered_rows),
                "rows_dropped": dropped_rows,
                "features_preview": self._clip_list(self.header[1:], 8),
                "classes_preview": self._clip_list([self.naming['to'][k] for k in sorted(self.naming['to'].keys())], 8),
            }
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

        # [Abi, 2025-09-09] Final banner for save()
        self._process_finished(
            stage="save",
            extra={"path": save_file, "elapsed_ms": elapsed_ms}
        )

    def load(self, save_file=None):
        # [Abi, 2025-09-09] Make path optional to be consistent with original demo `do()`
        if save_file is None:
            raise ValueError("load(save_file) requires a path")
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
                self.config["ap_whitelist"] = set([s.lower() for s in self.config["ap_whitelist"]])
            if self.config.get("ap_blacklist") is not None and not isinstance(self.config["ap_blacklist"], set):
                self.config["ap_blacklist"] = set([s.lower() for s in self.config["ap_blacklist"]])
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
        # [Abi, 2025-09-09] Keep classify fast after loading
        if self.header:
            self._hdr_to_idx = {h: i for i, h in enumerate(self.header[1:])}

        # [Abi, 2025-09-09] Final banner for load()
        self._process_finished(
            stage="load",
            extra={"path": save_file, "elapsed_ms": elapsed_ms}
        )


def do():
    # NOTE: This demo function is unchanged in spirit; requires explicit paths.
    ai = AI()
    # ai.learn("path/to/train.csv"); ai.save("model.gz")
    # ai.load("model.gz")
    # params below will only work if ai.x/ai.y exist (after learn)
    params = {'quantile': .3,
              'eps': .3,
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': 3}
    if ai.x is None or ai.y is None:
        logger.warning("[demo] ai.x/ai.y are None; call learn() first in your flow.")
        return
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

# ====================== APPEND BELOW THIS LINE ======================
# [Abi, 2025-09-09] Holdout evaluation + simple CLI for train/eval/classify/info

import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_holdout(ai, fname):
    """
    Evaluate an already-trained/loaded AI model on a holdout CSV.
    Uses the model's own header (ai.header) and mean_per_ap for imputation (no leakage).
    Ensemble = average of predict_proba across all trained models.
    """
    if not getattr(ai, "algorithms", None) or not ai.algorithms:
        raise RuntimeError("Model is not loaded/trained. Call ai.learn(...) or ai.load(...).")
    if not getattr(ai, "header", None):
        raise RuntimeError("Model header missing. Train or load a model first.")
    if ai.mean_per_ap is None:
        ai.logger.warning("mean_per_ap is None; falling back to default_rssi for all missing features.")

    feat_names = ai.header[1:]
    feat_to_idx = {h: i for i, h in enumerate(feat_names)}
    n_features = len(feat_names)

    y_true, y_pred = [], []
    total_rows = 0
    used_rows = 0
    skipped_rows = 0

    with open(fname, 'r') as f:
        reader = csv.reader(f)
        test_header = next(reader)
        test_cols = {name: i for i, name in enumerate(test_header)}

        # Build a list of feature columns we can map from test -> model
        available = [c for c in feat_names if c in test_cols]

        for row in reader:
            total_rows += 1
            # Label in column 0 (string)
            label_str = row[0]
            if label_str not in ai.naming['from']:
                ai.logger.warning("[EVAL] Unknown class '%s' not present in training; row skipped.", label_str)
                skipped_rows += 1
                continue
            y_true_id = ai.naming['from'][label_str]

            # Build vector aligned to model features
            vec = numpy.empty(n_features, dtype=float)
            vec[:] = float(ai.config["default_rssi"])
            # Set present features
            for feat in available:
                try:
                    raw = row[test_cols[feat]]
                    vec[feat_to_idx[feat]] = float(raw) if raw != "" else vec[feat_to_idx[feat]]
                except Exception:
                    # keep default/imputed
                    pass

            # Impute from training means where needed
            if ai.config.get("use_mean_imputation") and isinstance(ai.mean_per_ap, numpy.ndarray):
                default_val = float(ai.config["default_rssi"])
                need_impute = (vec == default_val)
                vec[need_impute] = ai.mean_per_ap[need_impute]

            # Aggregate probabilities across models
            proba_sum = None
            n_models = 0
            for name, clf in ai.algorithms.items():
                try:
                    p = clf.predict_proba(vec.reshape(1, -1))[0]
                    proba_sum = p if proba_sum is None else (proba_sum + p)
                    n_models += 1
                except Exception as e:
                    ai.logger.error("[EVAL] %s.predict_proba failed: %s", name, e)

            if n_models == 0:
                ai.logger.error("[EVAL] No models produced probabilities; row skipped.")
                skipped_rows += 1
                continue

            avg_proba = proba_sum / n_models
            pred_id = int(numpy.argmax(avg_proba))

            y_true.append(y_true_id)
            y_pred.append(pred_id)
            used_rows += 1

    if used_rows == 0:
        raise RuntimeError("No evaluable rows. Check headers/classes match the trained model.")

    labels_all = list(range(len(ai.naming['to'])))
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    target_names = [ai.naming['to'][i] for i in labels_all]
    report = classification_report(y_true, y_pred, labels=labels_all, target_names=target_names, zero_division=0)

    ai.logger.info(
        "EVAL_SUMMARY file=%s, rows_total=%d, rows_used=%d, rows_skipped=%d, accuracy=%.4f",
        fname, total_rows, used_rows, skipped_rows, acc
    )
    ai.logger.info("EVAL_CONFUSION labels=%s matrix=%s", target_names, cm.tolist())
    ai.logger.info("EVAL_REPORT\n%s", report)

    ai._process_finished(
        stage="eval",
        extra={
            "file": fname,
            "rows_total": total_rows,
            "rows_used": used_rows,
            "rows_skipped": skipped_rows,
            "accuracy": round(float(acc), 6),
            "labels": target_names,
            "confusion_matrix": cm.tolist(),
        }
    )

    return acc, cm, report


def _load_json(path_or_minus):
    if path_or_minus == "-" or path_or_minus is None:
        return json.loads(sys.stdin.read())
    with open(path_or_minus, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="FIND3 training/eval/classify helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a model from CSV")
    p_train.add_argument("--train_csv", required=True, help="Path to training CSV")
    p_train.add_argument("--save", required=True, help="Where to save model .gz")
    p_train.add_argument("--data_dir", default=None, help="Directory that may contain config.json")
    p_train.add_argument("--config", default=None, help="Optional JSON config file to override defaults")

    p_eval = sub.add_parser("eval", help="Evaluate a saved model on holdout CSV")
    p_eval.add_argument("--model", required=True, help="Path to saved model .gz")
    p_eval.add_argument("--test_csv", required=True, help="Path to holdout CSV")

    p_class = sub.add_parser("classify", help="Classify one JSON sample")
    p_class.add_argument("--model", required=True, help="Path to saved model .gz")
    p_class.add_argument("--json", default="-", help="Path to sensor JSON (or '-' for stdin)")

    p_info = sub.add_parser("info", help="Print model/config info")
    p_info.add_argument("--model", required=True, help="Path to saved model .gz")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = None
        if args.config:
            with open(args.config, "r") as f:
                cfg = json.load(f)
        ai = AI(path_to_data=args.data_dir, config=cfg)
        ai.learn(args.train_csv)
        ai.save(args.save)

    elif args.cmd == "eval":
        ai = AI()
        ai.load(args.model)
        evaluate_holdout(ai, args.test_csv)

    elif args.cmd == "classify":
        ai = AI()
        ai.load(args.model)
        sample = _load_json(args.json)
        out = ai.classify(sample)
        print(json.dumps(out, indent=2, default=str))

    elif args.cmd == "info":
        ai = AI()
        ai.load(args.model)
        ai._process_finished(stage="info", extra={"path": args.model})


if __name__ == "__main__":
    main()
# ====================== END FILE ======================




'''' SEPT 3rd 2025 #!/usr/bin/python3

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

'''''


