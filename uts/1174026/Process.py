# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:57:15 2020

@author: ROG
"""

# In[]

from multiprocessing import Process
import os

def runProc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__ == '__main__':
    print('Parent process %s. ' % os.getpid())
    p1 = Process(target=runProc, args=('test', ))
    p2 = Process(target=runProc, args=('test2', ))
    print('Child process will start.')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('Child process end.') 

# In[]
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
print('hello, world ', pid)
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
# In[]

import time
import concurrent.futures
from PIL import Image, ImageFilter

img_names = [
    'photo-1516117172878-fd2c41f4a759.jpg',
    'photo-1532009324734-20a7a5813719.jpg',
    'photo-1524429656589-6633a470097c.jpg',
    'photo-1530224264768-7ff8c1789d79.jpg',
    'photo-1564135624576-c5c88640f235.jpg',
    'photo-1541698444083-023c97d3f4b6.jpg',
    'photo-1522364723953-452d3431c267.jpg',
    'photo-1513938709626-033611b8cc03.jpg',
    'photo-1507143550189-fed454f93097.jpg',
    'photo-1493976040374-85c8e12f0c0e.jpg',
    'photo-1504198453319-5ce911bafcde.jpg',
    'photo-1530122037265-a5f1f91d3b99.jpg',
    'photo-1516972810927-80185027ca84.jpg',
    'photo-1550439062-609e1531270e.jpg',
    'photo-1549692520-acc6669e2f0c.jpg'
]

t1 = time.perf_counter()

size = (1200, 1200)


def process_image(img_name):
    img = Image.open(img_name)

    img = img.filter(ImageFilter.GaussianBlur(15))

    img.thumbnail(size)
    img.save(f'processed/{img_name}')
    print(f'{img_name} was processed...')


with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_image, img_names)


t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
# In[]
import time

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'


with concurrent.futures.ProcessPoolExecutor() as executor:
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)

    # for result in results:
    #     print(result)

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
# In[]
import os
import subprocess
import shlex
import shutil
import multiprocessing

from scapy.all import *
from PIL import Image
import numpy as np

FLAGS = None
_ = None


def do_reset():
    global FLAGS
    if os.path.exists(FLAGS.output):
        shutil.rmtree(FLAGS.output)
    if FLAGS.reset and os.path.exists(FLAGS.temp):
        shutil.rmtree(FLAGS.temp)
    if FLAGS.reset:
        for label, path in read_pcap(FLAGS.input):
            dst = os.path.abspath(
                    os.path.expanduser(os.path.join(FLAGS.temp, label)))
            split_pcap(path, dst)


def pkt2vec(pkt):
    ip = pkt['IP']
    hexst = raw(ip).hex()
    arr = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    arr = arr[0:4*375]
    arr = np.pad(arr, (0, 4*375-len(arr)), 'constant', constant_values=0)
    fv = np.reshape(arr, (-1, 4))
    fv = np.uint8(fv)
    return fv


def pkt2img(base, prefix, cnt):
    global FLAGS
    def process_pkt(pkt):
        if not pkt.haslayer('IP'):
            return
        ip = pkt['IP']
        if not (ip.haslayer('TCP') or ip.haslayer('UDP')):
            return
        if ip.haslayer('TCP'):
            l4 = 'TCP'
        elif ip.haslayer('UDP'):
            l4 = 'UDP'
        if len(raw(ip[l4].payload)) < FLAGS.payload:
            return
        fv = pkt2vec(pkt)
        dst = os.path.join(base, f'{prefix}-{cnt[0]:08d}.png')
        cnt[0] = cnt[0] + 1
        img = Image.fromarray(fv)
        img.save(dst)
    return process_pkt


def stop_filter(current):
    global FLAGS
    def process_pkt(pkt):
        if not pkt.haslayer('IP'):
            return False
        ip = pkt['IP']
        if not (ip.haslayer('TCP') or ip.haslayer('UDP')):
            return False
        if ip.haslayer('TCP'):
            l4 = 'TCP'
        elif ip.haslayer('UDP'):
            l4 = 'UDP'
        if len(raw(ip[l4].payload)) < FLAGS.payload:
            return False
        current[0] = current[0] + 1
        if current[0] > FLAGS.limit:
            return True
        return False
    return process_pkt


def read_pcap(root_dir, ext=('.pcap', '.pcapng')):
    queue = [root_dir]
    while len(queue) != 0:
        nest_dir = queue.pop()
        with os.scandir(nest_dir) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name.endswith(ext):
                        label = os.path.basename(os.path.dirname(entry.path))
                        yield label, entry.path
                elif not entry.name.startswith('.') and entry.is_dir():
                    queue.append(entry.path)


def split_pcap(src, dst):
    os.makedirs(dst, exist_ok=True)
    cmd = f'PcapSplitter -f {src} -o {dst} -m connection'
    cmd = shlex.split(cmd)
    subprocess.run(cmd)


def process_pcap(args):
    label = args[0]
    path = args[1]
    base = os.path.abspath(
             os.path.expanduser(os.path.join(FLAGS.output, label)))
    os.makedirs(base, exist_ok=True)
    prefix = os.path.basename(path)
    cnt = [0]
    current = [0]
    sniff(offline=path, prn=pkt2img(base, prefix, cnt), 
          store=False, stop_filter=stop_filter(current))
    return label, current[0]


def main():
    # Print Parameters
    print(f'Parsed: {FLAGS}')
    print(f'Unparsed: {_}')

    do_reset()

    cnt = dict()
    current = [0]
    with multiprocessing.Pool(FLAGS.process) as p:
        joined_result = p.imap_unordered(process_pcap, read_pcap(FLAGS.temp))
        for result in joined_result:
            print(f'{result[0]}: {result[1]}')


if __name__ == '__main__':
    root_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(root_path)
    os.chdir(root_dir)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help=('Target directory which has pcap files '
                              'in subdirectory'))
    parser.add_argument('--temp', type=str,
                        default='./splited_data',
                        help='Temporarory directory')
    parser.add_argument('--reset', type=bool,
                        default=False,
                        help='Clear temporary files')
    parser.add_argument('--output', type=str,
                        default='./img_data',
                        help='Output directory')
    parser.add_argument('--payload', type=int,
                        default=1,
                        help='Payload size of IP packets')
    parser.add_argument('--limit', type=int,
                        default=float('inf'),
                        help='Limit count per target pcap')
    parser.add_argument('--process', type=int,
                        default=multiprocessing.cpu_count(),
                        help='The number of process pool')

    FLAGS, _ = parser.parse_known_args()

    FLAGS.input = os.path.abspath(os.path.expanduser(FLAGS.input))
    FLAGS.temp = os.path.abspath(os.path.expanduser(FLAGS.temp))
    FLAGS.output = os.path.abspath(os.path.expanduser(FLAGS.output))

    main()
# In[]
from multiprocessing import Process # Can also use pool
import sqlite3
import os

print('Parent process %s.' % os.getpid())

conn = sqlite3.connect('process_test.sqlite')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE Foo
             (id INTEGER PRIMARY KEY, fakeid INTEGER)''')

conn.close()

ids = range(1000)
fakeids = range(0, 10000, 10)

def insert_data(i: int):
    print(f"Process {os.getpid()} started to insert data")

    conn = sqlite3.connect('process_test.sqlite')
    c = conn.cursor()
    num_entries = 1000 // 6
    for j in range(num_entries * i, num_entries * (i + 1)):
        t = (ids[j], fakeids[j])
        c.execute("INSERT INTO Foo VALUES (?, ?)", t)
        conn.commit()
    conn.close()

    print(f"Process {os.getpid()} finished")

processes = []
for i in range(6):
    p = Process(target=insert_data, args=(i,))
    processes.append(p)

for p in processes:
    p.start()
# In[]
import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

np.random.seed(seed=377)
# from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("data/train_features_clean_wmean_2_diff.csv")
test_features = pd.read_csv("data/test_features_clean_wmean_2_diff.csv")
train_labels = pd.read_csv("data/train_labels.csv")
sample = pd.read_csv("data/sample.csv")
stored_usefulness_matrix_t1 = pd.read_csv("data/feature_selection/usefulness_matrix_t1_sum.csv", index_col=0)
stored_usefulness_matrix_t3 = pd.read_csv("data/feature_selection/usefulness_matrix_t3_sum.csv", index_col=0)
best_kernels = pd.read_csv("data/best_kernels.csv", index_col=0)

# features
patient_characteristics = ["Age"]  # TIME VARIABLE IS EXCLUDED
vital_signs = ["Heartrate", "SpO2", "ABPs", "ABPm", "ABPd", "RRate", 'Temp']
tests = ['EtCO2', 'PTT', 'BUN', 'Lactate', 'Hgb', 'HCO3', 'BaseExcess',
         'Fibrinogen', 'Phosphate', 'WBC', 'Creatinine', 'PaCO2', 'AST', 'FiO2',
         'Platelets', 'SaO2', 'Glucose', 'Magnesium', 'Potassium', 'Calcium',
         'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
         'Bilirubin_total', 'TroponinI', 'pH']
dummy_tests = ['dummy_' + test for test in tests]

standard_features = patient_characteristics + vital_signs + tests
diff_features_suffixes = ['_n_extrema', '_diff_mean', '_diff_median', '_diff_max', '_diff_min']
diff_features = sum(
    [[VS + diff_features_suffix for VS in vital_signs] for diff_features_suffix in diff_features_suffixes], [])
all_features = standard_features + dummy_tests + diff_features

# labels
labels_tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
                'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
                'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
                'LABEL_EtCO2']
labels_sepsis = ['LABEL_Sepsis']
labels_VS_mean = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
all_labels = labels_tests + labels_sepsis + labels_VS_mean

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features.drop(labels="pid", axis=1)

# ---------------------------------------------------------
# ----------------- SET PARAMETERS TASK 1------------------
# ---------------------------------------------------------
use_diff = True
features_selection = False
threshold = 4
remove_outliers = True
shuffle = True
improve_kernels = False
submit = True


# ---------------------------------------------------------
# ----------------- DATA SELECTION ------------------------
# ---------------------------------------------------------
def _get_percentiles(data_set, min, max):
    perc = pd.DataFrame(columns=data_set.columns)
    perc.loc[0, :] = np.nanpercentile(np.array(data_set), min,
                                      axis=0, interpolation='lower')
    perc.loc[1, :] = np.nanpercentile(np.array(data_set), max,
                                      axis=0, interpolation='higher')
    return perc


if remove_outliers:
    percentiles = _get_percentiles(train_features, 0.01, 99.99)
    percentiles = percentiles[tests]
    for feature in percentiles.columns:
        mask = np.multiply(
            train_features[feature] > percentiles[feature][0],
            train_features[feature] < percentiles[feature][1])
        train_features = train_features[mask]
        train_labels = train_labels[mask]

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))


def build_set(selected_features, train_size, submit):
    # Definition of test and val data size:
    # task 1
    if submit:
        X = train_features.loc[:, selected_features]
        X_val = train_features.loc[train_size:, selected_features]
        X_test = test_features[selected_features]
    else:
        X = train_features.loc[0:train_size - 1, selected_features]
        X_val = train_features.loc[train_size:, selected_features]
        X_test = test_features[selected_features]

    # Standardize the data
    X = (X - np.mean(X, 0)) / np.std(X, 0)
    X_val = (X_val - np.mean(X_val, 0)) / np.std(X_val, 0)
    X_test = (X_test - np.mean(X_test, 0)) / np.std(X_test, 0)

    # Set NaN to 0
    X[np.isnan(X)] = 0
    X_val[np.isnan((X_val))] = 0
    X_test[np.isnan(X_test)] = 0

    return X, X_val, X_test


# Build sets
# task 1
train_size = int(train_features.shape[0] * 0.8)
selected_features_t1 = standard_features + dummy_tests
if use_diff:
    selected_features_t1 = selected_features_t1 + diff_features
# selected_features_t2 = vital_signs + diff_features
X_t1, X_val_t1, X_test_t1 = build_set(selected_features_t1, train_size, submit)

# Variable for storing prediction
Y_test_tot = pd.DataFrame(np.zeros([X_test_t1.shape[0], len(all_labels)]),
                          columns=all_labels)  # predictions for test set
Y_val_tot = pd.DataFrame(np.zeros([X_val_t1.shape[0], len(all_labels)]), columns=all_labels)  # predictions for val set

# --------------------------------------------------------
# ------------------- TRAINING TASK 1 --------------------
# ---------------------------------------------------------

labels_target = labels_tests + ['LABEL_Sepsis']
scores_t1 = np.ones(len(labels_target))

def tests_fit(i):
    label_target = labels_target[i]
    Y_t1 = train_labels[label_target].iloc[0:train_size]
    Y_val_t1 = train_labels[label_target].iloc[train_size:]

    if submit:
        Y_t1 = train_labels[label_target].iloc[:]
        Y_val_t1 = train_labels[label_target].iloc[train_size:]

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t1[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = sum(
            [[test, 'dummy_' + test] for test in useful_features if test in tests], []) \
                                    + [feature for feature in useful_features if feature in vital_signs + diff_features] \
            # + sum([sum(
        #     [[feature + suffix] for feature in useful_features if feature in vital_signs],
        #     []) for suffix in diff_features_suffixes], [])
        X_t1_useful = X_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_val_t1_useful = X_val_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
        X_test_t1_useful = X_test_t1[list(set(useful_features_augmented) & set(X_t1.columns))]
    else:
        X_t1_useful = X_t1
        X_val_t1_useful = X_val_t1
        X_test_t1_useful = X_test_t1

    # fit

    if not improve_kernels or best_kernels.at[label_target, 'kernel'] == 'poly1':
        #clf = svm.LinearSVC(C=1e-3, tol=1e-2, class_weight='balanced', verbose=0)
        clf = RandomForestClassifier(n_estimators=4500, class_weight="balanced_subsample") #top = 1000
    else:
        kernel_dict = {'poly2': ('poly', 2), 'poly3': ('poly', 3), 'rbf': ('rbf', 0)}
        kernel, degree = kernel_dict[best_kernels.at[label_target, 'kernel']]
        C = best_kernels.at[label_target, 'C']
        clf = svm.SVC(C=C, kernel=kernel, degree=degree, tol=1e-4, class_weight='balanced', verbose=0)

    clf.fit(X_t1_useful, Y_t1)

    # predict and save into dataframe
    # Y_temp = np.array([clf.decision_function(X_val_t1_useful)])
    # Y_val_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    # Y_temp = np.array([clf.decision_function(X_test_t1_useful)])
    # Y_test_pred = (1 / (1 + np.exp(-Y_temp))).flatten()
    #
    Y_val_pred = (1 - clf.predict_proba(X_val_t1_useful))[:, 0]
    Y_test_pred = (1 - clf.predict_proba(X_test_t1_useful))[:, 0]

    Y_test_tot.loc[:, label_target] = Y_test_pred

    score = np.mean([skmetrics.roc_auc_score(Y_val_t1, Y_val_pred)])
    scores_t1[i] = score
    print("ROC AUC -- score ", i, " ", label_target, " :", np.float(score))

# -------------------------------------
# -----------PARALLEL----------------------
# -------------------------------------
import multiprocessing

processes = []
for i in range(0, len(labels_target)):
    p = multiprocessing.Process(target=tests_fit, args=[i])
    p.start()
    processes.append(p)
for process in processes:
    process.join()

task1 = sum(scores_t1[:-1]) / len(scores_t1[:-1])
print("ROC AUC task1 score  ", task1)
task2 = scores_t1[-1]
print("ROC AUC task2 score ", task2)

# -------------------------------------
# ---------- BEGIN MAIN_ALL -----------
# -------------------------------------


# ---------------------------------------------------------
# ------------ DATA IMPORT AND DEFINITIONS ----------------
# ---------------------------------------------------------

# cleaned data import:
train_features = pd.read_csv("data/train_features_clean_columned_diff.csv")
test_features = pd.read_csv("data/test_features_clean_columned_diff.csv")
train_labels = pd.read_csv("data/train_labels.csv")
stored_usefulness_matrix_t1 = pd.read_csv("data/feature_selection/usefulness_matrix_t1_sum_old.csv", index_col=0)
stored_usefulness_matrix_t3 = pd.read_csv("data/feature_selection/usefulness_matrix_t3_sum_old.csv", index_col=0)

N_hours_test = 1
N_hours_VS = 4
houred_features = ['Age'] + \
                  sum([[test + str(i) for i in range(13 - N_hours_test, 13)] + ['dummy_' + test] for test in tests],
                      []) + \
                  sum([[VS + str(i) for i in range(13 - N_hours_VS, 13)] for VS in vital_signs], [])

all_features = patient_characteristics + vital_signs + tests + diff_features

# Drop pid feature:
train_features = train_features.drop(labels="pid", axis=1)
test_features = test_features.drop(labels="pid", axis=1)
# ---------------------------------------------------------
# ----------------- SET PARAMETERS T3 ------------------------
# ---------------------------------------------------------
use_diff = True
features_selection = True
remove_outliers = True
shuffle = False
threshold = 4
improve_kernels = False

# ---------------------------------------------------------
# ----------------- DATA SELECTION T3------------------------
# ---------------------------------------------------------
if remove_outliers:
    percentiles = _get_percentiles(train_features, 1e-2, 100 - 1e-2)
    percentiles = percentiles[
        sum([[houred_test for houred_test in houred_features if (test in houred_test and 'dummy' not in houred_test)]
             for test in tests], [])]
    for feature in percentiles.columns:
        mask = np.multiply(
            train_features[feature] > percentiles[feature][0],
            train_features[feature] < percentiles[feature][1])
        train_features = train_features[mask]
        train_labels = train_labels[mask]

if shuffle:
    rd_permutation = np.random.permutation(train_features.index)
    train_features = train_features.reindex(rd_permutation).set_index(np.arange(0, train_features.shape[0], 1))
    train_labels = train_labels.reindex(rd_permutation).set_index(np.arange(0, train_labels.shape[0], 1))

# task3
train_size = 15000
selected_houred_features_t3 = houred_features
if use_diff:
    selected_houred_features_t3 = selected_houred_features_t3 + diff_features
X_t3, X_val_t3, X_test_t3 = build_set(selected_houred_features_t3, train_size, submit)

# these dataframe will contain every prediction
# Y_test_tot = pd.DataFrame(np.zeros([X_test_t3.shape[0], len(all_labels)]),
#                           columns=all_labels)  # predictions for test set

# ---------------------------------------------------------
# ------------------- TRAINING TASK 3 --------------------
# ---------------------------------------------------------

labels_target = labels_VS_mean
# labels_target = ['LABEL_' + select_feature for select_feature in select_features]
scores_t3 = []
for i in range(0, len(labels_target)):
    # get the set corresponding tu the feature
    label_target = labels_target[i]
    Y_t3 = train_labels.loc[0:train_size - 1, label_target]
    Y_val_t3 = train_labels.loc[train_size:, label_target]

    if submit:
        Y_t3 = train_labels[label_target].iloc[:]
        Y_val_t3 = train_labels[label_target].iloc[train_size:]

    if features_selection:
        usefulness_column = stored_usefulness_matrix_t3[label_target].sort_values(ascending=False)
        useful_features_mask = np.array(usefulness_column) >= threshold
        useful_features = [feature for feature, mask in zip(usefulness_column.index, useful_features_mask) if mask]
        useful_features_augmented = \
            sum([[s for s in selected_houred_features_t3 if feature in s] for feature in useful_features], [])
        X_t3_useful = X_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_val_t3_useful = X_val_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
        X_test_t3_useful = X_test_t3[list(set(useful_features_augmented) & set(X_t3.columns))]
    else:
        X_t3_useful = X_t3
        X_val_t3_useful = X_val_t3
        X_test_t3_useful = X_test_t3

    # fit
    reg = LinearRegression()
    reg.fit(X_t3_useful, Y_t3)  

    # reg = Lasso(alpha=2e-1)
    # reg.fit(X_t3_useful, np.ravel(Y_t3))

    # predict and save into dataframe
    Y_test_pred = reg.predict(X_test_t3_useful).flatten()
    Y_val_pred = reg.predict(X_val_t3_useful).flatten()
    Y_test_tot.loc[:, label_target] = Y_test_pred

    #score = 0.5 + 0.5 * skmetrics.r2_score(Y_val_t3, Y_val_pred, sample_weight=None, multioutput='uniform_average')
    #scores_t3 = scores_t3 + [score]
    #print("Task3 score ", i, " ", label_target, " :", score)

#task3 = np.mean(scores_t3)
#print("Task3 score = ", task3)

#print("Total score = ", np.mean([task1, task2, task3]))

# -------------------------------------
# -----------SAVE----------------------
# -------------------------------------


# save into file
Y_test_tot.insert(0, 'pid', sample['pid'])
Y_test_tot.to_csv('submission.csv', header=True, index=False, float_format='%.7f')
Y_test_tot.to_csv('submission.zip', header=True, index=False, float_format='%.7f', compression='zip')
# In[]
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
# In[]
import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
# In[]
import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
# In[]
from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()
# In[]
from multiprocessing import Process, Pipe

def f(conn):
    conn.send([42, None, 'hello'])
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()
# In[]
from multiprocessing import Process, Lock

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()
# In[]
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
# In[]
from multiprocessing import Process, Manager

def f(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__ == '__main__':
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=f, args=(d, l))
        p.start()
        p.join()

        print(d)
        print(l)
# In[]
import os
from multiprocessing import Process, Lock
import time


# %% Definition of methods
def simpleMethod(label, lock):
    """For conveying with good style of Python code.
    Simple printing function - the competitor for stdout.
    Above also two strings as example of good Pydoc style.
    """
    with lock:
        time.sleep(0.4)  # A delay - resembling some work
        print("The process with data: ", label, os.getpid(), "running")


# %% Main process
lock = Lock()  # The lock for the processes
p = Process(target=simpleMethod, args=("spawned child", lock))
p.start()
p.join()

for i in range(5):
    Process(target=simpleMethod, args=(("process %s" % i), lock)).start()

with lock:
    print("Main thread finished")
# In[]
import time
import multiprocessing


def do_something(seconds):
    print(f'sleeping for {seconds} seconds')
    time.sleep(seconds)
    print('done sleeping')


if __name__ ==  '__main__' :        #necessary

    start = time.perf_counter()

    p1 = multiprocessing.Process(target = do_something, args =[2])
    p2 = multiprocessing.Process(target = do_something, args =[2])


    p1.start()
    p2.start()

    p1.join()
    p2.join()


    finish = time.perf_counter()

    print('finished in {} seconds'.format(finish - start))
# In[]
import concurrent.futures
import time


def do_something(seconds):
    print(f'sleeping for {seconds} seconds')
    time.sleep(seconds)
    return f'done sleeping {seconds} seconds'


if __name__ ==  '__main__' :        #necessary
    with concurrent.futures.ProcessPoolExecutor() as executor:

        seconds = [5,4,3,2,1]

        # p1 = executor.submit(do_something, 2)
        # p2 = executor.submit(do_something, 2)
        #
        # print(p1.result())
        # print(p2.result())

        # results = [executor.submit(do_something, sec) for sec in seconds]
        #
        # for result in concurrent.futures.as_completed(results):
        #     print (result.result())

        results = executor.map(do_something, seconds)
        for result in results:
            print(result)
# In[]
from multiprocessing import Process, Queue, cpu_count

POOL_BATCH_RESULT = "POOL_BATCH_RESULT"

def _fn_batch(fn, results, batchargs):
    result = []
    for args in batchargs:
        ret = fn(*args)
        result.append(ret)
    results.put({ POOL_BATCH_RESULT: result })

class Pool:
    def __init__(self, nprocesses = None, debug = False):
        self._debug = debug
        self._fn = None
        self._tasks = []
        self._results = Queue()
        processes = []
        self._nprocesses = nprocesses if not nprocesses in [None, 0] else cpu_count()
        return

    def AddTask(self, fn, args):
        self._fn = fn
        self._tasks.append(args)
        return

    def AddTaskBatch(self, fn, listargs):
        self._fn = fn
        self._tasks.extend(listargs)
        return

    def Launch(self, autoclose = True):

        # distribute tasks to each process

        num_list_args = len(self._tasks)
        if num_list_args == 0: return

        num_processes = self._nprocesses
        if self._nprocesses > num_list_args: num_processes = num_list_args

        num_items_per_process = self._nprocesses if self._nprocesses < num_list_args else num_list_args

        num_items_per_process = num_list_args // num_processes
        num_items_left = num_list_args % num_processes

        segments = []
        for i in range(0, num_processes):
            start = i * num_items_per_process
            stop  = start + num_items_per_process
            segments.append(stop)

        # balance tasks in all process

        idx_segment_head_for_combination = len(segments) - num_items_left

        if num_items_left > 0:
            for i, _ in enumerate(segments[idx_segment_head_for_combination:]):
                segments[i + idx_segment_head_for_combination] += i + 1
        del segments[-1]
        segments.insert(0, 0)

        # summary tasks

        if self._debug: print(
            ("Summary:") +
            ("\n\t%d items" % num_list_args) +
            ("\n\t%d processes" % num_processes) +
            ("\n\t%d items per process" % num_items_per_process) +
            ("\n\t%d items are combined to last %d processes" % (num_items_left, num_items_left)) +
            ("\n\tProcesses:"))

        # assign tasks to each process

        processes = []

        for i, segment in enumerate(segments):

            start = segments[i]
            stop  = segments[i + 1] if i < num_processes - 1 else None

            batch_args = self._tasks[start:stop]

            if self._debug: print("\t\tProcess#%d: %d items" % (i + 1, len(batch_args)))

            process = Process(target=_fn_batch, args=[self._fn, self._results, batch_args])
            processes.append(process)

        # run multi-processing

        result = []

        if self._debug: print("\t%d processes are created" % len(processes))

        for process in processes: process.start()

        # combine results

        for process in processes:
            ret = self._results.get()
            if type(ret) is dict and POOL_BATCH_RESULT in ret.keys():
                result.extend(ret[POOL_BATCH_RESULT])
            else: result.append(ret)

        for process in processes: process.join()

        if autoclose: self._results.close()

        if self._debug: print("\t%d processes are terminated" % len(processes))

        return result

# Eg.

# from PyVutils import Process

# def task(v1, v2, v3):
#     for v in range(0, 10000000): pass
#     return [v1, v2, v3]

# if __name__ == "__main__":
#     pool = Process.Pool(debug=True)
#     pool.AddTaskBatch(task, [(i, i, i) for i in range(0, 14)])
#     result = pool.Launch()
#     print(result)

def AdjustPrivileges(privileges, enable=True):
    # https://docs.microsoft.com/en-us/windows/win32/secauthz/privilege-constants
    import win32api, win32security
    flags = win32security.TOKEN_ADJUST_PRIVILEGES | win32security.TOKEN_QUERY
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), flags)
    id = win32security.LookupPrivilegeValue(None, privileges)
    if enable: new_privileges = [(id, win32security.SE_PRIVILEGE_ENABLED)]
    else: new_privileges = [(id, 0)]
    win32security.AdjustTokenPrivileges(token, 0, new_privileges)
    return
# In[]
import time
import multiprocessing


def do_something(seconds):
    print(f'sleeping for {seconds} seconds')
    time.sleep(seconds)
    print('done sleeping')


if __name__ ==  '__main__' :        #necessary

    start = time.perf_counter()

    p1 = multiprocessing.Process(target = do_something, args =[2])
    p2 = multiprocessing.Process(target = do_something, args =[2])


    p1.start()
    p2.start()

    p1.join()
    p2.join()


    finish = time.perf_counter()

    print('finished in {} seconds'.format(finish - start))
# In[]
