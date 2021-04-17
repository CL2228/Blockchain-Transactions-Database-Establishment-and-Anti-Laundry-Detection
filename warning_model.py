"""
    By Chenghui Li @ 2021.04.09
    This python script is a warning model of bitcoin mixing detection

    model selection:
    liblinear, weighted classes, sub_unlabeled rate 0.01, L2 Regularization, with coefficient 1e-3

    model_path: ./stage2_result/balanced/1e-3/liblinear/sub_u/logistic.pkl


    CSV Format of val data (doesn't contain Mixer(label))

    TxIn.csv   (Header):
    TxInID:ID(TxIn)
    AddPK
    Value
    TxTime
    NumIn
    NumOut

    TxOut.csv   (Header):
    TxOutID:ID(TxOut）
    AddPk
    Value
    TxTime
    NumIn
    NumOut


    Timeline:
    From 1609459436 To 1609871192
"""

from py2neo import Graph, Node
import numpy as np
from sklearn.linear_model._logistic import LogisticRegression, LogisticRegressionCV
import csv
from itertools import islice
import tqdm
import joblib
from os.path import join
import matplotlib.pyplot as plt
import math
import pandas as pd
import time

_lr_model_path = "./stage2_result/balanced/1e-3/liblinear/sub_u/logistic.pkl"
_csv_path = "../val_data/txs_663913_664912.csv"
_starttime = 1609459436
_endtime = 1609871192
_time_interval = 10800
_value_index = 1             # transfer to original ones
_time_remainder = 1e7           # only last seven digits show difference
_time_dividing_index = 3600     # by hour
_val_features_csv = "../val_data/databse_csv/features.csv"
_lr_threshold = 0.5648
_val_npy_path = "../val_data/databse_csv/features.npy"


def list_average(x:list):
    if len(x) == 0:
        return 0.0

    sum = 0
    for ele in x:
        sum += ele
    return sum / len(x)

def sigmoid(x):
    if x >=128:
        return 1.0
    elif x <= -128:
        return 0.0
    else:
        return 1 / (1 + math.exp(-x))


def final_eval():
    classifier = joblib.load(_lr_model_path)


def raw_data_analysis(csv_path):
    """
    only for the analysis of raw data, not for useful implementation
    :param csv_path:
    :return:
    """
    begin = time.time()
    print("start time: " + str(begin))
    df = pd.DataFrame(pd.read_csv(csv_path))
    print("time usage :" + str(time.time() - begin))


    max_txin = 0
    max_txout = 0
    # for i in tqdm.tqdm(range(2051288)):
    #     temp_txin_dict = eval(df.iat[i, 3])
    #     temp_txout_dict = eval(df.iat[i, 4])
    #     if len(temp_txin_dict) > max_txin:
    #         max_txin = len(temp_txin_dict)
    #         print("max txin number: " + str(max_txin))
    #     if len(temp_txout_dict) > max_txout:
    #         max_txout = len(temp_txout_dict)
    #         print("max txout number: " +str(max_txout))
    for i in range(34342, 34354):
        temp_txin_dict = eval(df.iat[i, 3])[0]
        temp_txout_dict = eval(df.iat[i, 4])[0]
        print("\ntxin:")
        print(temp_txin_dict)
        print(type(temp_txin_dict['addr']))
        print("\ntxout:")
        print(temp_txout_dict)
        print(type(temp_txout_dict['addr']))


def csv_creation4val(csv_path, tx_nums):
    """
    this function aims to create csv file for database establishment of validation data

    :param csv_path: directory of raw data
    :param tx_nums: number of transactions read from the raw file, a raw file usually contains 2M txs
    :return:  nothing returned, but csv files created.
    """
    begin = time.time()
    print("start time: " + str(begin))
    df = pd.DataFrame(pd.read_csv(csv_path))
    print("time usage :" + str(time.time() - begin))

    _txin_save_path = "../val_data/databse_csv/TxIn.csv"
    _txout_save_path = "../val_data/databse_csv/TxOut.csv"

    txin_writer = csv.writer(open(_txin_save_path, "w"))
    txout_writer = csv.writer(open(_txout_save_path, "w"))

    txin_writer.writerow(['TxInID:ID(TxIn)', 'AddPK', 'Value', 'TxTime', 'NumIn', 'NumOut', 'TxID'])
    txout_writer.writerow(['TxOutID:ID(TxOut)', 'AddPK', 'Value', 'TxTime', 'NumIn', 'NumOut', 'TxID'])

    txin_idnum = 0
    txout_idnum = 0

    csv_creation_begin_time = time.time()
    print("now starting proessing csv, start time: " + str(csv_creation_begin_time))


    for i in tqdm.tqdm(range(tx_nums)):
        tx_time = df.iat[i, 2]
        txin_list = eval(df.iat[i, 3])
        txout_list = eval(df.iat[i, 4])
        num_in = len(txin_list)
        num_out = len(txout_list)

        if not ((type(txin_list[0]['addr']) == str and txin_list[0]['addr'] == 'Satoshi Nakamoto')
                or (type(txin_list[0]['addr']) == list and txin_list[0]['addr'][0] == 'Satoshi Nakamoto')):

            for sub_dict in txin_list:  # process txins
                # print(sub_dict)
                if type(sub_dict['addr']) == str:
                    write_add = sub_dict['addr']
                else:
                    continue
                txin_writer.writerow([txin_idnum, write_add, sub_dict['value'], tx_time, num_in, num_out, df.iat[i, 1]])
                txin_idnum += 1


        for sub_dict in txout_list: # process txouts
            if type(sub_dict['addr']) == str:
                write_add = sub_dict['addr']
            else:
                continue
            txout_writer.writerow([txout_idnum, write_add, sub_dict['value'], tx_time, num_in, num_out, df.iat[i, 1]])
            txout_idnum += 1

    print("csv creation done, stop time: "+ str(time.time()) + "  time usage: " + str(time.time() - csv_creation_begin_time))


def one_time_interval_txinout(start_time, end_time, graph:Graph, record_dict:dict):
    temp_dict = {}


    print("now process time interval from " + str(start_time) + " to " + str(end_time) + "...")

    print("start filtering TxIn records...")
    bg_time = time.time()
    match_str = "match(txin:TxIn) where '" + str(start_time) + "'<txin.TxTime<='" + str(end_time) + "' return txin"
    txins = graph.run(match_str).data()
    print("TxIn records filtering done, time usage: " + str(time.time() - bg_time) + " s")
    print(str(len(txins)) + " records found.")

    print("start filtering TxOut records...")
    bg_time = time.time()
    match_str = "match(txout:TxOut) where '" + str(start_time) + "'<txout.TxTime<='" + str(end_time) + "' return txout"
    txouts = graph.run(match_str).data()
    print("TxOut records filtering done, time usage: " + str(time.time() - bg_time) + " s")
    print(str(len(txouts)) + " records found.")

    # process TxIn, the format of txin(example):
    # (_1871782:TxIn {AddPK: '1765mMDeMzphWsqkHUCyCR5odS8HRsy8Mb', NumIn: '7', NumOut: '2', TxInID: '3668', TxTime: '1609459436.0', Value: '0.01256764'})
    for txin in txins:
        txin = txin['txin']


        # creat a sub-dict if this add was not in the record_dict
        if txin['AddPK'] not in record_dict.keys():
            record_dict[txin['AddPK']] = {
                'AddPK': txin['AddPK'],
                'Vector': [0.0, 0.0],
                'PatternNum': 0,
                'TxOutBro': [],
                'TxInBro': [],
                'NumUniSuc': 0,
                'NumUniPre': 0,
                'NumInputs': 0,
                'NumOutputs': 0,
                'TotalAmountIn': 0.0,
                'TotalAmountOut': 0.0}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txin['AddPK'] not in temp_dict.keys():
            temp_dict[txin['AddPK']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': [],
                'TxOutBro': {},
                'TxInBro': {}}

        temp_dict[txin['AddPK']]['in_value'].append(float(txin['Value']) * _value_index)
        temp_dict[txin['AddPK']]['in_time'].append(float(txin['TxTime'][-7:]) / _time_dividing_index)

        if txin['TxID'] not in temp_dict[txin['AddPK']]['TxInBro'].keys():
            temp_dict[txin['AddPK']]['TxInBro'][txin['TxID']] = int(txin['NumIn']) - 1
        else:
            temp_dict[txin['AddPK']]['TxInBro'][txin['TxID']] -= 1

        if int(txin['NumOut']) == 1:
            record_dict[txin['AddPK']]['NumUniSuc'] += 1

        record_dict[txin['AddPK']]['NumOutputs'] += 1
        record_dict[txin['AddPK']]['TotalAmountOut'] += float(txin['Value']) * _value_index


    # process TxOut, the format of txout (example):
    # {'txout': Node('TxOut', AddID='1028', AddPk='1112CUnh1tdsd27qoyEaS4LyitvPZXScSY', Mixer='0', TxID='50236388', TxOutID='54', TxTime='1414775876', Value='5480')}
    for txout in txouts:
        txout = txout['txout']

        # creat a sub-dict if this add was not in the record_dict
        if txout['AddPK'] not in record_dict.keys():
            record_dict[txout['AddPK']] = {
                'AddPK': txout['AddPK'],
                'Vector': [0.0, 0.0],
                'PatternNum': 0,
                'TxOutBro': [],
                'TxInBro': [],
                'NumUniSuc': 0,
                'NumUniPre': 0,
                'NumInputs': 0,
                'NumOutputs': 0,
                'TotalAmountIn': 0.0,
                'TotalAmountOut': 0.0}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txout['AddPK'] not in temp_dict.keys():
            temp_dict[txout['AddPK']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': [],
                'TxOutBro': {},
                'TxInBro': {}}

        temp_dict[txout['AddPK']]['out_value'].append(float(txout['Value']) * _value_index)
        temp_dict[txout['AddPK']]['out_time'].append(float(txout['TxTime'][-7:]) / _time_dividing_index)

        if txout['TxID'] not in temp_dict[txout['AddPK']]['TxOutBro'].keys():
            temp_dict[txout['AddPK']]['TxOutBro'][txout['TxID']] = int(txout['NumOut']) - 1
        else:
            temp_dict[txout['AddPK']]['TxOutBro'][txout['TxID']] -= 1

        if int(txout['NumIn']) == 1:
            record_dict[txout['AddPK']]['NumUniPre'] += 1

        record_dict[txout['AddPK']]['NumInputs'] += 1
        record_dict[txout['AddPK']]['TotalAmountIn'] += float(txout['Value']) * _value_index

    for add in temp_dict.keys():
        for outbro in temp_dict[add]['TxOutBro'].keys():
            record_dict[add]['TxOutBro'].append(temp_dict[add]['TxOutBro'][outbro])
        for inbro in temp_dict[add]['TxInBro'].keys():
            record_dict[add]['TxInBro'].append(temp_dict[add]['TxInBro'][inbro])

        temp_vec = [sigmoid(list_average(temp_dict[add]['out_value']) - list_average(temp_dict[add]['in_value'])),
                    sigmoid(list_average(temp_dict[add]['out_time']) - list_average(temp_dict[add]['in_time']))]

        temp_vec = [(temp_vec[0] + record_dict[add]['PatternNum'] * record_dict[add]['Vector'][0]) / (
                    1 + record_dict[add]['PatternNum']),
                    (temp_vec[1] + record_dict[add]['PatternNum'] * record_dict[add]['Vector'][1]) / (
                                1 + record_dict[add]['PatternNum'])]

        record_dict[add]['Vector'] = temp_vec

    return record_dict


def feature_extraction(graph:Graph):
    record_dict = {}
    start_time = _starttime

    while(start_time < _endtime):
        end_time = min(_endtime, start_time + _time_interval)
        record_dict = one_time_interval_txinout(start_time, end_time, graph, record_dict)
        start_time += _time_interval

    feature_result = open("../val_data/databse_csv/features.csv", 'w')
    result_writer = csv.writer(feature_result)
    result_writer.writerow(['AddPk', 'Vector_0', 'Vector_1', 'TxOutAveBro', 'TxInAveBro', 'NumUniSuc',
                            'NumUniPre', 'NumInputs', 'NumOutputs', 'TotalAmountIn', 'TotalAmountOut',
                            'Ratio'])
    # TODO: finish the remaining sections

    for add in tqdm.tqdm(record_dict.keys()):
        if record_dict[add]['TotalAmountOut'] == 0:
            ratio = 65536
        else:
            ratio = record_dict[add]['TotalAmountIn'] / record_dict[add]['TotalAmountOut']
        result_writer.writerow([add,
                                record_dict[add]['Vector'][0],
                                record_dict[add]['Vector'][1],
                                list_average(record_dict[add]['TxOutBro']),
                                list_average(record_dict[add]['TxInBro']),
                                record_dict[add]['NumUniSuc'],
                                record_dict[add]['NumUniPre'],
                                record_dict[add]['NumInputs'],
                                record_dict[add]['NumOutputs'],
                                record_dict[add]['TotalAmountIn'],
                                record_dict[add]['TotalAmountOut'],
                                ratio])


def csv2npy(path):
    """
    transform data from csv to npy
    :return:    npy files, each row of which is a vector that represents features
    """
    csv_reader = csv.reader(open(path))

    list_container = []
    for ele in tqdm.tqdm(islice(csv_reader, 1, None)):
        list_container.append([float(i) for i in ele[1:]])

    npy_container = np.array(list_container)
    np.save("../val_data/databse_csv/features.npy", npy_container)


def evaluation(classifier_path, npy_path):
    val_data = np.load(npy_path)

    classifier = joblib.load(classifier_path)

    print(val_data.shape)

    result = np.array(classifier.predict_proba(val_data)[:, 1])
    plt.hist(result, bins=300)
    plt.show()



if __name__ == "__main__":
    # csv_creation4val(_csv_path, 1500000)
    # raw_data_analysis(_csv_path)

    # record_dict = {}
    # graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')
    # feature_extraction(graph)
    # one_time_interval_txinout(1609459436, 1609459439, graph, record_dict)

    # csv2npy(_val_features_csv)

    evaluation(_lr_model_path, _val_npy_path)
