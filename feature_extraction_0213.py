"""
    @Author: Chenghui Li, SCUT. Started on Jan 28, 2021.
    For Motifs collection and statistics in final thesis project

    1. Time Interval of data sets:
        * 3hs = 60 * 60 * 3 =  10,800 seconds
        [2014]
            min Tx time: 1414771239
            max Tx time: 1416317420
        [2015]
            min Tx time: 1433088535
            max Tx time: 1434208080
        [2016]
            min Tx time: 1451578297
            max Tx time: 1452358864
"""
from py2neo import Graph, Node
import math
import tqdm
import time
import xlwt
from itertools import islice
import csv
import random

_starttimes = [1414771238, 1433088534, 1451578296]
_endtimes = [1416317420, 1434208080, 1452358864]
_time_interval = 10800
_value_index = 1e-8             # transfer to original ones
_time_dividing_index = 3600     # by hour
_time_remainder = 1e7           # only last seven digits show difference


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
    # {'txin': Node('TxIn', AddID='2876', AddPk='111YjD4mzQ9KMFazZHqAuFsN5MUVmyuHv', Mixer='0', TxID='50236885', TxInID='247', TxTime='1414776532', Value='17936')}
    for txin in txins:
        txin = txin['txin']

        # creat a sub-dict if this add was not in the record_dict
        if txin['AddID'] not in record_dict.keys():
            record_dict[txin['AddID']] = {
                'AddPk': txin['AddPk'],
                'Mixer': txin['Mixer'],
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
        if txin['AddID'] not in temp_dict.keys():
            temp_dict[txin['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': [],
                'TxOutBro': {},
                'TxInBro': {}}

        temp_dict[txin['AddID']]['in_value'].append(float(txin['Value']) * _value_index)
        temp_dict[txin['AddID']]['in_time'].append(float(txin['TxTime'][-7:]) / _time_dividing_index)

        if txin['TxID'] not in temp_dict[txin['AddID']]['TxInBro'].keys():
            temp_dict[txin['AddID']]['TxInBro'][txin['TxID']] = int(txin['NumIn']) - 1
        else:
            temp_dict[txin['AddID']]['TxInBro'][txin['TxID']] -= 1

        if int(txin['NumOut']) == 1:
            record_dict[txin['AddID']]['NumUniSuc'] += 1

        record_dict[txin['AddID']]['NumOutputs'] += 1
        record_dict[txin['AddID']]['TotalAmountOut'] += float(txin['Value']) * _value_index


    # process TxOut, the format of txout (example):
    # {'txout': Node('TxOut', AddID='1028', AddPk='1112CUnh1tdsd27qoyEaS4LyitvPZXScSY', Mixer='0', TxID='50236388', TxOutID='54', TxTime='1414775876', Value='5480')}
    for txout in txouts:
        txout = txout['txout']

        # creat a sub-dict if this add was not in the record_dict
        if txout['AddID'] not in record_dict.keys():
            record_dict[txout['AddID']] = {
                'AddPk': txout['AddPk'],
                'Mixer': txout['Mixer'],
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
        if txout['AddID'] not in temp_dict.keys():
            temp_dict[txout['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': [],
                'TxOutBro': {},
                'TxInBro': {}}

        temp_dict[txout['AddID']]['out_value'].append(float(txout['Value']) * _value_index)
        temp_dict[txout['AddID']]['out_time'].append(float(txout['TxTime'][-7:]) / _time_dividing_index)

        if txout['TxID'] not in temp_dict[txout['AddID']]['TxOutBro'].keys():
            temp_dict[txout['AddID']]['TxOutBro'][txout['TxID']] = int(txout['NumOut']) - 1
        else:
            temp_dict[txout['AddID']]['TxOutBro'][txout['TxID']] -= 1

        if int(txout['NumIn']) == 1:
            record_dict[txout['AddID']]['NumUniPre'] += 1

        record_dict[txout['AddID']]['NumInputs'] += 1
        record_dict[txout['AddID']]['TotalAmountIn'] += float(txout['Value']) * _value_index


    for add in temp_dict.keys():
        for outbro in temp_dict[add]['TxOutBro'].keys():
            record_dict[add]['TxOutBro'].append(temp_dict[add]['TxOutBro'][outbro])
        for inbro in temp_dict[add]['TxInBro'].keys():
            record_dict[add]['TxInBro'].append(temp_dict[add]['TxInBro'][inbro])

        temp_vec = [sigmoid(list_average(temp_dict[add]['out_value']) - list_average(temp_dict[add]['in_value'])),
                    sigmoid(list_average(temp_dict[add]['out_time']) - list_average(temp_dict[add]['in_time']))]

        temp_vec = [(temp_vec[0] + record_dict[add]['PatternNum'] * record_dict[add]['Vector'][0]) / (1 + record_dict[add]['PatternNum']),
                    (temp_vec[1] + record_dict[add]['PatternNum'] * record_dict[add]['Vector'][1]) / (1 + record_dict[add]['PatternNum'])]

        record_dict[add]['Vector'] = temp_vec

    return record_dict

def feature_extraction_annual(year, graph):
    if year == 2014:
        year_index = 0
    elif year == 2015:
        year_index = 1
    elif year == 2016:
        year_index = 2
    else:
        print("wrong year!")
        return

    record_dict = {}

    start_time = _starttimes[year_index]

    while(start_time < _endtimes[year_index]):
        end_time = min(_endtimes[year_index], start_time + _time_interval)
        record_dict = one_time_interval_txinout(start_time, end_time, graph, record_dict)
        start_time += _time_interval

    feature_result = open("features_" + str(year) + ".csv", "w")
    result_writer = csv.writer(feature_result)
    result_writer.writerow(['AddPk', 'Vector_0', 'Vector_1', 'TxOutAveBro', 'TxInAveBro', 'NumUniSuc',
                            'NumUniPre', 'NumInputs', 'NumOutputs', 'TotalAmountIn', 'TotalAmountOut',
                            'Ratio', 'Mixer'])

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
                                ratio,
                                record_dict[add]['Mixer']])

def data_distribute(files):
    val_writer = csv.writer(open("./processed_data/validation/validation_set.csv", "w"))
    train_p_writer = csv.writer(open("./processed_data/train/positive/train_positive_set.csv", "w"))
    train_u_writer = csv.writer(open("./processed_data/train/unlabeled/train_unlabeled_set.csv", "w"))
    count = 0

    header = ['AddPk', 'Vector_0', 'Vector_1', 'TxOutAveBro', 'TxInAveBro', 'NumUniSuc', 'NumUniPre', 'NumInputs',
              'NumOutputs', 'TotalAmountIn', 'TotalAmountOut', 'Ratio', 'Mixer']
    val_writer.writerow(header)
    train_p_writer.writerow(header)
    train_u_writer.writerow(header)

    for file in files:
        file_reader = csv.reader(open(file))
        for ele in tqdm.tqdm(islice(file_reader, 1, None)):
            if random.random() >= 0.3:
                if ele[-1] == '1':
                    train_p_writer.writerow(ele)
                else:
                    train_u_writer.writerow(ele)
            else:
                val_writer.writerow(ele)


if __name__ == "__main__":
   files = ['features_2014.csv', 'features_2015.csv', 'features_2016.csv']
   data_distribute(files)
