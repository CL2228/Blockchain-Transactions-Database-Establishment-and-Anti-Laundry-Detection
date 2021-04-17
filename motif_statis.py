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

_starttimes = [1414771238, 1433088534, 1451578296]
_endtimes = [1416317420, 1434208080, 1452358864]
_time_interval = 10800
_value_index = 1e-8             # transfer to original ones
_time_dividing_index = 3600     # by hour
_time_remainder = 1e7           # only last seven digits show difference

def one_time_interval(start_time, end_time, graph:Graph, record_dict:dict):
    """     a slow version of motifs collection, has been abandoned
    :param start_time:
    :param end_time:
    :param graph:
    :param temp_dict:
    :param record_dict:
    :return:
    """
    temp_dict = {}
    print("now process time interval from " + str(start_time) + " to " + str(end_time) + "...")

    # filter transactions
    match_str = "match (tx:Tx) where '" + str(start_time) + "'<tx.TxTime<='" + str(end_time) + "' return tx"
    result = graph.run(match_str)
    print(match_str)
    txs = result.data()

    for tx in tqdm.tqdm(txs):
        tx = tx['tx']

        # process inputs
        match_str = "match (add:Address)<-[in2add:In2Add]-" \
                    "(txin:TxIn)-[in2tx:In2Tx]->(tx:Tx{TxID:'" + tx['TxID'] + "'}) return add, txin, tx"
        input_datas = graph.run(match_str).data()
        if len(input_datas) > 0:            # skip if number of input = 0, (coin-base)
            # each input data is: {add, in2add, txin, in2tx,tx}
            for ipdata in input_datas:
                # the task of each loop is to update add information in temp_dict

                # creat a sub-dict if this add was not in the record_dict
                if ipdata['add']['AddID'] not in record_dict.keys():
                    record_dict[ipdata['add']['AddID']] = {'AddPk': ipdata['add']['Address'],
                                                           'a1': 0,
                                                           'a2': 0,
                                                           'b1': 0,
                                                           'b2': 0,
                                                           'b3': 0,
                                                           'b4': 0,
                                                           'total': 0,
                                                           'vector': [0.0, 0.0],
                                                           'Mixer': ipdata['add']['Mixer']}

                # creat a sub-dict if this add was not in the 3-h interval temp-dict
                if ipdata['add']['AddID'] not in temp_dict.keys():
                    temp_dict[ipdata['add']['AddID']] = {'in_value': [],
                                                         'in_time': [],
                                                         'out_value': [],
                                                         'out_time': []}

                # update temp_dict
                temp_dict[ipdata['add']['AddID']]['in_value'].append(float(ipdata['txin']['Value']) * _value_index)
                temp_dict[ipdata['add']['AddID']]['in_time'].append(float(ipdata['tx']['TxTime'][-7:]) / _time_dividing_index)

        # process outputs
        match_str = "match (tx:Tx{TxID:'" + tx['TxID'] + "'})" \
                    "-[tx2out:Tx2Out]->(txout:TxOut)-[out2add:Out2Add]->(add:Address) return tx, txout, add"
        output_datas = graph.run(match_str).data()
        if len(output_datas) > 0:           # ship if there is no output (not gonna happen?)
            # each output data is {tx, txout, add}
            for opdata in output_datas:
                # the task of each loop os to update add information in temp_dict

                # creat a sub-dict if this add was not in the record_dict
                if opdata['add']['AddID'] not in record_dict.keys():
                    record_dict[opdata['add']['AddID']] = {'AddPk': opdata['add']['Address'],
                                                           'a1': 0,
                                                           'a2': 0,
                                                           'b1': 0,
                                                           'b2': 0,
                                                           'b3': 0,
                                                           'b4': 0,
                                                           'total': 0,
                                                           'vector': [0.0, 0.0],
                                                           'Mixer': opdata['add']['Mixer']}

                # creat a sub-dict if this add was not in the 3-h interval temp-dict
                if opdata['add']['AddID'] not in temp_dict.keys():
                    temp_dict[opdata['add']['AddID']] = {'in_value': [],
                                                         'in_time': [],
                                                         'out_value': [],
                                                         'out_time': []}

                # update temp_dict
                temp_dict[opdata['add']['AddID']]['out_value'].append(float(opdata['txout']['Value']) * _value_index)
                temp_dict[opdata['add']['AddID']]['in_time'].append(float(opdata['tx']['TxTime'][-7:]) / _time_dividing_index)

    # transactions loop done, now update record_dict
    for key in temp_dict.keys():
        sub_dict = temp_dict[key]
        ave_inval = 0
        ave_intime = 0
        ave_outval = 0
        ave_outtime = 0
        if len(sub_dict['in_value']) == 0:      # only output
            record_dict[key]['a2'] += 1         # record in record_dict
            ave_inval = 0
            ave_intime = 0
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])
        elif len(sub_dict['out_value']) == 0:   # only input
            record_dict[key]['a1'] += 1
            ave_outval = 0
            ave_outtime = 0
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])
        else:                                   # normal situation
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])

            if ave_intime > ave_outtime:
                if ave_inval >= ave_outval:
                    record_dict[key]['b1'] += 1
                else:
                    record_dict[key]['b3'] += 1
            else:
                if ave_inval >= ave_outval:
                    record_dict[key]['b2'] += 1
                else:
                    record_dict[key]['b4'] += 1


        # calculate the feature vector
        temp_vec_value = 1 / (1 + math.exp(-(ave_outval - ave_inval)))
        temp_vec_time = 1 / (1 + math.exp(-(ave_outtime - ave_intime)))
        if record_dict[key]['total'] == 0:
            record_dict[key]['vector'] = [temp_vec_value, temp_vec_time]
        else:
            record_dict[key]['vector'] = [(temp_vec_value + record_dict[key]['vector'][0] * record_dict[key]['total']) /(1 + record_dict[key]['total']),
                                          (temp_vec_time + record_dict[key]['vector'][1] * record_dict[key]['total']) /(1 + record_dict[key]['total'])]


        record_dict[key]['total'] += 1

    return record_dict


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
                'a1': 0,
                'a2': 0,
                'b1': 0,
                'b2': 0,
                'b3': 0,
                'b4': 0,
                'total': 0,
                'vector': [0.0, 0.0],
                'Mixer': txin['Mixer']}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txin['AddID'] not in temp_dict.keys():
            temp_dict[txin['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': []}

        # update temp_dict
        temp_dict[txin['AddID']]['in_value'].append(float(txin['Value']) * _value_index)
        temp_dict[txin['AddID']]['in_time'].append(float(txin['TxTime'][-7:]) / _time_dividing_index)

    # process TxOut, the format of txout (example):
    # {'txout': Node('TxOut', AddID='1028', AddPk='1112CUnh1tdsd27qoyEaS4LyitvPZXScSY', Mixer='0', TxID='50236388', TxOutID='54', TxTime='1414775876', Value='5480')}

    for txout in txouts:
        txout = txout['txout']

        # creat a sub-dict if this add was not in the record_dict
        if txout['AddID'] not in record_dict.keys():
            record_dict[txout['AddID']] = {
                'AddPk': txout['AddPk'],
                'a1': 0,
                'a2': 0,
                'b1': 0,
                'b2': 0,
                'b3': 0,
                'b4': 0,
                'total': 0,
                'vector': [0.0, 0.0],
                'Mixer': txout['Mixer']}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txout['AddID'] not in temp_dict.keys():
            temp_dict[txout['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': []}

        # update temp_dict
        temp_dict[txout['AddID']]['out_value'].append(float(txout['Value']) * _value_index)
        temp_dict[txout['AddID']]['out_time'].append(float(txout['TxTime'][-7:]) / _time_dividing_index)

    # TxIn and TxOut loops done, now update record_dict
    for key in temp_dict.keys():
        sub_dict = temp_dict[key]
        ave_inval = 0
        ave_intime = 0
        ave_outval = 0
        ave_outtime = 0
        temp_vec = [0.0, 0.0]

        if len(sub_dict['in_value']) == 0:      # only output
            record_dict[key]['a2'] += 1         # record in record_dict
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])

            temp_vec = [1.0, 1.0]


        elif len(sub_dict['out_value']) == 0:   # only input
            record_dict[key]['a1'] += 1
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])
            temp_vec = [0.0, 0.0]
        else:                                   # normal situation
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])

            if ave_intime > ave_outtime:
                if ave_inval >= ave_outval:
                    record_dict[key]['b1'] += 1
                else:
                    record_dict[key]['b3'] += 1
            else:
                if ave_inval >= ave_outval:
                    record_dict[key]['b2'] += 1
                else:
                    record_dict[key]['b4'] += 1

            # temp_vec = [1 / (1 + math.exp(-(ave_outval - ave_inval))), 1 / (1 + math.exp(-(ave_outtime - ave_intime)))]


        # calculate the feature vector
        # print("average output value: " + str(ave_outval))
        # print("average input value: " + str(ave_inval))
        # print("average output time: " + str(ave_outtime))
        # print("average input time: " + str(ave_intime))


        # if record_dict[key]['total'] == 0:
        #     record_dict[key]['vector'] = temp_vec
        # else:
        #     record_dict[key]['vector'] = [(temp_vec[0] + record_dict[key]['vector'][0] * record_dict[key]['total']) /(1 + record_dict[key]['total']),
        #                                   (temp_vec[1] + record_dict[key]['vector'][1] * record_dict[key]['total']) /(1 + record_dict[key]['total'])]


        record_dict[key]['total'] += 1
    print("This time interval done!\n")
    return record_dict


def data_statis(csv_path_list):
    #['62163664', '1DpMC774mHfQZHmbNfpu8Evccj1kgKjDrS', '1', '1', '0', '0', '0', '0', '2', '0']

    normal_dicts = []
    mixing_dicts = []

    for csv_path in csv_path_list:

        collection = open(csv_path)
        collect_reader = csv.reader(collection)

        normal_dict = {'a1': 0,
                       'a2': 0,
                       'b1': 0,
                       'b2': 0,
                       'b3': 0,
                       'b4': 0,
                       'total': 0}

        mixing_dict = {'a1': 0,
                       'a2': 0,
                       'b1': 0,
                       'b2': 0,
                       'b3': 0,
                       'b4': 0,
                       'total': 0}

        count = 0
        for cole in tqdm.tqdm(islice(collect_reader, 1, None)):

            if cole[-1] == '0':
                normal_dict['a1'] += int(cole[2])
                normal_dict['a2'] += int(cole[3])
                normal_dict['b1'] += int(cole[4])
                normal_dict['b2'] += int(cole[5])
                normal_dict['b3'] += int(cole[6])
                normal_dict['b4'] += int(cole[7])
                normal_dict['total'] += int(cole[-2])
            else:
                mixing_dict['a1'] += int(cole[2])
                mixing_dict['a2'] += int(cole[3])
                mixing_dict['b1'] += int(cole[4])
                mixing_dict['b2'] += int(cole[5])
                mixing_dict['b3'] += int(cole[6])
                mixing_dict['b4'] += int(cole[7])
                mixing_dict['total'] += int(cole[-2])

        print(mixing_dict['total'])
        print(normal_dict['total'])
        print("mixing: ")
        print("a1: " + str(mixing_dict['a1']) + " rate: " + str(mixing_dict['a1'] / mixing_dict['total']))
        print("a2: " + str(mixing_dict['a2']) + " rate: " + str(mixing_dict['a2'] / mixing_dict['total']))
        print("b1: " + str(mixing_dict['b1']) + " rate: " + str(mixing_dict['b1'] / mixing_dict['total']))
        print("b2: " + str(mixing_dict['b2']) + " rate: " + str(mixing_dict['b2'] / mixing_dict['total']))
        print("b3: " + str(mixing_dict['b3']) + " rate: " + str(mixing_dict['b3'] / mixing_dict['total']))
        print("b4: " + str(mixing_dict['b4']) + " rate: " + str(mixing_dict['b4'] / mixing_dict['total']))
        print("normal: ")
        print("a1: " + str(normal_dict['a1']) + " rate: " + str(normal_dict['a1'] / normal_dict['total']))
        print("a2: " + str(normal_dict['a2']) + " rate: " + str(normal_dict['a2'] / normal_dict['total']))
        print("b1: " + str(normal_dict['b1']) + " rate: " + str(normal_dict['b1'] / normal_dict['total']))
        print("b2: " + str(normal_dict['b2']) + " rate: " + str(normal_dict['b2'] / normal_dict['total']))
        print("b3: " + str(normal_dict['b3']) + " rate: " + str(normal_dict['b3'] / normal_dict['total']))
        print("b4: " + str(normal_dict['b4']) + " rate: " + str(normal_dict['b4'] / normal_dict['total']))
        print("\n\n")
        normal_dicts.append(normal_dict)
        mixing_dicts.append(mixing_dict)

    return normal_dicts, mixing_dicts


def write_result_data_to_excel():
    result_list = ['atf_result_2014.csv', 'atf_result_2015.csv', 'atf_result_2016.csv']
    normal_dicts, mixing_dicts = data_statis(result_list)

    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('result')
    worksheet.write(0, 1, 'a1')
    worksheet.write(0, 2, 'a2')
    worksheet.write(0, 3, 'b1')
    worksheet.write(0, 4, 'b2')
    worksheet.write(0, 5, 'b3')
    worksheet.write(0, 6, 'b4')
    worksheet.write(0, 7, 'total')

    total_normal_result = {'a1': 0, 'a2': 0, 'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0, 'total': 0}
    total_mixing_result = {'a1': 0, 'a2': 0, 'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0, 'total': 0}
    row = 1

    for i in range(3):
        worksheet.write(row, 0, 'normal')
        worksheet.write(row + 2, 0, 'mixing')
        worksheet.write(row, 1, normal_dicts[i]['a1'])
        worksheet.write(row, 2, normal_dicts[i]['a2'])
        worksheet.write(row, 3, normal_dicts[i]['b1'])
        worksheet.write(row, 4, normal_dicts[i]['b2'])
        worksheet.write(row, 5, normal_dicts[i]['b3'])
        worksheet.write(row, 6, normal_dicts[i]['b4'])
        worksheet.write(row, 7, normal_dicts[i]['total'])
        worksheet.write(row + 1, 1, normal_dicts[i]['a1'] / normal_dicts[i]['total'])
        worksheet.write(row + 1, 2, normal_dicts[i]['a2'] / normal_dicts[i]['total'])
        worksheet.write(row + 1, 3, normal_dicts[i]['b1'] / normal_dicts[i]['total'])
        worksheet.write(row + 1, 4, normal_dicts[i]['b2'] / normal_dicts[i]['total'])
        worksheet.write(row + 1, 5, normal_dicts[i]['b3'] / normal_dicts[i]['total'])
        worksheet.write(row + 1, 6, normal_dicts[i]['b4'] / normal_dicts[i]['total'])

        worksheet.write(row + 2, 1, mixing_dicts[i]['a1'])
        worksheet.write(row + 2, 2, mixing_dicts[i]['a2'])
        worksheet.write(row + 2, 3, mixing_dicts[i]['b1'])
        worksheet.write(row + 2, 4, mixing_dicts[i]['b2'])
        worksheet.write(row + 2, 5, mixing_dicts[i]['b3'])
        worksheet.write(row + 2, 6, mixing_dicts[i]['b4'])
        worksheet.write(row + 2, 7, mixing_dicts[i]['total'])
        worksheet.write(row + 3, 1, mixing_dicts[i]['a1'] / mixing_dicts[i]['total'])
        worksheet.write(row + 3, 2, mixing_dicts[i]['a2'] / mixing_dicts[i]['total'])
        worksheet.write(row + 3, 3, mixing_dicts[i]['b1'] / mixing_dicts[i]['total'])
        worksheet.write(row + 3, 4, mixing_dicts[i]['b2'] / mixing_dicts[i]['total'])
        worksheet.write(row + 3, 5, mixing_dicts[i]['b3'] / mixing_dicts[i]['total'])
        worksheet.write(row + 3, 6, mixing_dicts[i]['b4'] / mixing_dicts[i]['total'])

        row += 5

        total_mixing_result['a1'] += mixing_dicts[i]['a1']
        total_mixing_result['a2'] += mixing_dicts[i]['a2']
        total_mixing_result['b1'] += mixing_dicts[i]['b1']
        total_mixing_result['b2'] += mixing_dicts[i]['b2']
        total_mixing_result['b3'] += mixing_dicts[i]['b3']
        total_mixing_result['b4'] += mixing_dicts[i]['b4']
        total_mixing_result['total'] += mixing_dicts[i]['total']

        total_normal_result['a1'] += normal_dicts[i]['a1']
        total_normal_result['a2'] += normal_dicts[i]['a2']
        total_normal_result['b1'] += normal_dicts[i]['b1']
        total_normal_result['b2'] += normal_dicts[i]['b2']
        total_normal_result['b3'] += normal_dicts[i]['b3']
        total_normal_result['b4'] += normal_dicts[i]['b4']
        total_normal_result['total'] += normal_dicts[i]['total']

    worksheet.write(row, 0, 'normal')
    worksheet.write(row + 2, 0, 'mixing')
    worksheet.write(row, 1, total_normal_result['a1'])
    worksheet.write(row, 2, total_normal_result['a2'])
    worksheet.write(row, 3, total_normal_result['b1'])
    worksheet.write(row, 4, total_normal_result['b2'])
    worksheet.write(row, 5, total_normal_result['b3'])
    worksheet.write(row, 6, total_normal_result['b4'])
    worksheet.write(row, 7, total_normal_result['total'])
    worksheet.write(row + 1, 1, total_normal_result['a1'] / total_normal_result['total'])
    worksheet.write(row + 1, 2, total_normal_result['a2'] / total_normal_result['total'])
    worksheet.write(row + 1, 3, total_normal_result['b1'] / total_normal_result['total'])
    worksheet.write(row + 1, 4, total_normal_result['b2'] / total_normal_result['total'])
    worksheet.write(row + 1, 5, total_normal_result['b3'] / total_normal_result['total'])
    worksheet.write(row + 1, 6, total_normal_result['b4'] / total_normal_result['total'])

    worksheet.write(row + 2, 1, total_mixing_result['a1'])
    worksheet.write(row + 2, 2, total_mixing_result['a2'])
    worksheet.write(row + 2, 3, total_mixing_result['b1'])
    worksheet.write(row + 2, 4, total_mixing_result['b2'])
    worksheet.write(row + 2, 5, total_mixing_result['b3'])
    worksheet.write(row + 2, 6, total_mixing_result['b4'])
    worksheet.write(row + 2, 7, total_mixing_result['total'])
    worksheet.write(row + 3, 1, total_mixing_result['a1'] / total_mixing_result['total'])
    worksheet.write(row + 3, 2, total_mixing_result['a2'] / total_mixing_result['total'])
    worksheet.write(row + 3, 3, total_mixing_result['b1'] / total_mixing_result['total'])
    worksheet.write(row + 3, 4, total_mixing_result['b2'] / total_mixing_result['total'])
    worksheet.write(row + 3, 5, total_mixing_result['b3'] / total_mixing_result['total'])
    worksheet.write(row + 3, 6, total_mixing_result['b4'] / total_mixing_result['total'])
    workbook.save('result.xls')


def debugger():
    """
    :param txins:
    :param txouts:
    :param record_dict:
    :return:
    """
    temp_dict = {}

    record_dict = {}
    txins = [{'txin': Node('TxIn', AddID='1', AddPk='1', Mixer='0', TxID=1,
                           TxInID='1', TxTime='10', Value='20')},
             {'txin': Node('TxIn', AddID='1', AddPk='1', Mixer='0', TxID=2,
                           TxInID='2', TxTime='10', Value='20')}]
    txouts = [{'txout': Node('TxOut', AddID='1', AddPk='1', Mixer='0', TxID=1,
                             TxOutID='1', TxTime='5', Value='15')},
              {'txout': Node('TxOut', AddID='1', AddPk='1', Mixer='0', TxID=1,
                             TxOutID='2', TxTime='5', Value='15')}]

    # process TxIn, the format of txin(example):
    # {'txin': Node('TxIn', AddID='2876', AddPk='111YjD4mzQ9KMFazZHqAuFsN5MUVmyuHv', Mixer='0', TxID='50236885', TxInID='247', TxTime='1414776532', Value='17936')}
    for txin in txins:
        txin = txin['txin']

        # creat a sub-dict if this add was not in the record_dict
        if txin['AddID'] not in record_dict.keys():
            record_dict[txin['AddID']] = {
                'AddPk': txin['AddPk'],
                'a1': 0,
                'a2': 0,
                'b1': 0,
                'b2': 0,
                'b3': 0,
                'b4': 0,
                'total': 0,
                'vector': [0.0, 0.0],
                'Mixer': txin['Mixer']}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txin['AddID'] not in temp_dict.keys():
            temp_dict[txin['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': []}

        # update temp_dict
        temp_dict[txin['AddID']]['in_value'].append(float(txin['Value']) * _value_index)
        temp_dict[txin['AddID']]['in_time'].append(float(txin['TxTime'][-7:]) / _time_dividing_index)

    # process TxOut, the format of txout (example):
    # {'txout': Node('TxOut', AddID='1028', AddPk='1112CUnh1tdsd27qoyEaS4LyitvPZXScSY', Mixer='0', TxID='50236388', TxOutID='54', TxTime='1414775876', Value='5480')}

    for txout in txouts:
        txout = txout['txout']

        # creat a sub-dict if this add was not in the record_dict
        if txout['AddID'] not in record_dict.keys():
            record_dict[txout['AddID']] = {
                'AddPk': txout['AddPk'],
                'a1': 0,
                'a2': 0,
                'b1': 0,
                'b2': 0,
                'b3': 0,
                'b4': 0,
                'total': 0,
                'vector': [0.0, 0.0],
                'Mixer': txout['Mixer']}

        # creat a sub-dict if this add was not in the 3-h interval temp-dict
        if txout['AddID'] not in temp_dict.keys():
            temp_dict[txout['AddID']] = {
                'in_value': [],
                'in_time': [],
                'out_value': [],
                'out_time': []}

        # update temp_dict
        temp_dict[txout['AddID']]['out_value'].append(float(txout['Value']) * _value_index)
        temp_dict[txout['AddID']]['out_time'].append(float(txout['TxTime'][-7:]) / _time_dividing_index)

    # TxIn and TxOut loops done, now update record_dict
    for key in temp_dict.keys():
        sub_dict = temp_dict[key]
        ave_inval = 0
        ave_intime = 0
        ave_outval = 0
        ave_outtime = 0
        temp_vec = [0.0, 0.0]

        if len(sub_dict['in_value']) == 0:  # only output
            record_dict[key]['a2'] += 1  # record in record_dict
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])

            temp_vec = [1.0, 1.0]


        elif len(sub_dict['out_value']) == 0:  # only input
            record_dict[key]['a1'] += 1
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])
            temp_vec = [0.0, 0.0]
        else:  # normal situation
            for item in sub_dict['out_value']:
                ave_outval += item
            ave_outval /= len(sub_dict['out_value'])
            for item in sub_dict['out_time']:
                ave_outtime += item
            ave_outtime /= len(sub_dict['out_time'])
            for item in sub_dict['in_value']:
                ave_inval += item
            ave_inval /= len(sub_dict['in_value'])
            for item in sub_dict['in_time']:
                ave_intime += item
            ave_intime /= len(sub_dict['in_time'])

            if ave_intime > ave_outtime:
                if ave_inval >= ave_outval:
                    record_dict[key]['b1'] += 1
                else:
                    record_dict[key]['b3'] += 1
            else:
                if ave_inval >= ave_outval:
                    record_dict[key]['b2'] += 1
                else:
                    record_dict[key]['b4'] += 1

            # temp_vec = [1 / (1 + math.exp(-(ave_outval - ave_inval))), 1 / (1 + math.exp(-(ave_outtime - ave_intime)))]

        # calculate the feature vector
        # print("average output value: " + str(ave_outval))
        # print("average input value: " + str(ave_inval))
        # print("average output time: " + str(ave_outtime))
        # print("average input time: " + str(ave_intime))

        # if record_dict[key]['total'] == 0:
        #     record_dict[key]['vector'] = temp_vec
        # else:
        #     record_dict[key]['vector'] = [(temp_vec[0] + record_dict[key]['vector'][0] * record_dict[key]['total']) /(1 + record_dict[key]['total']),
        #                                   (temp_vec[1] + record_dict[key]['vector'][1] * record_dict[key]['total']) /(1 + record_dict[key]['total'])]

        record_dict[key]['total'] += 1
    print("This time interval done!\n")
    return record_dict





if __name__ == "__main__":
    # graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')
    # record_dict = {}
    #
    # start_time = _starttimes[2]
    #
    # while (start_time < _endtimes[2]):
    #     end_time = min(_endtimes[2], start_time + _time_interval)
    #     record_dict = one_time_interval_txinout(start_time, end_time, graph, record_dict)
    #     start_time += _time_interval
    #
    # ath_motif_result = open("atf_result_2016.csv", "w")
    # ath_writer = csv.writer(ath_motif_result)
    # ath_writer.writerow(['AddID', 'AddPk', 'a1', 'a2', 'b1', 'b2', 'b3', 'b4', 'total', 'Mixer'])
    #
    # for key in tqdm.tqdm(record_dict.keys()):
    #     ath_writer.writerow([key,
    #                          record_dict[key]['AddPk'],
    #                          record_dict[key]['a1'],
    #                          record_dict[key]['a2'],
    #                          record_dict[key]['b1'],
    #                          record_dict[key]['b2'],
    #                          record_dict[key]['b3'],
    #                          record_dict[key]['b4'],
    #                          record_dict[key]['total'],
    #                          record_dict[key]['Mixer']])


    # record_dict = debugger()
    # print(record_dict)
    write_result_data_to_excel()














