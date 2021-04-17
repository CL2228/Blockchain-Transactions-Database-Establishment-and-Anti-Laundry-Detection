"""@Author: Chenghui Li, SCUT. Started on Feb 13, 2021.

    THIRD version of csv creation, for the convenience of extracting features

    [The last version of feature]
    AddPk, Vector [2], TxOutAveBro, TxInAveBro, NumUniSuc, NumUniPre, NumInputs, NumOutputs, TotalAmountIn, TotalAmountOut,
    Ratio=in/out, label(Mixer)

    Before using the large-batch import offered by neo4j, we need to transfer all data to csv file

    This database doesn't contain relationships and nodes of block and chain



    [Tx.csv]   Tx     Node
        TxID:ID(Tx)     BlockID     NumIn       NumOut      TxTime


    [TxIn.csv]  TxIn    Node
        TxInID:ID(TxIn)   TxID      AddID      Value       TxTime       AddPk     Mixer   NumIn   NumOut

    [In2Tx.csv]   In2Tx     Relationship
        :START_ID(TxIn)       :END_ID(Tx)


    [TxOut.csv]  TxOut    Node
        TxOutID:ID(TxOut)   TxID    AddID   Value    TxTime         AddPk     Mixer     NumIn   NumOut

    [Tx2Out.csv]   Tx2Out     Relationship
        :START_ID(Tx)       :END_ID(TxOut)


"""

import csv
import os
import tqdm
import pandas as pd

data_2014 = "../data_sysu/dataset1_2014_11_1500000"
data_2015 = "../data_sysu/dataset2_2015_6_1500000"
data_2016 = "../data_sysu/dataset3_2016_1_1500000"
addresses_file = "addresses.txt"
blockhash_file = "blockhash.txt"
transaction_file = "tx.txt"
txhash_file = "txhash.txt"
txin_file = "txin.txt"
txout_file = "txout.txt"
label_path = "../data_csv/label.csv"
target_path = "../data_csv_0213"

def concat_labels():
    save_path = "label.csv"

    lb_path = "../data_sysu/label"
    files = os.listdir(lb_path)
    for file in tqdm.tqdm(files):
        f = pd.read_csv(os.path.join(lb_path, file))
        f.to_csv(save_path,encoding='utf_8_sig', index=False, header=True, mode='a+')

    # target_label = open('final_label.csv', "w")
    # writer = csv.writer(target_label)
    # with open('label.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         row = row[0].upper()
    #         print(row)
    #         writer.writerow([row])

def get_label(Bitcoin_Add):
    """
    return True if this Add
    :param Bitcoin_Add:  Bitcoin Address, UPPER FORM String
    :return:
    """
    f = open('final_label.csv')
    reader = csv.reader(f)
    labels = []
    for row in reader:
        labels.append(row[0])
    return Bitcoin_Add in labels

def address_csv_process(year_path):

    addresses_data = open(os.path.join(year_path, addresses_file))
    addresses = addresses_data.readlines()

    address_csv = open(target_path + "/" + year_path[22:26] + "/Address.csv", "w")
    address_writer = csv.writer(address_csv)
    address_writer.writerow(['AddID:ID(Address)', 'AddPk', 'Mixer'])

    f = open(label_path)
    reader = csv.reader(f)
    labels = {'start': 0}
    for row in reader:
        labels[row[0]] = 1

    count = 0

    for add in tqdm.tqdm(addresses):
        add = add.split()
        if add[1] in labels.keys():
            count += 1
            address_writer.writerow([int(add[0]), add[1], 1])
        else:
            address_writer.writerow([int(add[0]), add[1], 0])

    print("Number of Mixing addresses", count)
    addresses_data.close()

def tx_csv_process(year_path):
    tx_data = open(os.path.join(year_path, transaction_file))

    txs = tx_data.readlines()

    tx_csv = open(target_path + "/" + year_path[22:26] + "/Tx.csv", "w")
    tx_writer = csv.writer(tx_csv)

    tx_writer.writerow(['TxID:ID(Tx)', 'BlockID', 'NumIn', 'NumOut', 'TxTime'])

    for tx in tqdm.tqdm(txs):
        tx = tx.split()
        tx_writer.writerow([int(tx[0]), int(tx[1]), int(tx[2]), int(tx[3]), int(tx[4])])
    tx_data.close()


def txinout_csv_process(year_path):
    address_path = target_path + "/" + year_path[22:26] + "/Address.csv"
    address = open(address_path)
    add_reader = csv.reader(address)
    tx_path = target_path + "/" + year_path[22:26] + "/Tx.csv"
    txs = open(tx_path)
    tx_reader = csv.reader(txs)

    # add dictionary
    add_dict = {}
    for add in tqdm.tqdm(add_reader):
        add_dict[add[0]] = {'AddPk': add[1], 'Mixer': add[2]}
    # tx dictionary
    tx_dict = {}
    for tx in tqdm.tqdm(tx_reader):
        tx_dict[tx[0]] = {'TxTime': tx[4], 'NumIn': tx[2], 'NumOut': tx[3]}

    # process txin
    txin_data = open(os.path.join(year_path, txin_file))
    txins = txin_data.readlines()

    # creat csv files
    txin_csv = open(target_path + "/" + year_path[22:26] + "/TxIn.csv", "w")
    in2tx_csv = open(target_path + "/" + year_path[22:26] + "/In2Tx.csv", "w")
    txin_writer = csv.writer(txin_csv)
    in2tx_writer = csv.writer(in2tx_csv)

    # write header
    txin_writer.writerow(['TxInID:ID(TxIn)', 'TxID', 'AddID', 'Value', 'TxTime', 'AddPk', 'Mixer', 'NumIn', 'NumOut'])
    in2tx_writer.writerow([':START_ID(TxIn)', ':END_ID(Tx)'])

    # write data
    count = 0  # for TxIn index (Matching for Neo4j establishment)
    for txin in tqdm.tqdm(txins):
        txin = txin.split()
        txin_writer.writerow([count, int(txin[0]), int(txin[1]), int(txin[2]),
                              int(tx_dict[txin[0]]['TxTime']),
                              add_dict[txin[1]]['AddPk'],
                              add_dict[txin[1]]['Mixer'],
                              int(tx_dict[txin[0]]['NumIn']),
                              int(tx_dict[txin[0]]['NumOut'])])
        in2tx_writer.writerow([count, int(txin[0])])
        count += 1
    txin_data.close()


    # process txout
    txout_data = open(os.path.join(year_path, txout_file))
    txouts = txout_data.readlines()

    # creat csv files
    txout_csv = open(target_path + "/" + year_path[22:26] + "/TxOut.csv", "w")
    tx2out = open(target_path + "/" + year_path[22:26] + "/Tx2Out.csv", "w")
    txout_writer = csv.writer(txout_csv)
    tx2out_writer = csv.writer(tx2out)

    # write header
    txout_writer.writerow(['TxOutID:ID(TxOut)', 'TxID', 'AddID', 'Value', 'TxTime', 'AddPk', 'Mixer', 'NumIn', 'NumOut'])
    tx2out_writer.writerow([':START_ID(Tx)', ':END_ID(TxOut)'])

    # write data
    count = 0  # for TxOut index (Matching for Neo4j establishment)
    for txout in tqdm.tqdm(txouts):
        txout = txout.split()
        txout_writer.writerow([count, int(txout[0]), int(txout[1]), int(txout[2]),
                               int(tx_dict[txout[0]]['TxTime']),
                               add_dict[txout[1]]['AddPk'],
                               add_dict[txout[1]]['Mixer'],
                               int(tx_dict[txout[0]]['NumIn']),
                               int(tx_dict[txout[0]]['NumOut'])])
        tx2out_writer.writerow([int(txout[0]), count])
        count += 1
    txout_data.close()


def annual_process(year_path):
    print(year_path[22:26] + " start processing...\nAddresses processing...")
    address_csv_process(year_path)
    print("Addresses processing done, now starting processing Transactions...")
    tx_csv_process(year_path)
    print("Transactions processing done, now starting processing Txin and Txout...")
    txinout_csv_process(year_path)
    print(year_path[22:26] + " All Done!\n")





if __name__ == "__main__":
    to_process_list = [data_2014, data_2015, data_2016]
    for pt in to_process_list:
        annual_process(pt)


