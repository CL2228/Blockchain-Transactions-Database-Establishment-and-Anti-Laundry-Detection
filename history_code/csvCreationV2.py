"""@Author: Chenghui Li, SCUT. Started on Jan 29, 2021.

    SECOND version of csv creation, for the convenience of extracting ATH motif

    Before using the large-batch import offered by neo4j, we need to transfer all data to csv file

    [Block.csv]  Block   Node
        BlockID:ID(Block)   BlockHash       BlockTime       NumTx

    [Chain.csv]  CHAIN   Relationship
        :START_ID(Block)      :END_ID(Block)

    [Address.csv] Address  Node
        AddID:ID(Address)   AddPk         Mixer

    [Tx.csv]   Tx     Node
        TxID:ID(Tx)     BlockID     NumIn       NumOut      TxTime      TxHash

    [Tx2Block.csv]  Tx2Block    Relationship
        :START_ID(Tx)     :END_ID(Block)

    [TxIn.csv]  TxIn    Node
        TxInID:ID(Tx)   TxID      AddID      Value       TxTime       AddPk     Mixer

    [In2Tx.csv]   In2Tx     Relationship
        :START_ID(TxIn)       :END_ID(Tx)

    [In2Add.csv]    In2Add      Relationship
        :START_ID(TxIn)       :END_ID(Address)

    [TxOut.csv]  TxOut    Node
        TxOutID:ID(TxOut)   TxID    AddID   Value    TxTime         AddPk     Mixer

    [Tx2Out.csv]   Tx2Out     Relationship
        :START_ID(Tx)       :END_ID(TxOut)

    [Out2Add.csv]    OUt2Add      Relationship
        :START_ID(TxOut)       :END_ID(Address)

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
target_path = "../data_csv_0129"

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

def block_chain_csv_process(year_path):
    block_data = open(os.path.join(year_path, blockhash_file))
    blocks = block_data.readlines()

    block_csv = open(target_path + "/" + year_path[22:26] + "/Block.csv", "w")
    chain_csv = open(target_path + "/" + year_path[22:26] + "/Chain.csv", "w")
    block_writer = csv.writer(block_csv)
    chain_writer = csv.writer(chain_csv)
    block_writer.writerow(['BlockID:ID(Block)', 'BlockHash', 'BlockTime', 'NumTx'])
    chain_writer.writerow([':START_ID(Block)', ':END_ID(Block)'])

    first_block = blocks[0].split()
    block_writer.writerow([int(first_block[0]), first_block[1], int(first_block[2]), int(first_block[3])])

    for i in tqdm.tqdm(range(1, len(blocks))):
        block = blocks[i].split()
        block_writer.writerow([int(block[0]), block[1], int(block[2]), int(block[3])])
        chain_writer.writerow([int(block[0]), int(block[0]) - 1])
    block_data.close()

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
    tx_hash_data = open(os.path.join(year_path, txhash_file))
    tx_hashs = tx_hash_data.readlines()
    txs = tx_data.readlines()

    tx_csv = open(target_path + "/" + year_path[22:26] + "/Tx.csv", "w")
    tx_writer = csv.writer(tx_csv)
    tx2block_csv = open(target_path + "/" + year_path[22:26] + "/Tx2Block.csv", "w")
    tx2block_writer = csv.writer(tx2block_csv)

    tx_writer.writerow(['TxID:ID(Tx)', 'BlockID', 'NumIn', 'NumOut', 'TxTime', 'TxHash'])
    tx2block_writer.writerow([':START_ID(Tx)', 'TxID', 'BlockID', ':END_ID(Block)'])

    txhash_dict = {}
    for txhash in tqdm.tqdm(tx_hashs):
        txhash = txhash.split()
        txhash_dict[int(txhash[0])] = txhash[1]

    for tx in tqdm.tqdm(txs):
        tx = tx.split()
        txh = txhash_dict[int(tx[0])]
        tx_writer.writerow([int(tx[0]), int(tx[1]), int(tx[2]), int(tx[3]), int(tx[4]), txh])
        tx2block_writer.writerow([int(tx[0]), int(tx[0]), int(tx[1]), int(tx[1])])
    tx_data.close()

def txin_csv_process(year_path):
    txin_data = open(os.path.join(year_path, txin_file))
    txins = txin_data.readlines()

    # creat csv files
    txin_csv = open(target_path + "/" + year_path[22:26] + "/TxIn.csv", "w")
    in2tx_csv = open(target_path + "/" + year_path[22:26] + "/In2Tx.csv", "w")
    in2add_csv = open(target_path + "/" + year_path[22:26] + "/In2Add.csv", "w")
    txin_writer = csv.writer(txin_csv)
    in2tx_writer = csv.writer(in2tx_csv)
    in2add_writer = csv.writer(in2add_csv)

    # write header
    txin_writer.writerow(['TxInID:ID(TxIn)', 'TxID', 'AddID', 'Value'])
    in2tx_writer.writerow([':START_ID(TxIn)', ':END_ID(Tx)'])
    in2add_writer.writerow([':START_ID(TxIn)', ':END_ID(Address)'])

    # write data
    count = 0   # for TxIn index (Matching for Neo4j establishment)
    for txin in tqdm.tqdm(txins):
        txin = txin.split()
        txin_writer.writerow([count, int(txin[0]), int(txin[1]), int(txin[2])])
        in2tx_writer.writerow([count, int(txin[0])])
        in2add_writer.writerow([count, int(txin[1])])
        count += 1
    txin_data.close()

def txout_csv_process(year_path):
    txout_data = open(os.path.join(year_path, txout_file))
    txouts = txout_data.readlines()

    # creat csv files
    txout_csv = open(target_path + "/" + year_path[22:26] + "/TxOut.csv", "w")
    tx2out = open(target_path + "/" + year_path[22:26] + "/Tx2Out.csv", "w")
    out2add_csv = open(target_path + "/" + year_path[22:26] + "/Out2Add.csv", "w")
    txout_writer = csv.writer(txout_csv)
    tx2out_writer = csv.writer(tx2out)
    out2add_writer = csv.writer(out2add_csv)

    # write header
    txout_writer.writerow(['TxOutID:ID(TxOut)', 'TxID', 'AddID', 'Value'])
    tx2out_writer.writerow([':START_ID(Tx)', ':END_ID(TxOut)'])
    out2add_writer.writerow([':START_ID(TxOut)', ':END_ID(Address)'])

    # write data
    count = 0   # for TxOut index (Matching for Neo4j establishment)
    for txout in tqdm.tqdm(txouts):
        txout = txout.split()
        txout_writer.writerow([count, int(txout[0]), int(txout[1]), int(txout[2])])
        tx2out_writer.writerow([int(txout[0]), count])
        out2add_writer.writerow([count, int(txout[1])])
        count += 1
    txout_data.close()

def pd_test():
    data_txt = pd.read_csv('addresses.txt', sep=' ', header=None, names=['AddID:ID(Address)', 'AddPk'])
    Addcol = data_txt['AddPk']

    f = open(label_path)
    reader = csv.reader(f)
    labels = {'start': 0}
    for row in reader:
        labels[row[0]] = 1


    new_col = []
    count = 0
    for ad in tqdm.tqdm(Addcol):
        if ad in labels.keys():
            print(ad)
            print("WOW!")
            count += 1
    print(count)

    # new_row = ['0' for i in range(2513968)]
    # print(len(new_row))
    # data_txt['Mixer'] = new_row
    #
    # data_txt.to_csv('add.csv', index=False)

def annual_process(year_path):
    print(year_path[22:26] + " start processing...\nBlocks processing...")
    block_chain_csv_process(year_path)
    print("Blocks processing done, now starting processing Addresses...")
    address_csv_process(year_path)
    print("Addresses processing done, now starting processing Transactions...")
    tx_csv_process(year_path)
    print("Transactions processing done, now starting processing Txin and Txout...")
    txinout_csv_process(year_path)
    print(year_path[22:26] + " All Done!\n")

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
        tx_dict[tx[0]] = {'TxTime': tx[4]}

    # process txin
    txin_data = open(os.path.join(year_path, txin_file))
    txins = txin_data.readlines()

    # creat csv files
    txin_csv = open(target_path + "/" + year_path[22:26] + "/TxIn.csv", "w")
    in2tx_csv = open(target_path + "/" + year_path[22:26] + "/In2Tx.csv", "w")
    in2add_csv = open(target_path + "/" + year_path[22:26] + "/In2Add.csv", "w")
    txin_writer = csv.writer(txin_csv)
    in2tx_writer = csv.writer(in2tx_csv)
    in2add_writer = csv.writer(in2add_csv)

    # write header
    txin_writer.writerow(['TxInID:ID(TxIn)', 'TxID', 'AddID', 'Value', 'TxTime', 'AddPk', 'Mixer'])
    in2tx_writer.writerow([':START_ID(TxIn)', ':END_ID(Tx)'])
    in2add_writer.writerow([':START_ID(TxIn)', ':END_ID(Address)'])

    # write data
    count = 0  # for TxIn index (Matching for Neo4j establishment)
    for txin in tqdm.tqdm(txins):
        txin = txin.split()
        txin_writer.writerow([count, int(txin[0]), int(txin[1]), int(txin[2]),
                              int(tx_dict[txin[0]]['TxTime']),
                              add_dict[txin[1]]['AddPk'],
                              add_dict[txin[1]]['Mixer']])
        in2tx_writer.writerow([count, int(txin[0])])
        in2add_writer.writerow([count, int(txin[1])])
        count += 1
    txin_data.close()


    # process txout
    txout_data = open(os.path.join(year_path, txout_file))
    txouts = txout_data.readlines()

    # creat csv files
    txout_csv = open(target_path + "/" + year_path[22:26] + "/TxOut.csv", "w")
    tx2out = open(target_path + "/" + year_path[22:26] + "/Tx2Out.csv", "w")
    out2add_csv = open(target_path + "/" + year_path[22:26] + "/Out2Add.csv", "w")
    txout_writer = csv.writer(txout_csv)
    tx2out_writer = csv.writer(tx2out)
    out2add_writer = csv.writer(out2add_csv)

    # write header
    txout_writer.writerow(['TxOutID:ID(TxOut)', 'TxID', 'AddID', 'Value', 'TxTime', 'AddPk', 'Mixer'])
    tx2out_writer.writerow([':START_ID(Tx)', ':END_ID(TxOut)'])
    out2add_writer.writerow([':START_ID(TxOut)', ':END_ID(Address)'])

    # write data
    count = 0  # for TxOut index (Matching for Neo4j establishment)
    for txout in tqdm.tqdm(txouts):
        txout = txout.split()
        txout_writer.writerow([count, int(txout[0]), int(txout[1]), int(txout[2]),
                               int(tx_dict[txout[0]]['TxTime']),
                               add_dict[txout[1]]['AddPk'],
                               add_dict[txout[1]]['Mixer']])
        tx2out_writer.writerow([int(txout[0]), count])
        out2add_writer.writerow([count, int(txout[1])])
        count += 1
    txout_data.close()


if __name__ == "__main__":
    to_process_list = [data_2014, data_2015, data_2016]
    for pt in to_process_list:
        annual_process(pt)

    # block_chain_csv_process(data_2014)
    # address_csv_process(data_2014)
    # tx_csv_process(data_2014)
    # txinout_csv_process(data_2014)
