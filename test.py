import tqdm

tx_path = "../data_sysu/dataset1_2014_11_1500000/tx.txt"
tx_hash_path = "../data_sysu/dataset1_2014_11_1500000/txhash.txt"
txin_path = "../data_sysu/dataset1_2014_11_1500000/txin.txt"
txout_path = "../data_sysu/dataset1_2014_11_1500000/txout.txt"
address_path = "../data_sysu/dataset1_2014_11_1500000/addresses.txt"
block_hah_path = "../data_sysu/dataset1_2014_11_1500000/blockhash.txt"

def get_info_by_index(files, index, target_column):
    infos = []

    for file in files:
        file = file.split()
        if file[target_column] == index:
            infos.append(file)

    return infos


if __name__ == "__main__":
    tx = open(tx_path).readlines()
    tx_hash = open(tx_hash_path).readlines()
    txin = open(txin_path).readlines()
    txout = open(txout_path).readlines()
    address = open(address_path).readlines()
    block_hash = open(block_hah_path).readlines()

    sample_tx = tx[1].split()
    sample_tx_ID = sample_tx[0]
    sample_tx_block = sample_tx[1]
    sample_tx_time = sample_tx[4]
    sample_tx_hash = get_info_by_index(tx_hash, sample_tx[0], 0)[0][1]


    sample_input_info = get_info_by_index(txin, sample_tx_ID, 0)
    sample_output_info = get_info_by_index(txout, sample_tx_ID, 0)

    print(sample_input_info)
    for inpt in tqdm.tqdm(sample_input_info):
        inpt[1] = get_info_by_index(address, inpt[1], 0)[0][1]
    for otpt in tqdm.tqdm(sample_output_info):
        otpt[1] = get_info_by_index(address, otpt[1], 0)[0][1]

    print("transaction block: " + sample_tx_block)
    print("transaction ID: " + sample_tx_ID)
    print("transaction Hash: " + sample_tx_hash)
    print("transaction time: " + sample_tx[-1])
    print("transaction Inputs: ")
    for ele in sample_input_info:
        print("Address: " + ele[1] + ";  value: "+ ele[-1])
    print("transaction Outputs: ")
    for ele in sample_output_info:
        print("Address: " + ele[1] + ";  value: " + ele[-1])
