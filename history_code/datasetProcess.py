"""
    @Author: Chenghui Li, SCUT. Started on Jan 21, 2021.
    This Program aims to establish a Neo4j graph database of Bitcoin Mixing transaction dataset,
    For final thesis project
    <Detection Bitcoin Anomaly Transaction Using Data Mining Techniques & Mixing Transaction Database Establishment>

    Objects in the database:
    1. [Node]
        a. 'Block'-- a block
            'BLockID': ID of the block.
            'BlockHash': Hash of the block.
            'NumTx': Number of transactions this block contains.
            'BlockTime': Time when this block was mined.
        b. 'Tx'-- a single transaction that in a block
            'TxID': ID of this transaction.
            'BlockID': ID of the block that this transaction was contained.
            'NumIn': Number of input(s) this transaction contains.
            'NumOut': Number of output(s) this transaction contains.
            'TxTime': Creation time of this transaction.
            'TxHash': Hash of the transaction.
        c. 'Address'-- Bitcoin account address
            'AddID': ID of this address(only for the convenience of the project, not originally from Bitcoin data).
            'AddPk': Public Key of this address.
            'Mixer': 1/0 Whether this address belongs to a mixing service.
        d. 'TxIn'-- Txin a transaction
            'TxID': ID of the transaction this TxIn belongs to.
            'Value': The Value of this TxIn.
            'AddID': The ID of the address that this TxIn belongs to.
        d. 'TxOut'-- TxOut a transaction
            'TxID': ID of the transaction this TxOut belongs to.
            'Value': The Value of this TxOut.
            'AddID': The ID of the address that this TxOut belongs to.
    2.[Relationship]
        a. 'CHAIN'-- Chain that connecting blocks
            [Block n]--[CHAIN]->[Block n-1]
            'ParentID': ID of the parent block.
            'SonID': ID of the son block.
        b. 'Tx2Block'-- Connecting transaction to block
            [Tx]--[Tx2Block]->[Block]
            'TxID': ID of the transaction
            'BlockID': ID of the Block
        c. 'In2Tx'--Connecting TxIn to transaction as an input
            [TxIn]--[In2TX]->[Tx]
        d. 'Tx2Output'--Connecting transaction to TxOut as an output
            [Tx]--[Tx2Output]->[TxOut]
        e. 'UXTO2Add'--Connecting UTXO to address
            [TxIn/TxOut]--[UXTO2Add]->[Address]
"""


from py2neo import Graph, Node, Relationship, NodeMatcher
import os
import tqdm

data_2014 = "../data_sysu/dataset1_2014_11_1500000"
data_2015 = "../data_sysu/dataset2_2015_6_1500000"
data_2016 = "../data_sysu/dataset3_2016_1_1500000"
addresses_file = "addresses.txt"
blockhash_file = "blockhash.txt"
transaction_file = "tx.txt"
txhash_file = "txhash.txt"
txin_file = "txin.txt"
txout_file = "txout.txt"

dataset_roots = [data_2014, data_2015, data_2016]

def database_establish():
    graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')


def annual_process(graph:Graph, annual_data_root):
    print(annual_data_root[22:26] + " starting processing...\n first process Blocks...")

    node_matcher = NodeMatcher(graph)    # Node matcher for existing check

    block_data = open(os.path.join(annual_data_root, blockhash_file))   # Block processing
    blocks = block_data.readlines()
    for block in tqdm.tqdm(blocks):
        block = block.split()
        if node_matcher.match('Block', BlockID=block[0]).count() == 0:   # never add a same block twice
            new_block = Node('Block', BlockID=block[0], BlockHash=block[1],
                             BlockTime=block[2], NumTx=block[3])
            graph.create(new_block)
            parent_matcher = node_matcher.match('Block', BlockID=str(int(block[0]) - 1))  # find a parent block
            if parent_matcher.count() > 0:
                parent_node = parent_matcher.first()
                r = Relationship(new_block, 'CHAIN', parent_node)   # Add Relationship
                r['ParentID'] = parent_node['BlockID']
                r['SonID'] = new_block['BlockID']
                graph.create(r)
    block_data.close()
    print("block processing done, now starting transactions...")


    transactions_data = open(os.path.join(annual_data_root, transaction_file))
    transactions = transactions_data.readlines()
    for tx in tqdm.tqdm(transactions):
        tx = tx.split()
        if node_matcher.match("Tx", TxID=tx[0]).count() == 0:  # Never add a same transaction twice
            new_tx = Node('Tx', TxID=tx[0], BlockID=tx[1], NumIn=tx[2],
                          NumOut=tx[3], TxTime=tx[4])
            graph.create(new_tx)
            block_matcher = node_matcher.match('Block', BlockID=tx[1])
            assert block_matcher.count() > 0        # raise Error if cannot find the block
            block = block_matcher.first()
            r = Relationship(new_tx, 'Tx2Block', block)
            r['TxID'] = tx[0]
            r['BlockID'] = block['BlockID']
            graph.create(r)
    transactions_data.close()
    tx_hash_data = open(os.path.join(annual_data_root, txhash_file))
    tx_hashs = tx_hash_data.readlines()
    for hash in tqdm.tqdm(tx_hashs):
        hash = hash.split()
        assert node_matcher.match('Tx', TxID=hash[0]).count() > 0
        tx = node_matcher.match('Tx', TxID=hash[0]).first()
        tx['Tx'] = hash[1]
    tx_hash_data.close()
    print("Transactions processing done, now starting processing Address...")

    addresses_data = open(os.path.join(annual_data_root, addresses_file))
    addresses = addresses_data.readlines()
    for add in tqdm.tqdm(addresses):
        add = add.split()
        if node_matcher.match("Address", AddPk=add[1]).count() == 0:
            new_add = Node('Address', AddID=add[0], AddPk=add[1])
            graph.create(new_add)
    addresses_data.close()
    print("Addresses processing done, now starting processing TxIn...")

    transaction_inputs_data = open(os.path.join(annual_data_root, txin_file))
    transaction_inputs = transaction_inputs_data.readlines()
    for txin in tqdm.tqdm(transaction_inputs):
        txin = txin.split()
        if node_matcher.match('TxIn', TxID=txin[0], AddID=txin[1], Value=txin[2]).count() == 0:
            new_UTXO = Node('TxIn', TxID=txin[0], AddID=txin[1], Value=txin[2])
            graph.create(new_UTXO)
            assert node_matcher.match('Address', AddID=txin[1]).count() > 0
            assert node_matcher.match('Tx', TxID=txin[0]) > 0
            tx = node_matcher.match('Tx', TxID=txin[0]).first()
            add = node_matcher.match('Address', AddID=txin[1]).first()
            r = Relationship(new_UTXO, 'In2Tx', tx)
            graph.create(r)
            r = Relationship(new_UTXO, 'UTXO2Add', add)
            graph.create(r)
    transaction_inputs_data.close()
    print("TxIn processing done, now starting processing TxOut...")

    transaction_output_data = open(os.path.join(annual_data_root, txout_file))
    transaction_outputs = transaction_output_data.readlines()
    for txout in transaction_outputs:
        txout = txout.split()
        if node_matcher.match('TxOut', TxID=txout[0], AddID=txout[1], Value=txout[2]).count() == 0:
            new_UTXO = Node('TxOut', TxID=txout[0], AddID=txout[1], Value=txout[2])
            graph.create(new_UTXO)
            assert node_matcher.match('Address', AddID=txout[1]).count() > 0
            assert node_matcher.match('Tx', TxID=txout[0]) > 0
            tx = node_matcher.match('Tx', TxID=txout[0]).first()
            add = node_matcher.match('Address', AddID=txout[1]).first()
            r = Relationship(tx, 'Tx2Out', new_UTXO)
            graph.create(r)
            r = Relationship(new_UTXO, 'UTXO2Add', add)
            graph.create(r)
    transaction_output_data.close()
    print(annual_data_root[22:26] + " All Done!")

if __name__ == "__main__":
    graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')
    annual_process(graph, data_2014)