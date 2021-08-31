from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd



if __name__ == "__main__":
    graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')


    node_matcher = NodeMatcher(graph)
    persons = node_matcher.match('Tx', TxTime='1')

    # match_str = "match (n:Address{Mixer:'1'}) return n"

    # match_str = "match (h:Address)<-[i:In2Add]-(j:TxIn)-[k:In2Tx]->(n:Tx{TxID:'50230631'})" \
    #             " return h,i,j,k,n"

    match_str = "match (tx:Tx) where '1414771239'<=tx.TxTime<='1414771715' return tx"
    result = graph.run(match_str)
    a = result.data()
    print(len(a))
    for d in a:
        print(d['tx']['TxHash'])
        print("\n\n")










