import networkx as nx
from model_train_utils import ModelTrainUtils

graph=nx.DiGraph()
graph.add_nodes_from([0,1,2])
graph.add_edge(0,1,t=[0.2,0.3])
graph.add_edge(1,2,t=[0.1])

batch_loader=ModelTrainUtils.get_batch_loader(graph=graph,source_id=0,batch_size=1)

for key,value in batch_loader.items():
    print(f"key: {key}")
    print(f"value: {value}")