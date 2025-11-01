import networkx as nx
from model import TGAT
from model_train import ModelTrainer
from model_train_utils import ModelTrainUtils


config={
    'optimizer':'adam',
    'epochs':10,
    'early_stop':1,
    'patience':5,
    'lr':0.0005,
    'wandb':0
}

graph=nx.DiGraph()
graph.add_nodes_from([0,1,2,3])
graph.add_edge(0,1,t=[0.1,0.2])
graph.add_edge(1,2,t=[0.3,0.4])
graph.add_edge(2,3,t=[0.5,0.6])

train_data_loader=ModelTrainUtils.get_batch_loader(graph=graph,source_id=0,batch_size=2)
val_data_loader=ModelTrainUtils.get_batch_loader(graph=graph,source_id=0,batch_size=2)

model=TGAT(node_dim=1,latent_dim=32)

ModelTrainer.train(model=model,train_data_loader=train_data_loader,val_data_loader=val_data_loader,config=config)