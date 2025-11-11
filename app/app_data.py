from tgnn import DataUtils,ModelTrainUtils


"""
TO DO LIST:
1. temporal graph dataset 다시 생성
2. batch 어떻게 처리해서 효율적으로 구성할지
"""

graph_list_dict=DataUtils.DataLoader.load_from_pickle("train_20","graph")
data_loader_list_dict=ModelTrainUtils.get_data_loader_list_dict(graph_list_dict=graph_list_dict,random_src=False,batch_size=32)





