from tqdm import tqdm
from tgnn import DataUtils,GraphUtils


"""
TO DO LIST:
1. temporal graph dataset 다시 생성
2. batch 어떻게 처리해서 효율적으로 구성할지
"""

graph_list_dict=DataUtils.DataLoader.load_from_pickle("train_20","graph")

for graph_type,graph_list in tqdm(graph_list_dict.items()):
    dataset_dict_list=GraphUtils.compute_dataset_dict_list(graph_list=graph_list)





