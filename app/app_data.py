from tgnn import DataUtils


data=DataUtils.DataLoader.load_from_pickle("test_20","graph")
print(data.keys())
data.pop('all',None)
print(data.keys())
DataUtils.DataLoader.save_to_pickle(data=data,file_name='test_20',dir_type='graph')

"""
TO DO LIST:
1. temporal graph dataset 다시 생성
2. batch 어떻게 처리해서 효율적으로 구성할지
"""