from tgnn import DataUtils


data=DataUtils.DataLoader.load_from_pickle("test_20","graph")
print(data.keys())
data.pop('all',None)
print(data.keys())
DataUtils.DataLoader.save_to_pickle(data=data,file_name='test_20',dir_type='graph')