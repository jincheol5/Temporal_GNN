from tgnn import DataUtils


data=DataUtils.DataLoader.load_from_pickle("test_20","graph")
print(data.keys())
# train_20.pop('all',None)
# print(train_20.keys())
# DataUtils.DataLoader.save_to_pickle(data=train_20,file_name='train_20',dir_type='graph')