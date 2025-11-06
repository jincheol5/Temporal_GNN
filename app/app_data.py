from tgnn import DataUtils


train_20=DataUtils.DataLoader.load_from_pickle("train_20","graph")
print(train_20.keys())
train_20.pop('all',None)
print(train_20.keys())
DataUtils.DataLoader.save_to_pickle(data=train_20,file_name='train_20',dir_type='graph')