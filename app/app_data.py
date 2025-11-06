from tgnn import DataUtils

val_20=DataUtils.DataLoader.load_from_pickle("val_20","graph")

print(val_20.keys())

# val_20=val_20.pop('all',None)
# DataUtils.DataLoader.save_to_pickle(data=val_20,file_name='val_20',dir_type='graph')


# train_20=DataUtils.DataLoader.load_from_pickle("train_20","graph")

# for graph_type,graph_list in train_20.items():
#     print(f"type: {graph_type}")
#     print(f"list len: {len(graph_list)}")
#     print(f"data type: {type(graph_list[0])}")