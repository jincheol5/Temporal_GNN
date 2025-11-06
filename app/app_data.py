from tgnn import DataUtils

val_20=DataUtils.DataLoader.load_from_pickle("val_20","graph")

for graph_type,graph_list in val_20.items():
    print(f"type: {graph_type}")
    print(f"list len: {len(graph_list)}")
    print(f"data type: {type(graph_list[0])}")