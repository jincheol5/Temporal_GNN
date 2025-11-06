from tgnn import DataUtils

val_20=DataUtils.DataLoader.load_from_pickle("val_20","graph")

print(type(val_20))