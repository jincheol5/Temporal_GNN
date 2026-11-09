from utils import DataUtils


data=DataUtils.preprocess_dataset(dataset_name=f"enron")
graph_df=data["graph_df"]

# 동일 시각의 역방향 Interaction 확인
reverse_df = graph_df[["u", "i", "t"]].rename(
    columns={"u": "i", "i": "u"}
)

reverse_pairs = graph_df.merge(
    reverse_df.drop_duplicates(),
    on=["u", "i", "t"],
    how="inner"
)

reverse_pairs = reverse_pairs[
    reverse_pairs["u"] != reverse_pairs["i"]
]

print("역방향 Interaction이 존재하는 행 수:", len(reverse_pairs))