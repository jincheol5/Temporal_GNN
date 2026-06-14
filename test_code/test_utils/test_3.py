import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TemporalGraphDataset

"""
<< Test >> 
utils.train_utils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. train_utils.TemporalGraphDataset
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            dataset=TemporalGraphDataset(df=df)
            for idx in range(5):
                src,dst,event_t=dataset.__getitem__(idx=idx)
                print(f"{idx} row values:")
                print(src)
                print(dst)
                print(event_t)

        case 2:
            """
            Test. train_utils.TemporalGraphDataset using PyTorch DataLoader
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            dataset=TemporalGraphDataset(df=df)
            loader=DataLoader(
                dataset=dataset,
                batch_size=4,
                shuffle=False
            )
            for batch_idx,(src,dst,event_t) in enumerate(loader):
                print(f"{batch_idx+1} batch values:")
                print(src)
                print(dst)
                print(event_t)
                break

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)