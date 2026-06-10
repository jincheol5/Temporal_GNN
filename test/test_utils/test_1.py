import argparse
from utils import DataUtils

"""
<< Test >> 
utils.data_utils.DataUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. DataUtils.preprocess_dataset_to_df
            """
            df=DataUtils.preprocess_dataset_to_df(dataset_name=f"CollegeMsg")
            print(df)

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