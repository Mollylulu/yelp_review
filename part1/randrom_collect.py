import json
from pandas.io.json import json_normalize
import pandas as pd


DATA_FILE="yelp_academic_dataset_review"

with open("./{filename}.json".format(filename=DATA_FILE), "rb") as f:
    data_list = [json.loads(line) for line in f]

    # normalize data to Dataframe
    review_list = json_normalize(data_list)#pandas.DataFrame

    #print(review_list['asin'].sample(n=3,replace=False))
    sampled_id=review_list['business_id'].drop_duplicates().sample(n=200,replace=False)
    ret=review_list[review_list['business_id'].isin(sampled_id.values)]
    ret.to_csv("sampled_data.csv")