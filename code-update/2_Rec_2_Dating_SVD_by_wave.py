#%%
import sys
import os
import surprise
import pandas as pd
import matplotlib.pyplot as plt

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions

print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))


#%%

def runWave(df, wave):

    train, test = python_random_split(df, 0.75)

    # 'reader' is being used to get rating scale (for MovieLens, the scale is [1, 5]).
    # 'rating_scale' parameter can be used instead for the later version of surprise lib:
    # https://github.com/NicolasHug/Surprise/blob/master/surprise/dataset.py
    train_set = surprise.Dataset.load_from_df(train, reader=surprise.Reader(rating_scale=(1,10))).build_full_trainset()
    train_set

    svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=False)

    with Timer() as train_time:
        svd.fit(train_set)

    print("Took {} seconds for training.".format(train_time.interval))

    predictions = predict(svd, test, usercol='userID', itemcol='itemID')
    predictions.head()

    # 3.4 Remove rated items in the top k recommendations
    with Timer() as test_time:
        all_predictions = compute_ranking_predictions(svd, train, usercol='userID', itemcol='itemID', remove_seen=True)
        
    print("Took {} seconds for prediction.".format(test_time.interval))

    all_predictions.head()

    # 3.5 Evaluate how well SVD performs
    eval_rmse = rmse(test, predictions)
    eval_mae = mae(test, predictions)
    eval_rsquared = rsquared(test, predictions)
    eval_exp_var = exp_var(test, predictions)

    k = 3
    eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)

    print("============= Wave", wave, "============")

    print("RMSE:\t\t%f" % eval_rmse,
        "MAE:\t\t%f" % eval_mae,
        "rsquared:\t%f" % eval_rsquared,
        "exp var:\t%f" % eval_exp_var, sep='\n')

    print('----')

    print("MAP:\t%f" % eval_map,
        "NDCG:\t%f" % eval_ndcg,
        "Precision@K:\t%f" % eval_precision,
        "Recall@K:\t%f" % eval_recall, sep='\n')
    
    print("==========================================")

    return [eval_rmse, eval_mae, eval_rsquared, eval_exp_var, eval_map, eval_ndcg, eval_precision, eval_recall]


#%%

data = pd.read_csv("data/df_for_CF_cleaned.csv")
df_male = data[data["gender"]==1].loc[:,["wave","iid","pid","dec"]].reset_index(drop=True)
df_female = data[data["gender"]==0].loc[:,["wave","iid","pid","dec"]].reset_index(drop=True)

p_col = ["wave",'userID','itemID','rating']
df_male.columns = p_col
df_female.columns = p_col

# to rank females, uncomment the following line
df_male = df_female

rmse_list = list()
mae_list = list()
rsquared_list = list()
exp_var_list = list()
map_list = list()
ndcg_list = list()
precision_list = list()
recall_list = list()

wave_list = list(set(df_male["wave"]))
for wave in wave_list:
    cur_df = df_male[df_male["wave"]==wave].loc[:,["userID","itemID","rating"]]
    results = runWave(cur_df, wave)
    rmse_list.append(results[0])
    mae_list.append(results[1])
    rsquared_list.append(results[2])
    exp_var_list.append(results[3])
    map_list.append(results[4])
    ndcg_list.append(results[5])
    precision_list.append(results[6])
    recall_list.append(results[7])

#%%
rg = range(len(wave_list))
# plt.plot(rg, map_list, label="MAP")
# plt.plot(rg, ndcg_list, label="NDCG")
# plt.plot(rg, rmse_list, label="RMSE")
# plt.plot(rg, mae_list, label="MAE")
plt.plot(rg, precision_list, label="SVD")
als_prec = [0.543859649122807, 0.4215686274509804, 0.33333333333333337, 0.3425925925925926, 0.42592592592592593, 0.33333333333333337, 0.3020833333333333, 0.49122807017543857, 0.40170940170940184, 0.43137254901960775, 0.40650406504065045, 0.37037037037037024, 0.3888888888888889, 0.3508771929824561, 0.2252252252252252, 0.41025641025641024, 0.3623188405797101, 0.3333333333333333, 0.36781609195402293, 0.3846153846153846, 0.31060606060606066]
plt.plot(rg, als_prec, label="ALS")
plt.title("Precision for Top-3 Across All Waves")
plt.xlabel("Wave")
plt.ylabel("Precision")
# plt.plot(rg, recall_list, label="recall")
plt.ylim((0,1))
plt.legend()
plt.show()

# %%
def avg(input_list):
    return sum(input_list) / len(input_list)

avg_metrics = [avg(map_list), avg(ndcg_list), avg(rmse_list), avg(mae_list), avg(precision_list), avg(recall_list)]
names = ["Average MAP", "Average NDCG", "Average RMSE", "Average MAE", "Average Precision", "Average Recall"]

#%%
out_df = pd.DataFrame()
out_df["Metric"] = names
out_df["Average Value"] = avg_metrics

# %%
out_df
# %%
