#%%
import sys
import surprise
import pandas as pd

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions

print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))

#%%

data = pd.read_csv("data/df_for_CF_cleaned.csv")
df_all = data.loc[:,["iid","pid","dec"]].reset_index(drop=True)

p_col = ['userID','itemID','rating']
df_all.columns = p_col


#%%
train, test = python_random_split(df_all, 0.75)

# 'reader' is being used to get rating scale (for MovieLens, the scale is [1, 5]).
# 'rating_scale' parameter can be used instead for the later version of surprise lib:
# https://github.com/NicolasHug/Surprise/blob/master/surprise/dataset.py
train_set = surprise.Dataset.load_from_df(train, reader=surprise.Reader(rating_scale=(0,1))).build_full_trainset()
train_set

# %%
svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=False)

with Timer() as train_time:
    svd.fit(train_set)

print("Took {} seconds for training.".format(train_time.interval))

# %%
predictions = predict(svd, test, usercol='userID', itemcol='itemID')

# %% Remove rated items in the top k recommendations
with Timer() as test_time:
    all_predictions = compute_ranking_predictions(svd, train, usercol='userID', itemcol='itemID', remove_seen=True)
    
print("Took {} seconds for prediction.".format(test_time.interval))


# %% Evaluate how well SVD performs
eval_rmse = rmse(test, predictions)
eval_mae = mae(test, predictions)
eval_rsquared = rsquared(test, predictions)
eval_exp_var = exp_var(test, predictions)

k = 3
eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)


print("RMSE:\t\t%f" % eval_rmse,
      "MAE:\t\t%f" % eval_mae,
      "rsquared:\t%f" % eval_rsquared,
      "exp var:\t%f" % eval_exp_var, sep='\n')

print('----')

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
