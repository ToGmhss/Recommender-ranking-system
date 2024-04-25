# %%
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

# %%
def readData(path):
    df_raw = pd.read_csv(path, encoding="gbk")
    df_sub = df_raw.loc[:,["wave","iid","gender","pid","match","like", "like_o", "dec","dec_o"]]

    df_clean = df_sub.dropna(axis=0, how='any')
    df_clean[['pid']] = df_clean[['pid']].astype(int)

    return df_clean

# %%
def runAvgDom(df):

    wave_list = list(set(df['wave']))
    r_male_list = list()
    r_female_list = list()
    male_iids_list = list()
    female_iids_list = list()

    for wave in wave_list:

        df_sub = df[df["wave"]==wave]
        df_female = df_sub.loc[df_sub["gender"]==0].reset_index(drop=True)
        df_male = df_sub.loc[df_sub["gender"]==1].reset_index(drop=True)

        avg_dom_female, female_iids = getAvgDominance(df_female)
        avg_dom_male, male_iids = getAvgDominance(df_male)

        r_female_list.append(avg_dom_female)
        r_male_list.append(avg_dom_male)
        male_iids_list.append(male_iids)
        female_iids_list.append(female_iids)

    return wave_list, r_male_list, r_female_list, male_iids_list, female_iids_list
            
        

def getAvgDominance(df_player):

    player_iids = sorted(list(set(df_player["iid"])))
    num_of_games = len(player_iids)
    avg_dom_vec = list()
    for i in range(num_of_games):
        iid = player_iids[i]
        i_score = sum(df_player[df_player["iid"]==iid]["like_o"])
        avg_dom = 0
        for j in range(num_of_games):
            jid = player_iids[j]
            j_score = sum(df_player[df_player["iid"]==jid]["like_o"])
            avg_dom += i_score - j_score
        avg_dom /= num_of_games
        avg_dom_vec.append(avg_dom)

    avg_dom_np = np.array(avg_dom_vec)
    avg_dom_np = np.reshape(avg_dom_np, (num_of_games,1))
    M = np.ones(shape=(num_of_games, num_of_games))
    k = 1
    while True:
        k += 1
        # print("Current generation:", k)
        addition = np.dot((M/num_of_games)**k, avg_dom_np)
        avg_dom_np += addition
        if np.sum(addition) < 1e-6:
            if k != 2:
                print("ITERATION!", k)
            break

    return avg_dom_np, player_iids
    

#%%
if __name__ == "__main__":
    path = "data/Speed Dating Data.csv"
    in_df = readData(path)
    wave_list, r_male_list, r_female_list, male_iids_list, female_iids_list = runAvgDom(in_df)
    
    out_df = pd.DataFrame()
    out_df["wave"] = wave_list
    out_df["male_iids"] = male_iids_list
    out_df["dominance_rating_male"] = r_male_list
    out_df["female_iids"] = female_iids_list
    out_df["dominance_rating_female"] = r_female_list
    out_df.to_csv("avgdom_cvg_by_wave.csv", index=False)

