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

def runMassey(df):

    wave_list = list(set(df['wave']))
    r_male_list = list()
    r_female_list = list()
    male_iids_list = list()
    female_iids_list = list()

    for wave in wave_list:

        df_sub = df[df["wave"]==wave]
        df_female = df_sub.loc[df_sub["gender"]==0].reset_index(drop=True)
        df_male = df_sub.loc[df_sub["gender"]==1].reset_index(drop=True)

        S_male, num_males, male_iids = buildScoreMatrix(df_female, df_male)
        r_male = getMasseyRating(S_male, num_males)

        S_female, num_females, female_iids = buildScoreMatrix(df_male, df_female)
        r_female = getMasseyRating(S_female, num_females)

        r_male_list.append(r_male)
        r_female_list.append(r_female)
        male_iids_list.append(male_iids)
        female_iids_list.append(female_iids)

    return wave_list, r_male_list, r_female_list, male_iids_list, female_iids_list

def buildScoreMatrix(df_partner, df_player):
    #  get the iid lists for each wave
    partner_iids = sorted(list(set(df_partner["iid"])))
    player_iids = sorted(list(set(df_player["iid"])))
    # build a score matrix S to record the results of games between males
    num_players = len(player_iids)
    S = np.zeros(shape=(num_players, num_players))

    for iid in partner_iids:
        pid_list = list(df_partner[df_partner["iid"]==iid]["pid"])
        score_list = list(df_partner[df_partner["iid"]==iid]["like"])
        for i in range(len(pid_list)):
            pid_1 = pid_list[i]
            row = player_iids.index(pid_1)
            row_score = score_list[i]
            for j in range(i+1,len(pid_list)):
                pid_2 = pid_list[j]
                col = player_iids.index(pid_2)
                col_score = score_list[j]
                S[row][col] += (row_score-col_score)
    
    return S, num_players, player_iids


def getMasseyRating(S, num_players): # Input: score matrix S, Output: Massey rating vector r
    
    num_of_games = 0
    game_list = list()
    v = list()
    # turn S into a game list and record the margin of vicotry v
    for i in range(num_players):
        for j in range(i, num_players):
            if S[i][j] != 0:
                num_of_games += 1
                if S[i][j] > 0: i_score = 1
                else: i_score = -1
                new_game = [i, j, i_score]
                game_list.append(new_game)
                v.append(abs(S[i][j]))

    # turn the game list into a game matrix B
    B = np.zeros(shape=(num_of_games+1, num_players))
    for k in range(num_of_games):
        new_game = game_list[k]
        i = new_game[0]
        j = new_game[1]
        i_score = new_game[2]
        j_score = - i_score
        B[k][i] = i_score
        B[k][j] = j_score

    # modify B and v to make the LS solvable
    v.append(0)
    for col in range(num_players):
        B[-1][col] = 1

    r = np.linalg.lstsq(B,v,rcond=None)[0]

    return r

#%%
if __name__ == "__main__":
    path = "data/Speed Dating Data.csv"
    in_df = readData(path)
    wave_list, r_male_list, r_female_list, male_iids_list, female_iids_list = runMassey(in_df)
    out_df = pd.DataFrame()
    out_df["wave"] = wave_list
    out_df["male_iids"] = male_iids_list
    out_df["massey_rating_male"] = r_male_list
    out_df["female_iids"] = female_iids_list
    out_df["massey_rating_female"] = r_female_list
    out_df.to_csv("massey_by_wave.csv", index=False)

out_df = readData("data/Speed Dating Data.csv")
out_df.to_csv("df_for_CF_cleaned.csv", index=False)



