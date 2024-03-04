from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

regular_season_data_url = "https://raw.githubusercontent.com/samirk786/whartonNSL/main/NSL_Regular_Season_Data.csv"
group_round_games_url = "https://raw.githubusercontent.com/samirk786/whartonNSL/main/NSL_Group_Round_Games.csv"

regular_season_data = pd.read_csv(regular_season_data_url)
group_round_games = pd.read_csv(group_round_games_url)

regular_season_data.dropna(subset=['HomeScore', 'AwayScore', 'Home_xG', 'Away_xG', 'Home_shots', 'Away_shots'], inplace=True)
regular_season_data = regular_season_data.drop(columns=["Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20"])

df = regular_season_data
df2 = group_round_games

df['HomeScore'] = df['HomeScore'].astype(float)
df['AwayScore'] = df['AwayScore'].astype(float)
df['Homewin'] = df['HomeScore'] - df['AwayScore'] > 0
df['Awaywin'] = df['HomeScore'] - df['AwayScore'] < 0
df['draw'] = df['HomeScore'] - df['AwayScore'] == 0

def determine_match_outcome(row):
    if row['Homewin']:
        return "homewin"
    elif row['Awaywin']:
        return "awaywin"
    else:
        return "draw"

df['MatchOutcome'] = df.apply(determine_match_outcome, axis=1)

# Drop the redundant columns
df.drop(columns=['Homewin', 'Awaywin', 'draw'], inplace=True)

# Split the data into train and test sets
train, test = train_test_split(df, test_size=0.2)

# Load the datasets
train_data = TabularDataset(train)
test_data = TabularDataset(test)

label = 'MatchOutcome'

predictor = TabularPredictor(label=label).fit(train_data)

y_pred = predictor.predict(test_data)
predictor.evaluate(test_data, silent=True)
