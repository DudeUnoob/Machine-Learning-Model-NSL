from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


regular_season_data_url = "https://raw.githubusercontent.com/samirk786/whartonNSL/main/NSL_Regular_Season_Data.csv"
group_round_games_url = "https://raw.githubusercontent.com/samirk786/whartonNSL/main/NSL_Group_Round_Games.csv"


regular_season_data = pd.read_csv(regular_season_data_url)
group_round_games = pd.read_csv(group_round_games_url)


regular_season_data.dropna(subset=['HomeScore', 'AwayScore', 'Home_xG', 'Away_xG', 'Home_shots', 'Away_shots'], inplace=True)


regular_season_data = regular_season_data.drop(columns=["Unnamed: 17", "Unnamed: 18", "Unnamed: 19", "Unnamed: 20", ])

df = regular_season_data


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


df.drop(columns=['Homewin', 'Awaywin', 'draw'], inplace=True)


train, test = train_test_split(df, test_size=0.2)


train_data = TabularDataset(train)

#Notice the change here where we don't use feature engineering in the prediction
test_data = pd.DataFrame(test)

# Specify target label
label = 'MatchOutcome'

# Train the model
predictor = TabularPredictor(label=label).fit(train_data)

# Make predictions
y_pred = predictor.predict(test_data.drop(columns=[label]))

# Evaluate model performance
predictor.evaluate(test_data, silent=True)
