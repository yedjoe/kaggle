from os.path import join
from pathlib import Path
from typing import List, Tuple, Union

from pandas import concat, DataFrame, read_csv, Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Variables
data_dir: str = join(Path(__file__).parent.absolute(), 'data')
train_path: str = join(data_dir, 'train.csv')
test_path: str = join(data_dir, 'test.csv')
submission_path: str = join(data_dir, 'submission.csv')
features: list = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
target: str = 'Survived'
pk: str = 'PassengerId'
random_state: int = 8
mea: DataFrame = DataFrame({ 'model_id': [], 'mea': [] }).set_index('model_id')

# Load Dataset
train: DataFrame = read_csv(train_path, index_col=pk)
test: DataFrame = read_csv(test_path, index_col=pk)

# Analyse the data
missing_ft_prcnt: Series = (train.drop(target, axis='columns').isna().sum() * 100 / len(train.index)).astype('float64').round(decimals=2)
print(missing_ft_prcnt[missing_ft_prcnt > 0].sort_values(ascending=False))

## TRAIN
# Split train & trial dataset
train_x: DataFrame
train_y: Series
trial_x: DataFrame
trial_y: Series

train_x, trial_x, train_y, trial_y = train_test_split(
    train[features],
    train[target],
    random_state=random_state
)

# # Option 1: Drop NA
# dropped_train_df = train_df.dropna(axis='index', subset=[target])

# # Create models
# rf_models: Tuple[RandomForestRegressor] = (
#     RandomForestRegressor(n_estimators=10, random_state=random_state),
#     RandomForestRegressor(n_estimators=100, random_state=random_state),
#     RandomForestRegressor(n_estimators=1000, random_state=random_state),
#     RandomForestRegressor(n_estimators=10000, random_state=random_state),
#     RandomForestRegressor(n_estimators=20000, random_state=random_state),
#     RandomForestRegressor(n_estimators=10000, criterion='mae', random_state=random_state),
#     RandomForestRegressor(n_estimators=10000, min_samples_split=20, random_state=random_state), # Smallest MEA
#     RandomForestRegressor(n_estimators=10000, max_depth=7, random_state=random_state)
# )

# results: List[List[Union[int, float]]] = []

# for num, rf_model in enumerate(rf_models):
#     print(f'Model Num. {num}')

#     # Fit models
#     rf_model.fit(train_features_df, train_y_s)

#     # Predict with trial dataset
#     results.append([
#         num,
#         mean_absolute_error(
#             list(map(lambda prediction: int(round(prediction)), rf_model.predict(trial_features_df))),
#             trial_y_s
#         )
#     ])

# mea_df: DataFrame = DataFrame(results, columns=['model_num', 'mea']) \
#     .set_index('model_num') \
#     .sort_values('mea')

# print(mea_df)

# 2nd Option: Imputation
mean_impt: SimpleImputer = SimpleImputer(strategy='mean')
imputed_train_x = mean_impt.fit_transform(train_x)
imputed_trial_x = mean_impt.transform(trial_x)

# Create models
rf_models: Tuple[RandomForestRegressor] = (
    RandomForestRegressor(n_estimators=10000, random_state=random_state),
    RandomForestRegressor(n_estimators=10000, criterion='mae', random_state=random_state),
    RandomForestRegressor(n_estimators=10000, min_samples_split=20, random_state=random_state),
    RandomForestRegressor(n_estimators=10000, max_depth=7, random_state=random_state) # Smallest MEA
)

results: List[List[Union[int, float]]] = []

for num, rf_model in enumerate(rf_models):
    print(f'Model Num. {num}')

    # Fit models
    rf_model.fit(train_x, train_y)

    # Predict with trial dataset
    results.append((
        f'imp{num}',
        mean_absolute_error(
            list(map(lambda prediction: int(round(prediction)), rf_model.predict(trial_x))),
            trial_y
        )
    ))

mea = concat([
    mea,
    DataFrame(results, columns=['model_id', 'mea']).set_index('model_id')
], axis='index').sort_values(by='mea')

print(mea)


## TEST
# rf_model = train_random_forest()
# rf_model.fit(train_df[features], train_df[target])

# test_features_df = test_df[features].dropna()

# pred_df = concat([
#     test_df,
#     DataFrame({ target: map(lambda pred: int(round(pred)), rf_model.predict(test_features_df)) }, index=test_features_df.index)
# ], axis='columns', join='outer').sort_index()

# pred_df[target].to_csv(SUBMISSION_PATH, float_format='%g')
