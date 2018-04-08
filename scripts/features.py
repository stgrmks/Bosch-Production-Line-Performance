author = 'MSteger'

import numpy as np
import pandas as pd
import gc, os
import itertools

def noDupe_date(filepath):
    features = pd.read_csv(filepath, nrows = 1).columns.tolist()
    seen = np.zeros(52)
    mask = []

    for feature in features:
        if feature in ['Id', 'S24', 'S25']:
            mask.append(feature)
            continue

        station = int(feature.split('_')[1][1:])
        if seen[station]: continue
        seen[station] = 1
        mask.append(feature)

    return mask

def cat_encoder(filepaths, features, na_dummy = 6666666):
    train_test = pd.DataFrame()

    for filepath in filepaths:
        tmp = pd.read_csv(filepath, usecols = features)
        train_test = pd.concat([train_test, tmp], axis = 0)
        del tmp
        gc.collect()

    train_test.fillna(na_dummy, inplace = True)
    return pd.get_dummies(train_test)


def compute_date_features(filepaths, features, na_dummy = 6666666):
    train_test = pd.DataFrame()

    for filepath in filepaths:
        train_test = pd.concat([train_test, pd.read_csv(filepath, usecols = features)], axis = 0)

    features = list(set(features) - set(['Id']))

    date_features = train_test.Id.copy().to_frame()
    date_features['minDate'] = train_test[features].min(axis = 1)
    date_features['maxDate'] = train_test[features].max(axis = 1)
    date_features['avgDate'] = train_test[features].mean(axis = 1)
    date_features['medDate'] = train_test[features].median(axis = 1)
    date_features['stdDate'] = train_test[features].std(axis = 1)
    date_features['minDate_maxDate_min'] = date_features.groupby(by=['minDate'])['maxDate'].transform('min')
    date_features['minDate_maxDate_max'] = date_features.groupby(by=['minDate'])['maxDate'].transform('max')
    date_features['maxDate_minDate_min'] = date_features.groupby(by=['maxDate'])['minDate'].transform('min')
    date_features['maxDate_minDate_max'] = date_features.groupby(by=['maxDate'])['minDate'].transform('max')

    lst = ['minDate', 'maxDate', 'minDate_maxDate_min', 'minDate_maxDate_max', 'maxDate_minDate_min', 'maxDate_minDate_max']
    for i, j in enumerate(list(itertools.combinations(lst, 2))):
        date_features['duration' + str(i + 1)] = date_features[j[0]] - date_features[j[1]]
        date_features[j[0] + '_' + j[1] + '_Same'] = ((date_features[j[0]] == date_features[j[1]]) * 1).astype(np.int8)

    for col in features:
        date_features['Station-' + col.split('_')[1]] = (train_test[col].notnull() * 1).astype(np.int8)

    date_features['StationLen'] = date_features[[col for col in date_features.columns if 'Station' in col]].sum(axis=1).astype(np.int8)

    startDate_Id = date_features[['Id', 'minDate']].copy()
    startDate_Id = startDate_Id.sort_values(by = ['minDate', 'Id'], ascending = True)
    for i in range(1, 6):
        startDate_Id['diff' + str(i)] = startDate_Id['Id'].diff(periods=i).fillna(na_dummy).astype(np.int8)
        startDate_Id['rev_diff' + str(i)] = startDate_Id['Id'].diff(periods=-i).fillna(na_dummy).astype(np.int8)

    date_features = date_features.merge(startDate_Id.drop('minDate', axis=1), on = 'Id')

    return date_features.fillna(na_dummy)

def compute_numeric_features(filepaths, features, na_dummy = 6666666):
    train_test = pd.DataFrame()

    for filepath in filepaths:
        train_test = pd.concat([train_test, pd.read_csv(filepath, usecols=features)], axis=0)

    train_test.fillna(na_dummy, inplace = True)
    train_test['dupe'] = train_test.drop('Id', axis=1).duplicated() * 1
    train_test['hash'] = train_test.drop('Id', axis=1).apply(lambda x: hash(tuple(x)), axis=1)
    train_test['dupe_count'] = train_test.groupby(['hash'])['hash'].transform('count')
    train_test.drop('hash', axis=1, inplace=True)

    return train_test.drop('Id', axis = 1)

if __name__ == '__main__':
    data_folder = r'/data/'
    date_filepaths = [os.path.join(data_folder, 'train_date.csv.gz'), os.path.join(data_folder, 'test_date.csv.gz')]
    cat_features = ['L1_S24_F1559', 'L3_S32_F3851', 'L1_S24_F1827', 'L1_S24_F1582', 'L3_S32_F3854', 'L1_S24_F1510', 'L1_S24_F1525', 'Id']
    cat_filepaths = [os.path.join(data_folder, 'train_categorical.csv.gz'), os.path.join(data_folder, 'test_categorical.csv.gz')]
    numeric_filepaths = [os.path.join(data_folder, 'train_numeric.csv.gz'), os.path.join(data_folder, 'test_numeric.csv.gz')]
    numeric_cols = ['Id', 'L1_S24_F1846', 'L3_S32_F3850', 'L1_S24_F1695', 'L1_S24_F1632', 'L3_S33_F3855', 'L1_S24_F1604', 'L3_S29_F3407', 'L3_S33_F3865', 'L3_S38_F3952', 'L1_S24_F1723']

    dates_noDupe = noDupe_date(filepath = date_filepaths[0])
    date_features = compute_date_features(filepaths = date_filepaths, features = dates_noDupe)
    categorical_features = cat_encoder(filepaths = cat_filepaths, features = cat_features)
    numeric_features = compute_numeric_features(filepaths = numeric_filepaths, features = numeric_cols)

    X = numeric_features.merge(categorical_features, on = 'Id').merge(date_features, on = 'Id')
    y = pd.read_csv(os.path.join(data_folder, 'train_numeric.csv.gz'), usecols = ['Response'])['Response'].astype(np.int8)

    X[:y.shape[0]].to_hdf(os.path.join(data_folder, 'train_features.hdf'), key = 'features', format = 'table')
    X[y.shape[0]:].to_hdf(os.path.join(data_folder, 'test_features.hdf'), key = 'features', format = 'table')

