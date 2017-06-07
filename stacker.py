import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
import pickle
import random
import xgboost as xgb


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.kbest = params.pop('kbest')
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_test, y_test ):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds')
        self.early_stopping = params.pop('early_stopping_rounds')
        self.kbest = params.pop('kbest')
        self.gbdt = None

    def train(self, x_train, y_train, x_test, y_test, num_boost_round=None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, early_stopping_rounds=self.early_stopping, verbose_eval=False)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def add_manager_targets2(df_train, df_test):
    df_train.reset_index(inplace=True)
    prior_0, prior_1, prior_2 = df_train[["high", "medium", "low"]].mean()

    index = list(range(df_train.shape[0]))
    random.seed(0)
    random.shuffle(index)

    df_train['manager_level_low'] = np.nan
    df_train['manager_level_medium'] = np.nan
    df_train['manager_level_high'] = np.nan

    for i in range(5):
        test_index = index[int((i * df_train.shape[0]) / 5):int(((i + 1) * df_train.shape[0]) / 5)]
        train_index = list(set(index).difference(test_index))

        cv_train = df_train.iloc[train_index]
        cv_test = df_train.iloc[test_index]

        for m in cv_train.groupby('manager_id'):
            test_subset = cv_test[cv_test.manager_id == m[0]].index

            df_train.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 0).mean()
            df_train.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 1).mean()
            df_train.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 2).mean()

    # now for the test data

    df_test['manager_level_low'] = np.nan
    df_test['manager_level_medium'] = np.nan
    df_test['manager_level_high'] = np.nan

    for m in df_train.groupby('manager_id'):
        test_subset = df_test[df_test.manager_id == m[0]].index

        df_test.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 0).mean()
        df_test.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 1).mean()
        df_test.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 2).mean()

    for table in [df_train, df_test]:
        table['manager_level_low'] = table['manager_level_low'].fillna(prior_0)
        table['manager_level_medium'] = table['manager_level_medium'].fillna(prior_1)
        table['manager_level_high'] = table['manager_level_high'].fillna(prior_2)

    return df_train, df_test


def main():

    NFOLDS = 5
    SEED = 0

    tr_sparse_features = np.load('../input/train_text.npy')
    te_sparse_features = np.load('../input/test_text.npy')
    features_to_use = pickle.load(open("../input/experiment_feats.pkl", "rb"))
    feat_feats = pickle.load(open('../input/text_features.pkl', "rb"))
    train_df = pd.read_pickle('../input/experiment_train.pkl')
    test_df = pd.read_pickle('../input/experiment_test.pkl')

    train_df, test_df = add_manager_targets2(train_df, test_df)
    y = train_df['interest_level']
    for table in [train_df, test_df]:
        table['avg_areas'] = table['avg_areas'].fillna(0)
        table['avg_height'] = table['avg_height'].fillna(0)
        table['avg_width'] = table['avg_width'].fillna(0)
        table['stdpm'] = table['stdpm'].fillna(0)

    et_params = {
        'n_jobs': -1,
        'n_estimators': 475,
        'max_features': None,
        'max_depth': 90,
        'min_samples_leaf': 2,
        'kbest':135
    }

    rf_params = {
        'n_jobs': -1,
        'n_estimators': 475,
        'max_features': 'sqrt',
        'max_depth': 90,
        'min_samples_leaf': 2,
        'kbest':135
    }

    xgb_params = {
        'colsample_bytree': 0.6,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.01,
        'objective': 'multi:softprob',
        'max_depth': 4,
        'min_child_weight': 1.92,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 10000,
        'early_stopping_rounds': 25,
        'kbest':90
    }

    xg = XgbWrapper(seed=SEED, params=xgb_params)
    et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

    skf = StratifiedKFold(NFOLDS, shuffle=True)

    clfs = [xg, et, rf]

    dataset_blend_train = np.zeros((train_df.shape[0], 3*len(clfs)))
    dataset_blend_test = np.zeros((test_df.shape[0], 3*len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        cv_scores = []
        dataset_blend_test_low = np.zeros((test_df.shape[0], NFOLDS))
        print(dataset_blend_test_low.shape)
        dataset_blend_test_med = np.zeros((test_df.shape[0], NFOLDS))
        dataset_blend_test_high = np.zeros((test_df.shape[0], NFOLDS))
        for i, (train, test) in enumerate(skf.split(train_df, y)):
            print("Fold{}".format(i))
            x_train, x_test = train_df.iloc[train], train_df.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            tr_feat, te_feat = tr_sparse_features[train], tr_sparse_features[test]

            skb = SelectKBest(chi2, k=clf.kbest)
            tr_sp_feats = skb.fit_transform(tr_feat, y_train)
            te_sp_feats = skb.transform(te_feat)
            test_df_feats = skb.transform(te_sparse_features)

            final_train_x = np.hstack((x_train[features_to_use], tr_sp_feats))
            final_test_x = np.hstack((x_test[features_to_use], te_sp_feats))
            final_test_df = np.hstack((test_df[features_to_use], test_df_feats))

            clf.train(final_train_x, y_train, final_test_x, y_test)
            y_submission = clf.predict(final_test_x)
            cv_scores.append(log_loss(y_test, y_submission))
            test_preds = clf.predict(final_test_df)
            dataset_blend_test_low[:, i] = test_preds[:, 0]
            dataset_blend_test_med[:, i] = test_preds[:, 1]
            dataset_blend_test_high[:, i] = test_preds[:, 2]
            dataset_blend_train[test, j*3:j*3+3] = y_submission

        print('{} scored {}'.format(clf, np.mean(cv_scores)))

        dataset_blend_test[:, j*3] = dataset_blend_test_low.mean(1)
        dataset_blend_test[:, j*3+1] = dataset_blend_test_med.mean(1)
        dataset_blend_test[:, j*3+2] = dataset_blend_test_high.mean(1)

    print('Predicting Blend')
    params = {'eta':.01, 'colsample_bytree':1, 'subsample':1, 'seed':0, 'nthread':-1,
                'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':3, 'silent':1,
                          'min_child_weight':1, 'max_depth':4,
                          }
    dtrain = xgb.DMatrix(dataset_blend_train, label=y)
    bst = xgb.cv(params, dtrain, 999999, 5, early_stopping_rounds=10, verbose_eval=200)
    best_rounds = np.argmin(bst['test-mlogloss-mean'])
    model = xgb.train(params, dtrain, best_rounds, verbose_eval=False)
    preds = model.predict(xgb.DMatrix(dataset_blend_test))

    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = test_df.listing_id.values
    out_df.to_csv("../output/first_stacked.csv", index=False)

if __name__ == '__main__':
    main()
