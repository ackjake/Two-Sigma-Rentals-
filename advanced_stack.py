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

    prior_0, prior_1, prior_2 = df_train[["low", "medium", "high"]].mean()

   # df_train.reset_index(inplace=True)

    index=list(range(df_train.shape[0]))
    random.seed(0)
    random.shuffle(index)

    df_train['manager_level_low'] = np.nan
    df_train['manager_level_medium'] = np.nan
    df_train['manager_level_high'] = np.nan

    for i in range(5):
        test_index = index[int((i*df_train.shape[0])/5):int(((i+1)*df_train.shape[0])/5)]
        train_index = list(set(index).difference(test_index)) 

        cv_train = df_train.iloc[train_index]
        cv_test  = df_train.iloc[test_index]

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
    feat_feats = pickle.load(open('../input/text_features.pkl', "rb"))
    features_to_use = pickle.load(open('../input/features_to_use.pkl', "rb"))
    train_df = pd.read_pickle('../input/processed_train_df.pkl')
    test_df = pd.read_pickle('../input/processed_test_df.pkl')

    y = train_df['interest_level']

    et_params1 = {
        'n_jobs': -1,
        'n_estimators':800,
        'max_features': None,
        'max_depth': None,
        'criterion':'gini',
        'kbest':135
    }

    rf_params1 = {
        'n_jobs': -1,
        'n_estimators':800,
        'max_features': 'sqrt',
        'max_depth': 90,
        'min_samples_leaf': 1,
        'criterion':'gini',
        'kbest':135
    }

    rf_params2 = {
        'n_jobs': -1,
        'n_estimators':800,
        'max_features': None,
        'max_depth': 10,
        'min_samples_leaf': 1,
        'min_samples_split':2,
        'criterion':'gini',
        'kbest':24
     }

    rf_params3 = {
        'n_jobs': -1,
        'n_estimators':800,
        'max_features': 'sqrt',
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'criterion':'gini',
        'kbest':193
     }

    rf_params4 = {
        'n_jobs': -1,
        'n_estimators':800,
        'max_features': 'sqrt',
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'criterion':'entropy',
        'kbest':193
     }


    xgb_params1 = {
        'colsample_bytree': 0.6,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'max_depth': 4,
        'min_child_weight': 1.92,
        'gamma':.055,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 10000000,
        'early_stopping_rounds': 25,
        'kbest':90
    }

    xgb_params2 = {
        'colsample_bytree': 0.43,
        'silent': 1,
        'subsample': 0.96,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'max_depth': 6,
        'min_child_weight':.005,
        'gamma':.17,
        'lambda':.0055,
        'alpha':.088,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 100000000,
        'early_stopping_rounds': 25,
        'kbest':203
    }

    xgb_params3 = {
        'colsample_bytree': 0.66,
        'silent': 1,
        'subsample': 0.76,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'max_depth': 8,
        'min_child_weight': 22.4,
        'gamma':.23,
        'lambda':.0002,
        'alpha':.03,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 1000000,
        'early_stopping_rounds': 25,
        'kbest':187
    }

    xgb_params4 = {
        'colsample_bytree': 0.52,
        'silent': 1,
        'subsample': .75,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'max_depth': 10,
        'min_child_weight': 19.92,
        'gamma':.5,
        'lambda':.84,
        'alpha':.26,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 1000000,
        'early_stopping_rounds': 25,
        'kbest':264
    }

    xgb_params5 = {
        'colsample_bytree': 0.52,
        'silent': 1,
        'subsample': .9,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'max_depth': 1,
        'min_child_weight': 1,
        'gamma':0,
        'lambda':1,
        'alpha':0,
        'eval_metric': 'mlogloss',
        'num_class':3,
        'nrounds': 1000000,
        'early_stopping_rounds': 25,
        'kbest':264
    }

    xg1 = XgbWrapper(seed=SEED, params=xgb_params1)
    xg2 = XgbWrapper(seed=SEED, params=xgb_params2)
    xg3 = XgbWrapper(seed=SEED, params=xgb_params3)
    xg4 = XgbWrapper(seed=SEED, params=xgb_params4)
    xg5 = XgbWrapper(seed=SEED, params=xgb_params5)

    rf1 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params1)
    rf2 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params2)
    rf3 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params3)
    rf4 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params4)

    et1 = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params1)

    clfs = [xg1, xg2, xg3, xg4, xg5, rf2, rf3, rf4, et1]

    dataset_blend_train = np.zeros((train_df.shape[0], 3*len(clfs)))
    dataset_blend_test = np.zeros((test_df.shape[0], 3*len(clfs)))
    skf = StratifiedKFold(NFOLDS, shuffle=True)

    for j, clf in enumerate(clfs):
        train_df, test_df = add_manager_targets2(train_df, test_df)
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

    np.save('../input/stacked_train1.npy', dataset_blend_train)
    np.save('../input/stacked_test1.npy', dataset_blend_test)

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
    out_df.to_csv("../output/advanced_stacked.csv", index=False)

if __name__ == '__main__':
    main()
