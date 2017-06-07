import numpy as npdfdfadf:
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utilsi

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import pickle
import xgboost as xgb


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_test, y_test ):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)


class KerasWrapper(object):
    def __init__(self, model):
        self.clf = model

    def train(self, x_train, y_train, x_test, y_test):
        self.clf.fit(x_train, np_utils.to_categorical(y_train))

    def predict(self, x):
        return self.clf.evaluate(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds')
        self.early_stopping = params.pop('early_stopping_rounds')
        self.gbdt = None

    def train(self, x_train, y_train, x_test, y_test, num_boost_round=None):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist,
                              early_stopping_rounds=self.early_stopping, verbose_eval=50)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))i


def main():

    train = pd.read_pickle('../input/processed_train_df.pkl')
    y = train['interest_level'].values

    train_df = np.load('../input/stacked_train1.npy')
    test_df = np.load('../input/stacked_test1.npy')

    rf_params = {
        'n_jobs': -1,
        'n_estimators':1200,
        'max_features': 'sqrt',
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'criterion':'entropy',
        'kbest':193
     }

    xgb_params = {
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

    lr_params = {
        'C':1.0,
        'penalty':'l2'
    }

    def build_keras():
        model = Sequential()
        model.add(Dense(3, activation='relu', input_dim=train_df[1]))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        return model

    keras_nn = KerasWrapper(KerasClassifier(build_fn=build_keras, epochs=200, 
        batch_size=5, verbose=10))

    lr1 = SklearnWrpper(clf=LogisticRegression, seed=SEED, params=lr_params)

    xg2 = XgbWrapper(seed=SEED, params=xgb_params2)
    xg3 = XgbWrapper(seed=SEED, params=xgb_params3)
    xg4 = XgbWrapper(seed=SEED, params=xgb_params4)

    rf1 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params1)
    rf2 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params2)
    rf3 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params3)
    rf4 = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params4)

    et1 = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params1)

    clfs = [rf1, xg2, xg3, xg4, xg1, rf2, rf3, rf4, et1]

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

    np.save('../input/stacked_train2.npy', dataset_blend_train)
    np.save('../input/stacked_test2.npy', dataset_blend_test)

    print('Predicting Blend')
    clf = LogisticRegression(C=1.0)
    clf.fit(dataset_blend_train)
    clf.predict(dataset_blend_test)

    out_df = pd.DataFrame(preds)
    out_df.columns = ["low", "medium", "high"]
    out_df["listing_id"] = test_df.listing_id.values
    out_df.to_csv("../output/final_sub.csv", index=False)

if __name__ == '__main__':
    main()
