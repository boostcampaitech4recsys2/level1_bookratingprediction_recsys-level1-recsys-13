import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from private_mb import data_exp_load, exp_data_split, exp_data_loader

def LGBM(args, data):
    X_train, X_valid, y_train, y_valid = data['X_train'], data['X_valid'], data['y_train'], data['y_valid']
    
    # to compare predict result with other models by user_id, isbn
    # rating_train = train[['user_id','isbn','rating']] 
    # rating_test = test[['user_id','isbn','rating']] 

    # LGBM - Classifier
    params = {'objective':'rmse',
              'boosting_type':args.LGBM_ALG,
              'lambda': args.LGBM_LAMBDA,
              'learning_rate': args.LR,
              'max_depth': args.LGBM_MAX_DEPTH,
              'num_leaves': args.LGBM_NUM_LEAVES,
            }

    if args.LGBM_TYPE == 'C':
        lgbm = LGBMClassifier(**params, n_estimators=1500, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        lgbm = LGBMRegressor(**params, n_estimators=12000, random_state=args.SEED)

    evaluation = [(X_train, y_train),(X_valid, y_valid)]
    lgbm.fit(X_train, y_train, eval_set = evaluation, eval_metric='rmse', early_stopping_rounds=100, verbose=False)

    return lgbm
    
def CATB(args, data):
    X_train, X_valid, y_train, y_valid = data['X_train'], data['X_valid'], data['y_train'], data['y_valid']
    
    params = {'iterations': args.CATB_ITER,
            'learning_rate':args.LR,
            'depth': args.CATB_DEPTH
            }
    
    if args.LGBM_TYPE == 'C':
        catb = CatBoostClassifier(**params, eval_metric='rmse', random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        catb = CatBoostRegressor(**params, random_state=args.SEED)

    evaluation = [(X_train, y_train),(X_valid, y_valid)]
    catb.fit(X_train, y_train, eval_set = evaluation, verbose=1000, early_stopping_rounds=500)

    return catb

def XGB(args, data):
    X_train, X_valid, y_train, y_valid = data['X_train'], data['X_valid'], data['y_train'], data['y_valid']
    
    # to compare predict result with other models by user_id, isbn
    # rating_train = train[['user_id','isbn','rating']] 
    # rating_test = test[['user_id','isbn','rating']] 

    # XGB - Classifier
    if args.XGB_BOOSTER == 'gbtree':
        params = {'objective':'reg:linear',
                'eval_metric':'rmse',
                'booster':args.XGB_BOOSTER,
                'n_estimators':args.XGB_N_ESTI,
                'reg_lambda': args.XGB_LAMBDA,
                'learning_rate': args.LR,
                'max_depth': args.XGB_MAX_DEPTH,
                'min_child_weight': args.XGB_MIN_CHILD,
                'base_score' : 7
                }
    else: #args.XGB_BOOSTER == 'gblinear'
        parmas = {'objective':'reg:linear',
                'eval_metric':'rmse',
                'booster':args.XGB_BOOSTER,
                'n_estimators':args.XGB_N_ESTI,
                'reg_lambda': args.XGB_LAMBDA,
                'learning_rate': args.LR,
                'base_score' : 7
                }

    if args.XGB_TYPE == 'C':
        params['objective']='mult:softmax'
        params['eval_metric']='merror'
        xgb = XGBClassifier(**params, early_stopping_rounds=100, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        xgb = XGBRegressor(**params, early_stopping_rounds=100, random_state=args.SEED)

    xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose=True)

    return xgb




def modify_range(rating):
    if rating < 0:
        return 0
    elif rating > 10:
        return 10
    else:
        return rating
        
def rmse(real, predict):
    pred = list(map(modify_range, predict))  
    pred = np.array(pred)
    return np.sqrt(np.mean((real-pred) ** 2))


def feat_comb(filenames, data):
    X_train, X_valid = data['X_train'], data['X_valid']
    isbn2idx = data['isbn2idx']
    user2idx = data['user2idx']

    file_list = sum(filenames, [])
    filepath = 'submit/'
    output_path = [filepath+f+'.csv' for f in file_list]

    for idx, path in enumerate(output_path):
        output = pd.read_csv(path)
        output['isbn'] = output['isbn'].map(isbn2idx)
        output['user_id'] = output['user_id'].map(user2idx)
        #output['rating'] = output['rating'].map(round) # round output ratings - maybe only when classifier?
        output.rename(columns={'rating':f'output_{idx}'}, inplace=True)

        X_train.merge(output, on=['user_id', 'isbn'], how='left')
        X_valid.merge(output, on=['user_id', 'isbn'], how='left')

    return X_train, X_valid