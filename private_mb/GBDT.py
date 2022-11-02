import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
 
def LGBM(args, params):
    # to compare predict result with other models by user_id, isbn
    # rating_train = train[['user_id','isbn','rating']] 
    # rating_test = test[['user_id','isbn','rating']] 

    # LGBM - Classifier
    # params = {'objective':'rmse',
    #           'boosting_type':args.LGBM_ALG,
    #           'lambda': args.LGBM_LAMBDA,
    #           'learning_rate': args.LR,
    #           'max_depth': args.LGBM_MAX_DEPTH,
    #           'num_leaves': args.LGBM_NUM_LEAVES,
    #         }

    if args.LGBM_TYPE == 'C':
        lgbm = LGBMClassifier(**params, n_estimators=1500, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        lgbm = LGBMRegressor(**params, n_estimators=12000, random_state=args.SEED)

    # evaluation = [(X_train, y_train),(X_valid, y_valid)]
    lgbm.fit(data['X_train'], data['y_train'], eval_metric='rmse', early_stopping_rounds=100, verbose=False)

    return lgbm


def CATB(args, params):
    # params = {'iterations': args.CATB_ITER,
    #         'eval_metric': 'rmse'
    #         'learning_rate':hp.choice('learning_rate', np.arange(*args.LR_RANGE)),
    #         'depth': hp.choice('depth', np.arange(*args.CATB_DEPTH, dtype=int))
    #         'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(*args.CATB_COLS))
    #         }
    
    if args.CATB_TYPE == 'C':
        catb = CatBoostClassifier(**params, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        catb = CatBoostRegressor(**params, random_state=args.SEED)

    # evaluation = [(X_train, y_train),(X_valid, y_valid)]
    catb.fit(data['X_train'], data['y_train'], early_stopping_rounds=100, verbose=False)
    return catb

def XGB(args, params):    
    # to compare predict result with other models by user_id, isbn
    # rating_train = train[['user_id','isbn','rating']] 
    # rating_test = test[['user_id','isbn','rating']] 

    # XGB - Classifier
    # params = {'objective':'reg:linear',
    #         'eval_metric':'rmse',
    #         'booster':args.XGB_BOOSTER,
    #         'n_estimators':args.XGB_N_ESTI,
    #         'reg_lambda': args.XGB_LAMBDA,
    #         'learning_rate': args.LR,
    #         } #args.XGB_BOOSTER == 'gblinear'
    
    # if args.XGB_BOOSTER == 'gbtree':
    #     params['max_depth'] = args.XGB_MAX_DEPTH
    #     params['min_child_weight'] = args.XGB_MIN_CHILD

    if args.XGB_TYPE == 'C':
        params['objective']='mult:softmax'
        params['eval_metric']='merror'
        xgb = XGBClassifier(**params, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        xgb = XGBRegressor(**params, random_state=args.SEED)

    xgb.fit(X_train, y_train, early_stopping_rounds=100, verbose=False)

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

def all_params(args):
    # XGB parameters
    xgb_reg_params = {
        'objective':'reg:linear',
        'eval_metric':'rmse',
        'n_estimators':args.N_EST,
        'booster':args.XGB_BOOSTER,
        'min_child_weight': hp.choice('min_child_weight', np.arange(*args.LGBM_MIN_CHILD_W dtype=int)),
        'reg_lambda': args.LAMBDA,
        'learning_rate': args.LR,
        } #args.XGB_BOOSTER == 'gblinear

    if args.XGB_BOOSTER == 'gbtree':
        xgb_reg_params['max_depth'] = args.XGB_MAX_DEPTH
        xgb_reg_params['min_child_weight'] = args.XGB_MIN_CHILD

    xgb_fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
        'verbose': False
    }
    xgb_para = dict()
    xgb_para['reg_params'] = xgb_reg_params
    xgb_para['fit_params'] = xgb_fit_params


    # LightGBM parameters
    lgb_reg_params = {
        'learning_rate':    hp.choice('learning_rate', np.arange(*args.LR_RANGE)),
        'max_depth':        hp.choice('max_depth',        np.arange(*args.MAX_DEPTH, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(*args.LGBM_MIN_CHILD_W dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'lambda':           hp.uniform('lambda', 1.1,1.5)
        'n_estimators':     args.N_EST,
    }
    lgb_fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params

    return 

    # CatBoost parameters
    ctb_reg_params = {
        'n_estimators': args.N_EST
        'eval_metric': 'rmse'
        'learning_rate':     hp.choice('learning_rate', np.arange(*args.LR_RANGE)),
        'depth':             hp.choice('depth', np.arange(*args.CATB_DEPTH, dtype=int))
        'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(*args.CATB_COLS))
        }
    ctb_fit_params = {
        'early_stopping_rounds': 100,
        'verbose': False
    }
    ctb_para = dict()
    ctb_para['reg_params'] = ctb_reg_params
    ctb_para['fit_params'] = ctb_fit_params

    return xgb_para, lgb_para, ctb_para


class HPOpt:

    def __init__(self, args, data):
        self.x_train = data[['x_train']]
        self.x_test  = data['x_test']
        self.y_train = data['y_train']
        self.y_test  = data['y_test']

        self.xgb_para, self.lgb_para, self.ctb_para = all_params(args)
    
    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        if space == 'xgb':
            space_dict = self.xgb_para 
        elif space == 'lgb':
            space_dict = self.lgb_para 
        else:
            space_dict = self.ctb_para 
            
        try:
            result = fmin(fn=fn, space=space_dict, algo=algo, max_eval=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result

    def xgb_reg(self, para):
        reg = XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)
    
    def lgb_reg(self, para):
        reg = LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)
    
    def ctb_reg(self, para):
        reg = CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)
    
    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])

    pred = reg.predict(self.x_test)
    loss = rmse(self.y_test, pred)
    return {'loss': loss, 'status':STATUS_OK}