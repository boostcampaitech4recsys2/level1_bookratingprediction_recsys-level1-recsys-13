import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe
 
def XGB(args, data, params):    
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

    X_train, y_train, X_valid, y_valid = data['X_train'], data['y_train'], data['X_valid'], data['y_valid']

    if args.TYPE == 'C':
        params['objective']='mult:softmax'
        params['eval_metric']='merror'
        xgb = XGBClassifier(**params, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        xgb = XGBRegressor(**params, random_state=args.SEED)
    
    evaluation = [(X_train, y_train),(X_valid, y_valid)]
    xgb.fit(X_train, y_train, eval_set=evaluation, early_stopping_rounds=100, verbose=1000)

    return xgb

def LGBM(args, data, params):
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

    X_train, y_train, X_valid, y_valid = data['X_train'], data['y_train'], data['X_valid'], data['y_valid']

    if args.TYPE == 'C':
        lgbm = LGBMClassifier(**params, n_estimators=1500, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        lgbm = LGBMRegressor(**params, n_estimators=12000, random_state=args.SEED)

    evaluation = [(X_train, y_train),(X_valid, y_valid)]
    lgbm.fit(X_train, y_train, eval_set=evaluation, eval_metric='rmse', early_stopping_rounds=100, verbose=1000)

    return lgbm


def CATB(args, data, params):
    # params = {'iterations': args.CATB_ITER,
    #         'eval_metric': 'rmse'
    #         'learning_rate':hp.choice('learning_rate', np.arange(*args.LR_RANGE)),
    #         'depth': hp.choice('depth', np.arange(*args.CATB_DEPTH, dtype=int))
    #         'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(*args.CATB_COLS))
    #         }
    X_train, y_train, X_valid, y_valid = data['X_train'], data['y_train'], data['X_valid'], data['y_valid']

    if args.TYPE == 'C':
        catb = CatBoostClassifier(**params, random_state=args.SEED)
        # rmse(y_test,catboost_pred_cl.squeeze(1))
    else:
        catb = CatBoostRegressor(**params, random_state=args.SEED)

    evaluation = [(X_train, y_train),(X_valid, y_valid)]
    catb.fit(X_train, y_train, eval_set = evaluation, early_stopping_rounds=100, verbose=1000)
    return catb


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
        'objective':'reg:squarederror',
        'n_estimators':args.N_EST,
        'booster':args.XGB_BOOSTER,
        'reg_lambda':           hp.uniform('reg_lambda', *args.XGB_LAMBDA),
        'learning_rate':        hp.loguniform('learning_rate', np.log(0.05), np.log(0.3))
        } #args.XGB_BOOSTER == 'gblinear

    if args.XGB_BOOSTER == 'gbtree':
        xgb_reg_params['max_depth'] = hp.choice('max_depth', np.arange(*args.MAX_DEPTH))
        xgb_reg_params['min_child_weight'] = hp.choice('min_child_weight', np.arange(*args.MIN_CHILD_W))

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
        'n_estimators':     args.N_EST,
        'learning_rate':    hp.loguniform('learning_rate', np.log(0.05), np.log(0.3)),
        'max_depth':        hp.choice('max_depth',        np.arange(*args.MAX_DEPTH)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(*args.MIN_CHILD_W)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.5),
        'subsample':        hp.uniform('subsample', 0.8, 1),
        'reg_lambda':       hp.uniform('reg_lambda', *args.LGBM_LAMBDA)
    }
    lgb_fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params

    # CatBoost parameters
    ctb_reg_params = {
        'n_estimators': args.N_EST,
        'eval_metric': 'RMSE',
        'learning_rate':     hp.loguniform('learning_rate', np.log(0.05), np.log(0.3)),
        'max_depth':         hp.choice('max_depth', np.arange(*args.MAX_DEPTH)),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 0.5)
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
        self.x_train = data['X_train']
        self.x_test  = data['X_valid']
        self.y_train = data['y_train']
        self.y_test  = data['y_valid']

        self.xgb_para, self.lgb_para, self.ctb_para = all_params(args)
    
    def process(self, fn_name, space):
        fn = getattr(self, fn_name)
        if space == 'xgb':
            space_dict = self.xgb_para 
        elif space == 'lgb':
            space_dict = self.lgb_para 
        else:
            space_dict = self.ctb_para 
            
        try:
            result = fmin(fn=fn, space=space_dict, algo=tpe.suggest, max_evals=100, trials=Trials())
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