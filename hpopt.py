import time
import argparse
import pandas as pd

from src import seed_everything

from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import NeuralCollaborativeFiltering, WideAndDeepModel, DeepCrossNetworkModel
from src import CNN_FM
from src import DeepCoNN

import wandb

from private_mb import HPOpt, XGB, LGBM, CATB, rmse, feat_comb

def main(args):
    seed_everything(args.SEED)

    ######################## SET WANDB
    if args.WANDB:
        wandb.init(project="test-project", entity="ai-tech-4-recsys13")
        wandb.run.name = 'data_mb_' + args.MODEL + '_EPOCH:' + str(args.EPOCHS) + '_EMBDIM:' + str(args.FFM_EMBED_DIM)
        wandb.config = {
            "learning_rate": args.LR ,
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "architecture": args.MODEL,
        }

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_load(args)
    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.MODEL == 'CNN_FM':
        data = image_data_load(args)
    elif args.MODEL == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    elif args.MODEL in ('LGBM','CATB','XGB'):
        data = context_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.MODEL=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.MODEL=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)

    elif args.MODEL in ('XGB', 'LGBM', 'CATB'):
        data = context_data_split(args, data)

        if args.FEAT_COMB: # Feature Combine Ensemble
            data['X_train'], data['X_valid'] = feat_comb(args.ENSEMBLE_FILES, data)
            
        hpopt = HPOpt(args, data)

    else:
        pass

    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='FM':
        model = FactorizationMachineModel(args, data)
    elif args.MODEL=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data)
    elif args.MODEL=='NCF':
        model = NeuralCollaborativeFiltering(args, data)
    elif args.MODEL=='WDN':
        model = WideAndDeepModel(args, data)
    elif args.MODEL=='DCN':
        model = DeepCrossNetworkModel(args, data)
    elif args.MODEL=='CNN_FM':
        model = CNN_FM(args, data)
    elif args.MODEL=='DeepCoNN':
        model = DeepCoNN(args, data)
    elif args.MODEL=='XGB':
        if args.TYPE == 'C':
            xgb_opt_param = hpopt.process(fn_name='xgb_cls', space='xgb')
        else:
            xgb_opt_param = hpopt.process(fn_name='xgb_reg', space='xgb')
        model = XGB(args, data, xgb_opt_param)
        print(xgb_opt_param)
    elif args.MODEL=='LGBM':
        if args.TYPE == 'C':
            lgb_opt_param = hpopt.process(fn_name='lgb_cls', space='lgb')
        else:
            lgb_opt_param = hpopt.process(fn_name='lgb_reg', space='lgb')
        model = LGBM(args, data, lgb_opt_param)
        print(lgb_opt_param)
    elif args.MODEL=='CATB':
        if args.TYPE == 'C':
            ctb_opt_param = hpopt.process(fn_name='ctb_cls', space='ctb')
        else:
            ctb_opt_param = hpopt.process(fn_name='ctb_reg', space='ctb')
        model = CATB(args, data, ctb_opt_param)
        print(ctb_opt_param)
    else:
        pass

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    if args.MODEL in ('LGBM', 'CATB', 'XGB'):
        pass
    else:
        model.train()
    
    if args.WANDB:
        wandb.finish()

    ######################## INFERENCE
    
    #if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN'):
    #    predicts = model.predict(data['test_dataloader'])
    #elif args.MODEL=='CNN_FM':
    #    predicts  = model.predict(data['test_dataloader'])
    #elif args.MODEL=='DeepCoNN':
    #    predicts  = model.predict(data['test_dataloader'])
    if args.MODEL in ('LGBM', 'CATB', 'XGB'):
        print(f'--------------- {args.MODEL} PREDICT ---------------')
        predicts  = model.predict(data['test'])
        # print('RMSE(LGBM):', rmse(data['test'], predicts))
    else:
        pass

    ######################## SAVE PREDICT
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('LGBM', 'CATB', 'XGB'):
        print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
        submission['rating'] = predicts
    else:
        pass
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/FFM_data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN', 'LGBM', 'CATB', 'XGB'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--WANDB', type=bool, default=False, help='wandb 기록 여부를 선택할 수 있습니다.')
    

    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=16, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### FFM
    arg('--FFM_EMBED_DIM', type=int, default=16, help='FFM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### NCF
    arg('--NCF_EMBED_DIM', type=int, default=16, help='NCF에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--NCF_MLP_DIMS', type=list, default=(16, 16), help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### WDN
    arg('--WDN_EMBED_DIM', type=int, default=16, help='WDN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--WDN_MLP_DIMS', type=list, default=(16, 16), help='WDN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--WDN_DROPOUT', type=float, default=0.2, help='WDN에서 Dropout rate를 조정할 수 있습니다.')

    ############### DCN
    arg('--DCN_EMBED_DIM', type=int, default=16, help='DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DCN_MLP_DIMS', type=list, default=(16, 16), help='DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--DCN_DROPOUT', type=float, default=0.2, help='DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--DCN_NUM_LAYERS', type=int, default=3, help='DCN에서 Cross Network의 레이어 수를 조정할 수 있습니다.')

    ############### CNN_FM
    arg('--CNN_FM_EMBED_DIM', type=int, default=128, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--CNN_FM_LATENT_DIM', type=int, default=8, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')

    ############### DeepCoNN
    arg('--DEEPCONN_VECTOR_CREATE', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--DEEPCONN_EMBED_DIM', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_LATENT_DIM', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_CONV_1D_OUT_DIM', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_KERNEL_SIZE', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_WORD_DIM', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_OUT_DIM', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

    ############### GBDT (공통)
    arg('--TYPE', type=str, default='R', help='Classifier(C)와 Regressor(R) 중 고를 수 있습니다.')
    arg('--N_EST', type=int, default=200, help='학습에 활용될 weak learner의 반복 수를 조정할 수 있습니다.')

    arg('--LR_RANGE', nargs='+',default='0.05,0.3',
        type=lambda s: [float(item) for item in s.split(',')],
        help='Learning Rate를 np.arange 형식에 맞춰 넣을 수 있습니다.')
    arg('--MAX_DEPTH', nargs='+',default='5,16,1',
        type=lambda s: [int(item) for item in s.split(',')],
        help='트리의 최대 길이를 np.arange 형식에 맞춰 넣을 수 있습니다.')
    arg('--COLS', nargs='+',default='0.3,0.8',
        type=lambda s: [float(item) for item in s.split(',')],
        help='colsample_bylevel 를 np.arange 형식에 맞춰 넣을 수 있습니다.')    
    arg('--MIN_CHILD_W', nargs='+',default='1,8,1',
        type=lambda s: [int(item) for item in s.split(',')],
        help='min_child_weight를 np.arange 형식에 맞춰 넣을 수 있습니다.')
    
    ############### XGB
    arg('--XGB_BOOSTER', type=str, default='gbtree', help='XGB에서 실행시킬 알고리즘(gbtree, gblinear)을 정의할 수 있습니다.')
    arg('--XGB_LAMBDA', nargs='+',default='0,1',
        type=lambda s: [float(item) for item in s.split(',')],
        help='regularization 정규화 값을 조정할 범위로 조정할 수 있습니다.')

    ############### LGBM
    arg('--LGBM_ALG', type=str, default='gbdt', help='LGBM에서 실행시킬 알고리즘을 정의할 수 있습니다.')
    arg('--LGBM_NUM_LEAVES', type=int, default=500, help='LGBM에서 전체 Tree의 leaves 수를 조정할 수 있습니다.')
    arg('--LGBM_LAMBDA', nargs='+',default='1.1,1.5',
        type=lambda s: [float(item) for item in s.split(',')],
        help='regularization 정규화 값을 조정할 범위로 조정할 수 있습니다.')

    ############### FEATURE COMBINE
    arg('--FEAT_COMB', type=bool, default=False, help='FEATURE COMBINE 여부를 선택할 수 있습니다.')

    arg("--ENSEMBLE_FILES", nargs='+',required=False,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')

    args = parser.parse_args()
    main(args)
