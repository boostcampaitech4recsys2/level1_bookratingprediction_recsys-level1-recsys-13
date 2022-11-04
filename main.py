import time
import argparse
import pandas as pd

from src import seed_everything

from src.data import context_data_load, context_data_split, context_data_loader
from src.data import fm_data_load, fm_data_split, fm_data_loader
from src.data import ffm_data_load, ffm_data_split, ffm_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import ncf_data_load, ncf_data_split, ncf_data_loader
from src.data import dcn_data_load, dcn_data_split, dcn_data_loader
from src.data import wdn_data_load, wdn_data_split, wdn_data_loader

from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader

from src import FactorizationMachineModel, FieldAwareFactorizationMachineModel
from src import NeuralCollaborativeFiltering, WideAndDeepModel, DeepCrossNetworkModel
from src import CNN_FM
from src import DeepCoNN

import wandb

from gb import XGB, LGBM, CATB, rmse

def predicts_map(x: float) -> float:
    if x < 1:
        return 1.0
    elif x > 10:
        return 10.0
    else:
        return x
        
def main(args):
    seed_everything(args.SEED)

    ######################## SET WANDB
    if args.WANDB:
        wandb.init(project="test-project", entity="ai-tech-4-recsys13")
        wandb.run.name =  args.MODEL #+ '_EPOCH:' + str(args.EPOCHS) + '_EMBDIM:' + str(args.FFM_EMBED_DIM)
        wandb.config = {
            "learning_rate": args.LR ,
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "architecture": args.MODEL,
        }

    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM'):
        data = fm_data_load(args)
    elif args.MODEL in ('FFM'):
        data = ffm_data_load(args)
    elif args.MODEL in ('WDN'):
        data = wdn_data_load(args)
    elif args.MODEL in ('DCN'):
        data = dcn_data_load(args)
    elif args.MODEL in ('NCF'):
        data = ncf_data_load(args)
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

    args.NCF_MLP_DIMS = [int(i) for i in args.NCF_MLP_DIMS.split()]
    args.WDN_MLP_DIMS = [int(i) for i in args.WDN_MLP_DIMS.split()]
    args.DCN_MLP_DIMS = [int(i) for i in args.DCN_MLP_DIMS.split()]

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM'):
        data = fm_data_split(args, data)
        data = fm_data_loader(args, data)

    elif args.MODEL in ('FFM'):
        data = ffm_data_split(args, data)
        data = ffm_data_loader(args, data)
    
    elif args.MODEL in ('WDN'):
        data = wdn_data_split(args, data)
        data = wdn_data_loader(args, data)
    
    elif args.MODEL in ('DCN'):
        data = dcn_data_split(args, data)
        data = dcn_data_loader(args, data)

    elif args.MODEL in ('NCF'):
        data = ncf_data_split(args, data)
        data = ncf_data_loader(args, data)

    elif args.MODEL=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.MODEL=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)

    elif args.MODEL in ('LGBM', 'CATB', 'XGB'):
        data = context_data_split(args, data)
    
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
    elif args.MODEL=='LGBM':
        model = LGBM(args, data)
    elif args.MODEL=='CATB':
        model = CATB(args, data)
    elif args.MODEL=='XGB':
        model = XGB(args, data)
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
        submission['rating'] = submission['rating'].apply(predicts_map)
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)
    else:
        pass
    # now = time.localtime()
    # now_date = time.strftime('%Y%m%d', now)
    # now_hour = time.strftime('%X', now)
    # save_time = now_date + '_' + now_hour.replace(':', '')
    # submission.to_csv('submit/{}_{}.csv'.format(save_time, args.MODEL), index=False)



if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/raw_data/', help='Data path를 설정할 수 있습니다.')
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
    arg('--NCF_MLP_DIMS', type=str, default="16 16", help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### WDN
    arg('--WDN_EMBED_DIM', type=int, default=16, help='WDN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--WDN_MLP_DIMS', type=str, default="16 16", help='WDN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--WDN_DROPOUT', type=float, default=0.2, help='WDN에서 Dropout rate를 조정할 수 있습니다.')

    ############### DCN
    arg('--DCN_EMBED_DIM', type=int, default=16, help='DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DCN_MLP_DIMS', type=str, default="16 16", help='DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
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

    ############### LGBM
    arg('--LGBM_TYPE', type=str, default='R', help='LGBM Classifier(C)와 Regressor(R) 중 고를 수 있습니다.')
    arg('--LGBM_ALG', type=str, default='gbdt', help='LGBM에서 실행시킬 알고리즘을 정의할 수 있습니다.')
    arg('--LGBM_LAMBDA', type=int, default=0.1, help='LGBM에서 regularization 정규화 값을 조정할 수 있습니다.')
    arg('--LGBM_MAX_DEPTH', type=int, default=10, help='LGBM에서 트리의 최대 깊이를 조정할 수 있습니다.')
    arg('--LGBM_NUM_LEAVES', type=int, default=500, help='LGBM에서 전체 Tree의 leaves 수를 조정할 수 있습니다.')

    ############### CATB
    arg('--CATB_TYPE', type=str, default='R', help='CATB Classifier(C)와 Regressor(R) 중 고를 수 있습니다.')
    arg('--CATB_ITER', type=int, default=10000, help='same as n_estiamtors')
    arg('--CATB_DEPTH', type=int, default=10, help='Depth of the tree')

    ############### XGB
    arg('--XGB_TYPE', type=str, default='R', help='LGBM Classifier(C)와 Regressor(R) 중 고를 수 있습니다.')
    arg('--XGB_BOOSTER', type=str, default='gbtree', help='XGB에서 실행시킬 알고리즘(gbtree, gblinear)을 정의할 수 있습니다.')
    arg('--XGB_N_ESTI', type=int, default='10', help='XGB에서 학습에 활용될 weak leaner의 반복 수를 조정할 수 있습니다.')
    arg('--XGB_LAMBDA', type=int, default=1, help='XGB에서 L2 regularization 정규화 값을 조정할 수 있습니다.')
    arg('--XGB_MIN_CHILD', type=int, default=1, help='XGB에서 leaf node에 포함되는 최소 관측치의 수를 조정할 수 있습니다.[0,inf]')
    arg('--XGB_MAX_DEPTH', type=int, default=6, help='XGB에서 트리의 최대 깊이를 조정할 수 있습니다. [0,inf]')


    ############### FEATURE COMBINE
    arg('--FEAT_COMB', type=bool, default=False, help='FEATURE COMBINE 여부를 선택할 수 있습니다.')
    arg("--ENSEMBLE_FILES", nargs='+',required=False,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. 이 때, .csv와 같은 확장자는 입력하지 않습니다.')


    args = parser.parse_args()
    main(args)
