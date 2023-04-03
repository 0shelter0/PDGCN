import sys
sys.path.append(".")
from train_stage2 import *
import torch.multiprocessing as mp

if __name__ == "__main__":
    cfg=Config('volleyball')
    cfg.inference_module_name = 'pdgcn_volleyball'

    cfg.device_list = "0,2,3"
    cfg.use_gpu = True
    cfg.use_multi_gpu = True
    cfg.training_stage = 2
    cfg.train_backbone = True
    cfg.test_before_train = False
    cfg.test_interval_epoch = 1

    # vgg16 setup
    # cfg.backbone = 'vgg16'
    # cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
    # cfg.out_size = 22, 40
    # cfg.emb_features = 512

    # res18 setup
    # cfg.backbone = 'res18'
    # cfg.stage1_model_path = 'result/Volleyball_res18_stage1_2022-05-25/stage1_epoch30_90.28%.pth'
    # cfg.out_size = 23, 40
    # cfg.emb_features = 512

    # inv3 setup
    cfg.backbone = 'inv3'
    cfg.stage1_model_path = 'result/stage1_VD_inv3_epoch98_91.92%.pth'
    cfg.out_size = 87, 157
    cfg.emb_features=1056


    cfg.train_dropout_prob = 0.3
    cfg.batch_size = 2
    cfg.test_batch_size = 2
    cfg.num_frames = 3
    cfg.load_backbone_stage2 = True
    cfg.train_learning_rate = 1e-4


    cfg.max_epoch = 30
    cfg.lr_plan = {11: 1e-5}
    cfg.actions_weights = [1., 1., 2., 3., 1., 2., 2., 0.2, 1.]
    actions_loss_weight = 1.0

    cfg.exp_note = '2branch_Volleyball_inv3-2heads'
    
    world_size = torch.cuda.device_count()
    # rank is auto-allocated by DDP when calling mp.spawn
    mp.spawn(train_net, args=(world_size, cfg), nprocs=world_size)
