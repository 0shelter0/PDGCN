import sys
sys.path.append(".")
from train_net_dynamic import *

cfg=Config('volleyball')
cfg.inference_module_name = 'arg_volleyball'

cfg.device_list = "0,1,2,3"
cfg.use_gpu = True
cfg.use_multi_gpu = True
cfg.training_stage = 2
cfg.train_backbone = False
cfg.test_before_train = False
cfg.test_interval_epoch = 1


# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18 setup
cfg.backbone = 'res18'
cfg.stage1_model_path = 'result/Volleyball_res18_stage1_2022-05-25_02-06-20/stage1_epoch30_90.28%.pth'
cfg.out_size = 23, 40
cfg.emb_features = 512

# inv3 setup
# cfg.backbone = 'inv3'
# cfg.stage1_model_path = 'result/stage1_VD_inv3_epoch98_91.92%.pth'
# cfg.out_size = 87, 157
# cfg.emb_features = 1056#512


cfg.batch_size = 32
cfg.test_batch_size = 16
cfg.num_frames = 3
cfg.load_backbone_stage2 = True
cfg.train_learning_rate = 2e-4 #2e-4
# cfg.lr_plan = {11: 3e-5, 21: 1e-5} #original settings
cfg.lr_plan = {11: 1e-4, 21: 3e-5}
cfg.max_epoch = 30
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]

cfg.exp_note = 'ARG_Volleyball_inv3'
train_net(cfg)