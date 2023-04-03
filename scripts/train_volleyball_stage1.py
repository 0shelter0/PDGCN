import sys
sys.path.append(".")
from train_stage1 import *

cfg=Config('volleyball')

cfg.use_multi_gpu = True
cfg.device_list="0,2,3"
cfg.training_stage=1
cfg.stage1_model_path=''
cfg.train_backbone=True
cfg.test_before_train = False

# inv3 setup
# cfg.backbone = 'inv3'
# cfg.image_size = 720, 1280
# cfg.out_size = 87, 157
# cfg.emb_features = 1056#512

# vgg16 setup
# cfg.backbone = 'vgg16'
# cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
# cfg.out_size = 22, 40
# cfg.emb_features = 512

# res18 setup
cfg.backbone = 'res18'
cfg.stage1_model_path = 'result/basemodel_VD_res18.pth'
cfg.out_size = 23, 40
cfg.emb_features = 512

cfg.num_before = 5
cfg.num_after = 4

cfg.batch_size=24
cfg.test_batch_size=12
cfg.num_frames=1
# cfg.train_learning_rate=1e-5
# cfg.lr_plan={}
# cfg.max_epoch=200
cfg.train_learning_rate=1e-4
cfg.lr_plan={30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch=120
cfg.set_bn_eval = False
cfg.actions_weights=[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]  

cfg.exp_note='Volleyball_res18'
train_net(cfg)
