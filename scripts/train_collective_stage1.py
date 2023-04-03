import sys
sys.path.append(".")
from train_stage1 import *

cfg=Config('collective')

cfg.device_list="2,3"
cfg.training_stage=1
cfg.train_backbone=True


cfg.image_size=480, 720
cfg.num_boxes=13
cfg.num_actions=5
cfg.num_activities=4
cfg.num_frames=10

# inv3 setup
# cfg.backbone='inv3'
# cfg.out_size=57,87
# cfg.emb_features = 1056

# res18 setup
# cfg.backbone = 'res18'
# cfg.out_size = 15, 23
# cfg.emb_features = 512

# vgg16 setup
cfg.backbone = 'vgg16'
cfg.out_size = 15, 22
cfg.emb_features = 512

cfg.batch_size=8
cfg.test_batch_size=8 
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={ }
cfg.max_epoch=120

cfg.inference_module_name = 'collective'
cfg.exp_note='Collective_stage1_vgg16'
train_net(cfg)