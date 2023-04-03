import sys
sys.path.append(".")
from train_stage2 import *
import torch.multiprocessing as mp


if __name__ == "__main__":
  cfg=Config('collective')
  cfg.inference_module_name = 'pdgcn_collective'

  # ctx setup
  cfg.device_list="0,2,3"
  cfg.training_stage=2
  cfg.test_interval_epoch = 1
  cfg.use_gpu = True
  cfg.use_multi_gpu = True
  cfg.train_backbone = True
  cfg.load_backbone_stage2 = True
  cfg.test_before_train = False

  # vgg16 setup
  # cfg.image_size = 480, 720
  # cfg.backbone = 'vgg16'
  # cfg.stage1_model_path = 'result/basemodel_VD_vgg16.pth'
  # cfg.out_size = 15, 22
  # cfg.emb_features = 512

  # res18 setup
  # cfg.image_size = 480, 720
  # cfg.backbone = 'res18'
  # cfg.stage1_model_path = 'result/Collective_stage1_res18_stage1_rank0_2023-03-26_04-51-03/stage1_epoch16_86.48%.pth'
  # # 'result/Collective_stage1_res18_stage1_rank0_2023-03-23_10-44-03/stage1_epoch86_83.83%.pth'
  # cfg.out_size = 15, 23
  # cfg.emb_features = 512

  # inv3 setup
  cfg.image_size = 480, 720
  cfg.backbone = 'inv3'
  cfg.stage1_model_path = 'result/Collective_stage1_inv3_stage1/stage1_epoch78_92.71%.pth'
  cfg.out_size = 57,87
  cfg.emb_features=1056

  # training setup
  cfg.num_boxes = 13
  cfg.num_actions = 5
  cfg.num_activities = 4
  cfg.num_frames = 4
  cfg.batch_size = 2 # 2
  cfg.test_batch_size = 2

  cfg.train_learning_rate = 5e-5 # 5e-5 1e-4
  cfg.train_dropout_prob = 0.5 # 0.3
  cfg.weight_decay = 1e-4
  cfg.lr_plan = { }
  cfg.max_epoch = 30 # 30

  # cfg.activities_weights = [0.5, 3.0, 2.0, 1.0]
  # cfg.actions_weights = [0.5, 1.0, 4.0, 3.0, 2.0]
  cfg.actions_loss_weight = 1.0

  cfg.exp_note='Pose_collective_inv3_without_PE'


  world_size = torch.cuda.device_count()
  # rank is auto-allocated by DDP when calling mp.spawn
  mp.spawn(train_net, args=(world_size, cfg), nprocs=world_size)
