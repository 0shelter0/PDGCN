from backbone.backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from infer_module.ARG_infer_module import GCN_Module
from infer_module.sce_module import MultiHeadLayerEmbfeatureContextEncoding
from infer_module.positional_encoding import Context_PositionEmbeddingSine, Embfeature_PositionEmbedding
from infer_module.cdgcn_module import MultiLayerCDGCN
import collections


class PDGCN_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(PDGCN_volleyball, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_heads_context_pose = 2
        num_features_context = 128

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        pose_dim = 256

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.fc_emb_2 = nn.Linear(NFB, 512)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        self.nl_emb_2 = nn.LayerNorm([512])
        self.pose_fc = nn.Sequential(nn.Linear(34,1024),nn.Linear(1024,pose_dim))
        

        # TCE Module Loading
        self.multilayer_head_embfeature_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context, 1,
                num_features_context, NFB, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16, 768 / 2) # 768 / 2 for inv3
        self.boxes_pe = Embfeature_PositionEmbedding(cfg)
        self.poses_pe = Embfeature_PositionEmbedding(cfg, num_pos_feats=pose_dim/2)

        self.multilayer_head_pose_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context_pose, 1,
                num_features_context, pose_dim, N, context_dropout_ratio=0.1)


        boxes_context_dim = NFB + num_heads_context * num_features_context # 1024 + 4*128
        pose_context_dim = pose_dim + num_heads_context_pose * num_features_context
        gcn_dim = 512
        gcn_dim_pose = 512
        # self.bbox_fc = nn.Sequential(nn.Linear(context_dim, 1024), nn.Linear(1024, 256))
        self.bbox_fc = nn.Linear(boxes_context_dim, gcn_dim)
        
        self.poses_fc = nn.Linear(pose_context_dim, gcn_dim_pose)

        # CDGCN Module
        self.boxes_CDGCN = MultiLayerCDGCN(gcn_dim, 4, 1, cfg)
        self.poses_CDGCN = MultiLayerCDGCN(gcn_dim_pose, 2, 1, cfg)

        self.box_nl = nn.LayerNorm([T, N, gcn_dim])
        self.pose_nl = nn.LayerNorm([T, N, gcn_dim_pose])
        self.dropout_box = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.dropout_pose = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_activities = nn.Linear(gcn_dim, self.cfg.num_activities)
        self.fc_actions = nn.Linear(gcn_dim, self.cfg.num_actions)
        self.pose_fc_activities = nn.Linear(gcn_dim_pose, self.cfg.num_activities)
        self.pose_fc_actions = nn.Linear(gcn_dim_pose, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, poses = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features, inplace=True)

        # pose [B, T*N, 17, 2]
        poses = poses.reshape(B*T*N, -1)
        poses_features = self.pose_fc(poses).reshape(B,T,N,-1) # B,T,N,pose_dim

        # Context Positional Encoding
        context = outputs[-1]
        context_pose = self.context_positionembedding1(context)


        # Embedded Feature Context Encoding
        context_states = self.multilayer_head_embfeature_context_encoding(boxes_features, context, N)
        context_states = context_states.reshape(B, T, N, -1)
        boxes_features = torch.cat((boxes_features, context_states), dim=3)
        boxes_features = boxes_features.reshape(B*T*N, -1)
        boxes_features = self.bbox_fc(boxes_features).reshape(B, T, N, -1)

        # pose feature context encoding
        pose_context = self.multilayer_head_pose_context_encoding(poses_features, context_pose, N)
        pose_context = pose_context.reshape(B, T, N, -1)
        poses_features = torch.cat((poses_features, pose_context), dim=3)
        poses_features = poses_features.reshape(B*T*N, -1)
        poses_features = self.poses_fc(poses_features).reshape(B, T, N, -1)


        # Dynamic graph inference

        graph_boxes_features = self.boxes_CDGCN(boxes_features, boxes_in_flat)
        graph_poses_features = self.poses_CDGCN(poses_features, boxes_in_flat)
   
        torch.cuda.empty_cache()

        
        if self.cfg.backbone == 'inv3' or self.cfg.backbone == 'res18':
            graph_boxes_features = graph_boxes_features.reshape(B, T, N, -1)
            graph_boxes_features = self.box_nl(graph_boxes_features)
            graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
            boxes_features = boxes_features.reshape(B, T, N, -1)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_box(boxes_states)

            graph_poses_features = graph_poses_features.reshape(B, T, N, -1)
            graph_poses_features = self.pose_nl(graph_poses_features)
            graph_poses_features = F.relu(graph_poses_features, inplace=True)
            poses_features = poses_features.reshape(B, T, N, -1)
            poses_states = graph_poses_features + poses_features
            poses_states = self.dropout_pose(poses_states)

        # Predict actions
        boxes_states_flat=boxes_states.reshape(B*T*N, -1)  #B*T*N, NFS
        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num

        poses_states_flat = poses_states.reshape(B*T*N, -1)
        actions_scores2 = self.pose_fc_actions(poses_states_flat)

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(B * T, -1)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        poses_states_pooled, _ = torch.max(poses_states, dim=2)
        poses_states_pooled_flat = poses_states_pooled.reshape(B * T, -1)
        activities_scores2 = self.pose_fc_activities(poses_states_pooled_flat)

        # Temporal fusion
        actions_scores = actions_scores.reshape(B,T,N,-1)
        actions_scores = torch.mean(actions_scores,dim=1).reshape(B*N,-1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        actions_scores2 = actions_scores2.reshape(B,T,N,-1)
        actions_scores2 = torch.mean(actions_scores2,dim=1).reshape(B*N,-1)
        activities_scores2 = activities_scores2.reshape(B, T, -1)
        activities_scores2 = torch.mean(activities_scores2, dim=1).reshape(B, -1)

        return {'activities': [activities_scores, activities_scores2], 'actions': [actions_scores, actions_scores2]}
        # return {'activities': activities_scores, 'actions': actions_scores}
        # return {'activities': [activities_scores, activities_scores2]}
        # return {'activities': activities_scores + activities_scores2, 'actions': actions_scores + actions_scores2}


class PDGCN_collective(nn.Module):
    """
    main module of GCN for the collective dataset
    """
    def __init__(self, cfg):
        super(PDGCN_collective, self).__init__()
        self.cfg = cfg
        num_heads_context = 4
        num_features_context = 128

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes # 1024
        pose_dim = 256

        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)
        # self.avgpool_person = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.fc_emb_2 = nn.Linear(NFB, 512)
        self.nl_emb_1 = nn.LayerNorm([NFB])
        self.nl_emb_2 = nn.LayerNorm([512])
        self.pose_fc = nn.Sequential(nn.Linear(34,1024),nn.Linear(1024,pose_dim))
        self.pose_fc_2 = nn.Linear(34, pose_dim)
        self.nl_pose_1 = nn.LayerNorm([pose_dim])
        

        # TCE Module Loading
        self.multilayer_head_embfeature_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context, 1,
                num_features_context, NFB, N, context_dropout_ratio=0.1)
        self.context_positionembedding1 = Context_PositionEmbeddingSine(16, 768 / 2) # 768 / 2 for inv3
        self.boxes_pe = Embfeature_PositionEmbedding(cfg)

        self.multilayer_head_pose_context_encoding = \
            MultiHeadLayerEmbfeatureContextEncoding(
                num_heads_context, 1,
                num_features_context, pose_dim, N, context_dropout_ratio=0.1)


        boxes_context_dim = NFB + num_heads_context * num_features_context # 1024 + 4*128
        pose_context_dim = pose_dim + num_heads_context * num_features_context
        gcn_dim = 512
        gcn_dim_pose = 512
        # self.bbox_fc = nn.Sequential(nn.Linear(context_dim, 1024), nn.Linear(1024, 256))
        self.bbox_fc = nn.Linear(boxes_context_dim, gcn_dim)
        
        self.poses_fc = nn.Linear(pose_context_dim, gcn_dim_pose)

        # CDGCN Module
        self.boxes_CDGCN = MultiLayerCDGCN(gcn_dim, 4, 1, cfg)
        self.poses_CDGCN = MultiLayerCDGCN(gcn_dim_pose, 4, 1, cfg)

        self.box_nl = nn.LayerNorm([T, gcn_dim])
        self.pose_nl = nn.LayerNorm([T, gcn_dim_pose])
        self.dropout_box = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.dropout_pose = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_activities = nn.Linear(gcn_dim, self.cfg.num_activities)
        self.fc_actions = nn.Linear(gcn_dim, self.cfg.num_actions)
        self.pose_fc_activities = nn.Linear(gcn_dim_pose, self.cfg.num_activities)
        self.pose_fc_actions = nn.Linear(gcn_dim_pose, self.cfg.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def loadpart(self, pretrained_state_dict, model, prefix):
        num = 0
        model_state_dict = model.state_dict()
        pretrained_in_model = collections.OrderedDict()
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, '') in model_state_dict:
                pretrained_in_model[k.replace(prefix, '')] = v
                num += 1
        model_state_dict.update(pretrained_in_model)
        model.load_state_dict(model_state_dict)
        print(str(num) + ' parameters loaded for ' + prefix)

    def forward(self, batch_data):
        images_in, boxes_in, bboxes_num_in, poses = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features_all = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,
        boxes_features_all = boxes_features_all.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features_all = self.fc_emb_1(boxes_features_all)  # B,T,N, NFB
        boxes_features_all = self.nl_emb_1(boxes_features_all)
        boxes_features_all = F.relu(boxes_features_all, inplace=True)

        # pose [B, T*N, 17, 2]
        poses = poses.reshape(B*T*N, -1)
        poses_features_all = self.pose_fc(poses).reshape(B,T,N,-1) # B,T,N,pose_dim

        # poses_features = self.pose_fc_2(poses).reshape(B,T,N,-1)
        # poses_features = self.nl_pose_1(poses_features)
        # poses_features = F.relu(poses_features, inplace=True)

        # Context Positional Encoding
        context_all = outputs[-1] # B*T, 768, OH, OW
        _, ctx_d, ctx_h, ctx_w = context_all.shape
        context_pose_all = self.context_positionembedding1(context_all) # B*T, 768, OH, OW
        context_all = context_all.reshape(B,T,ctx_d,ctx_h,ctx_w)
        context_pose_all = context_pose_all.reshape(B,T,ctx_d,ctx_h,ctx_w)

        # embedded feature positional encoding
        # boxes_features = self.boxes_pe(boxes_features, boxes_in_flat)

        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T,
        actions_scores = []
        activities_scores = []
        actions_scores2 = []
        activities_scores2 = []
        for b in range(B):
            n = bboxes_num_in[b][0]
            boxes_features = boxes_features_all[b, :, :n, :].reshape(1, T, n, -1)  # 1,T,N,d
            poses_features = poses_features_all[b, :, :n, :].reshape(1, T, n, -1)  # 1,T,N,d
            context = context_all[b] # T, 768, OH, OW
            context_pose = context_pose_all[b] # T, 768, OH, OW

            # Embedded Feature Context Encoding
            context_states = self.multilayer_head_embfeature_context_encoding(boxes_features, context, n)
            context_states = context_states.reshape(1, T, n, -1)
            boxes_features = torch.cat((boxes_features, context_states), dim=3)
            boxes_features = boxes_features.reshape(T*n, -1)
            boxes_features = self.bbox_fc(boxes_features).reshape(1, T, n, -1)

            # pose feature context encoding
            pose_context = self.multilayer_head_pose_context_encoding(poses_features, context_pose, n)
            pose_context = pose_context.reshape(1, T, n, -1)
            poses_features = torch.cat((poses_features, pose_context), dim=3)
            poses_features = poses_features.reshape(T*n, -1)
            poses_features = self.poses_fc(poses_features).reshape(1, T, n, -1)

            # Dynamic graph inference

            graph_boxes_features = self.boxes_CDGCN(boxes_features, boxes_in_flat) # 1, T, n, -1
            graph_poses_features = self.poses_CDGCN(poses_features, boxes_in_flat)
   
            torch.cuda.empty_cache()

            if self.cfg.backbone == 'vgg16' or self.cfg.backbone == 'res18':
                # residual first
                boxes_states = graph_boxes_features + boxes_features
                boxes_states = boxes_states.permute(0, 2, 1, 3).contiguous().view(n, T, -1) # n, T, d
                boxes_states = self.box_nl(boxes_states)
                boxes_states = F.relu(boxes_states, inplace=True)
                boxes_states = self.dropout_box(boxes_states)

                poses_states = graph_poses_features + poses_features
                poses_states = poses_states.permute(0, 2, 1, 3).contiguous().view(n, T, -1) # n, T, d
                poses_states = self.pose_nl(poses_states)
                poses_states = F.relu(poses_states, inplace=True)
                poses_states = self.dropout_pose(poses_states)
            elif self.cfg.backbone == 'inv3':
                graph_boxes_features = graph_boxes_features.permute(0, 2, 1, 3).contiguous().view(n, T, -1)
                graph_boxes_features = self.box_nl(graph_boxes_features)
                graph_boxes_features = F.relu(graph_boxes_features, inplace=True)
                boxes_features = boxes_features.permute(0, 2, 1, 3).contiguous().view(n, T, -1) # n, T, d
                boxes_states = graph_boxes_features + boxes_features
                boxes_states = self.dropout_box(boxes_states)

                graph_poses_features = graph_poses_features.permute(0, 2, 1, 3).contiguous().view(n, T, -1)
                graph_poses_features = self.pose_nl(graph_poses_features)
                graph_poses_features = F.relu(graph_poses_features, inplace=True)
                poses_features = poses_features.permute(0, 2, 1, 3).contiguous().view(n, T, -1) # n, T, d
                poses_states = graph_poses_features + poses_features
                poses_states = self.dropout_pose(poses_states)
                

            # Predict actions
            boxes_states_flat = boxes_states.permute(1, 0, 2).contiguous().view(T*n, -1)  #T*n, -1
            actn_score = self.fc_actions(boxes_states_flat).reshape(T, n, -1)  #T, n, actn_num
            actn_score = torch.mean(actn_score, dim=0).reshape(n, -1)  # N, actn_num
            actions_scores.append(actn_score)

            poses_states_flat = poses_states.permute(1, 0, 2).contiguous().view(T*n, -1)  #T*n, -1
            actn_score2 = self.pose_fc_actions(poses_states_flat).reshape(T, n, -1)  #T, n, actn_num
            actn_score2 = torch.mean(actn_score2, dim=0).reshape(n, -1)  # N, actn_num
            actions_scores2.append(actn_score2)


            # Predict activities
            boxes_states_pooled, _ = torch.max(boxes_states, dim=0) # T, d
            acty_score = self.fc_activities(boxes_states_pooled)  # T, acty_num
            acty_score = torch.mean(acty_score, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores.append(acty_score)

            poses_states_pooled, _ = torch.max(poses_states, dim=0)# T, d
            acty_score2 = self.pose_fc_activities(poses_states_pooled)
            acty_score2 = torch.mean(acty_score2, dim=0).reshape(1, -1)  # 1, acty_num
            activities_scores2.append(acty_score2)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        actions_scores2 = torch.cat(actions_scores2, dim=0)  # ALL_N,actn_num
        activities_scores2 = torch.cat(activities_scores2, dim=0)  # B,acty_num

        return {'activities': [activities_scores, activities_scores2], 'actions': [actions_scores, actions_scores2]}
        # return {'activities': activities_scores2, 'actions': actions_scores2}



class ARG_volleyball(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, cfg):
        super(ARG_volleyball, self).__init__()
        self.cfg = cfg

        T, N = self.cfg.num_frames, self.cfg.num_boxes
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph


        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'res18':
            self.backbone = MyRes18(pretrained=True)
        elif cfg.backbone == 'alex':
            self.backbone = MyAlex(pretrained=True)
        else:
            assert False

        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


        self.roi_align = RoIAlign(*self.cfg.crop_size)
        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        # self.fc_actions = nn.Linear(NFG, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFG, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)

    def forward(self, batch_data):
        images_in, boxes_in = batch_data

        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        if not self.training:
            B = B * 3
            T = T // 3
            images_in.reshape((B, T) + images_in.shape[2:])
            boxes_in.reshape((B, T) + boxes_in.shape[2:])

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * N,))  # B*T*N,

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH, OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)

        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N, D, K, K,

        boxes_features = boxes_features.reshape(B, T, N, -1)  # B,T,N, D*K*K

        # Embedding
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)

        # GCN
        graph_boxes_features = boxes_features.reshape(B, T * N, NFG)

        #         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)
        #             visual_info.append(relation_graph.reshape(B,T,N,N))

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = boxes_features.reshape(B, T, N, NFB)

        #         boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features

        boxes_states = self.dropout_global(boxes_states)

        NFS = NFG

        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFS)  # B*T*N, NFS
        # actions_scores = self.fc_actions(boxes_states_flat)  # B*T*N, actn_num

        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  # B*T, acty_num

        # Temporal fusion
        # actions_scores = actions_scores.reshape(B, T, N, -1)
        # actions_scores = torch.mean(actions_scores, dim=1).reshape(B * N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B, -1)

        if not self.training:
            B = B // 3
            # actions_scores = torch.mean(actions_scores.reshape(B, 3, N, -1), dim=1).reshape(B * N, -1)
            activities_scores = torch.mean(activities_scores.reshape(B, 3, -1), dim=1).reshape(B, -1)

        # return [activities_scores] # actions_scores, #'boxes_states':boxes_states
        # return {'activities':activities_scores, 'actions_scores':actions_scores}
        return {'activities':activities_scores}
