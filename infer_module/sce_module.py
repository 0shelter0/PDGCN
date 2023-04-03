import torch.nn as nn
import torch
import torch.nn.functional as F

#################       Context Encoding Module        ###################


class EmbfeatureContextEncodingTransformer(nn.Module):
    def __init__(self, num_features_context, NFB, N, layer_id, num_heads_per_layer, context_dropout_ratio = 0.1):
        super(EmbfeatureContextEncodingTransformer, self).__init__()
        self.num_features_context = num_features_context
        if layer_id == 1:
            self.downsample2 = nn.Conv2d(768, num_features_context, kernel_size = 1, stride=1)
            nn.init.kaiming_normal_(self.downsample2.weight)
            '''nn.init.kaiming_normal_(self.downsample1.weight)
            nn.init.kaiming_normal_(self.downsample2.weight)
            self.downsample = nn.Conv2d(D, num_features_context, kernel_size=1, stride=1)'''
            self.emb_roi = nn.Linear(NFB, num_features_context, bias=True)
        elif layer_id > 1:
            self.downsample = nn.Conv2d(768, num_features_context, kernel_size=1, stride=1)
            self.emb_roi = nn.Linear(num_features_context * num_heads_per_layer, num_features_context, bias=True)
            nn.init.kaiming_normal_(self.downsample.weight)
        self.N = N
        self.dropout = nn.Dropout(context_dropout_ratio)
        self.layernorm1 = nn.LayerNorm(num_features_context)
        self.FFN = nn.Sequential(
            nn.Linear(num_features_context,num_features_context, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(context_dropout_ratio),
            nn.Linear(num_features_context,num_features_context, bias = True)
        )
        self.layernorm2 = nn.LayerNorm(num_features_context)
        self.att_map = None


    def forward(self, roi_feature, image_feature, N, layer_id = -1):
        """

        :param roi_feature:   # B*T*N, NFB torch.Size([1, 4, 2, 1024])
        :param image_feature: # B*T, D, OH, OW torch.Size([4, 768, 28, 43])
        :return:
        """
        NFC = self.num_features_context
        BT, _,OH,OW = image_feature.shape # BT =4
        
        # assert N==12
        assert layer_id>=1
        if layer_id == 1:
            image_feature = self.downsample2(image_feature) # [4, 128, 28, 43]
            emb_roi_feature = self.emb_roi(roi_feature) # B*T*N, D [1, 4, 2, 128]
        elif layer_id > 1:
            emb_roi_feature = self.emb_roi(roi_feature)
            image_feature = self.downsample(image_feature)
        emb_roi_feature = emb_roi_feature.reshape(BT, N, 1, 1, NFC) # B*T, N, 1, 1, D
        image_feature = image_feature.reshape(BT, 1, NFC, OH, OW) # B*T, 1, D, OH, OW
        image_feature = image_feature.transpose(2,3) # B*T, 1, OH, D, OW

        a = torch.matmul(emb_roi_feature, image_feature) # B*T, N, OH, 1, OW
        a = a.reshape(BT, N, -1) # B*T, N, OH*OW
        # a = a / math.sqrt(NFC)
        A = F.softmax(a, dim=2)  # B*T, N, OH*OW
        self.att_map = A
        image_feature = image_feature.transpose(3,4).reshape(BT, OH*OW, NFC)

        context_encoding_roi = self.dropout(torch.matmul(A, image_feature).reshape(BT*N, NFC))
        emb_roi_feature = emb_roi_feature.reshape(BT*N, NFC)
        context_encoding_roi = self.layernorm1(context_encoding_roi + emb_roi_feature)
        context_encoding_roi = context_encoding_roi + self.FFN(context_encoding_roi)
        context_encoding_roi = self.layernorm2(context_encoding_roi)
        return context_encoding_roi


class MultiHeadLayerEmbfeatureContextEncoding(nn.Module):
    def __init__(self, num_heads_per_layer, num_layers, num_features_context, NFB, N, context_dropout_ratio=0.1):
        super(MultiHeadLayerEmbfeatureContextEncoding, self).__init__()
        self.CET = nn.ModuleList()
        for i in range(num_layers):
            for j in range(num_heads_per_layer):
                self.CET.append(EmbfeatureContextEncodingTransformer(num_features_context, NFB, N, i+1, num_heads_per_layer, context_dropout_ratio))
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.vis_att_map = torch.empty((0, 13, 43 * 78), dtype = torch.float32)
        self.output_emb = nn.Linear(num_features_context*num_heads_per_layer, num_features_context, bias=True)

    def forward(self, roi_feature, image_feature, N):
        """
        :param roi_feature:   # B*T*N, NFB,
        :param image_feature: # B*T, D, OH, OW
        :return:
        """
        for i in range(self.num_layers):
            MHL_context_encoding_roi= []
            for j in range(self.num_heads_per_layer):
                MHL_context_encoding_roi.append(self.CET[i*self.num_heads_per_layer + j](roi_feature, image_feature, N, i+1))
            roi_feature = torch.cat(MHL_context_encoding_roi, dim=1)
        
        # roi_feature = self.output_emb(roi_feature)
        return roi_feature
