import torch
import math
from utils import *
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCDGCN(nn.Module):
    """CDGCN reason module"""
    def __init__(self, d_model, num_heads, cfg, dropout_ratio=0.1):
        super(MultiHeadCDGCN, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_Q = nn.Linear(d_model, self.d_head * num_heads, bias = False)
        self.W_K = nn.Linear(d_model, self.d_head * num_heads, bias = False)
        self.W_V = nn.Linear(d_model, self.d_head * num_heads, bias = False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_ratio)
        self.OH, self.OW = cfg.out_size
        self.pos_threshold = cfg.pos_threshold # 0.2


    def forward(self, x: torch.Tensor, boxes_in_flat: torch.Tensor):
        """
        x: [B, T, N, D]
        """
        B, T, N, d = x.shape
        # Prepare position mask
        # graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        # graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        # graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        # graph_boxes_positions = graph_boxes_positions[:, :2].reshape(-1, N, 2)  # B*T, N, 2

        # graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B*T, N, N

        # position_mask = (graph_boxes_distances > (self.pos_threshold * self.OW))
        # position_mask = position_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        
        # T_sum = torch.sum(x, dim=1) # B, N, d
        # T_sum = T_sum.unsqueeze(dim=1)
        # T_sum = x / T_sum # B, T, N, d
        T_sum = F.softmax(x, dim=1)
        T_sum = torch.mul(x, T_sum)
        TAtt = torch.sum(T_sum, dim=1) # B, N, d
        TAtt = TAtt.unsqueeze(1).repeat(1, T, 1, 1)
        TAtt = TAtt.view((B*T, N, -1))

        x = x.view(B*T, N, -1)
        Q = self.W_Q(x).view(B*T, N, self.num_heads, self.d_head).transpose(1, 2) # BT, n_heads, N, d_head
        K = self.W_K(TAtt).view(B*T, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(TAtt).view(B*T, N, self.num_heads, self.d_head).transpose(1, 2)

        A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_head) # QK^T [BT, n_heads, N, N]
        # A[position_mask] = 0
        A = F.relu(A, inplace=True)

        I_N = torch.eye(N, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        A = A + I_N

        # D = torch.sum(A, dim = -1) # [BT, n_heads, N]
        # D = torch.pow(D, -1)
        # D = D.unsqueeze(-1)
        # A = D*A

        # A = torch.softmax(A, dim=-1)
        output = torch.matmul(A, V)
        output = output.transpose(1, 2).contiguous().view(B, T, N, self.num_heads*self.d_head)
        # output = self.layer_norm(self.dropout(output) + x) # [B, T, N, d]

        return output


class MultiLayerCDGCN(nn.Module):
    """Multi head and Multi Layer CDGCN"""
    def __init__(self, d_model, num_heads, num_layers, cfg, dropout_ratio=0.1):
        super(MultiLayerCDGCN, self).__init__()
        self.CDGCN = nn.ModuleList()
        for i in range(num_layers):
            self.CDGCN.append(MultiHeadCDGCN(d_model, num_heads, cfg, dropout_ratio))
        self.num_layers = num_layers
    

    def forward(self, x, boxes_in_flat):
        """
        x: [B, T, N, D]
        """
        output = x
        for i in range(self.num_layers):
            output = self.CDGCN[i](output, boxes_in_flat)
        
        return output


if __name__ == "__main__":
    x = torch.randn(4, 3, 12, 256)
    cdgcn = MultiLayerCDGCN(256, 4, 2)

    output = cdgcn(x)
    print(output.shape)