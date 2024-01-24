from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

#协同降维自编码器（Collaborative Denoising Autoencoder，CDAE）模型。用于学习用户和物品的嵌入表示。
class CDAE(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, is_sparse=False):
        super(CDAE, self).__init__()

        # 定义模型的参数和层结构
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE

        # 用户嵌入层，用于学习用户的嵌入表示
        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()

        # 编码器和解码器的结构，用于学习用户和物品的嵌入表示
        self.encoder_x = nn.Sequential(
            nn.Linear(self.NUM_MOVIE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_x = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_MOVIE)
            )
        self.encoder_y = nn.Sequential(
            nn.Linear(self.NUM_BOOK, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
            )
        self.decoder_y = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_BOOK)
            )

        # 正交权重参数，用于强制用户和物品的嵌入表示正交
        self.orthogonal_w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(EMBED_SIZE, EMBED_SIZE).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)

        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # ReLU 激活函数
        self.relu = nn.ReLU

    # 正交映射方法。给定两个输入张量 z_x 和 z_y，通过矩阵乘法将它们分别映射到正交权重矩阵上的新空间。
    def orthogonal_map(self, z_x, z_y):
        # 将输入张量 z_x 映射到正交权重矩阵上的新空间
        mapped_z_x = torch.matmul(z_x, self.orthogonal_w)
        # 将输入张量 z_y 映射到正交权重矩阵上的新空间（转置正交权重矩阵进行映射）
        mapped_z_y = torch.matmul(z_y, torch.transpose(self.orthogonal_w, 1, 0))
        # 返回映射后的张量1和张量2
        return mapped_z_x, mapped_z_y


    def forward(self, batch_user, batch_user_x, batch_user_y):
        """
        CDAE 模型的前向传播过程

        参数：
        - batch_user: 用户索引张量
        - batch_user_x: X特征张量
        - batch_user_y: Y特征张量

        返回：
        - preds_x: X的重构预测
        - preds_y: Y的重构预测
        - preds_x2y: 从X到Y的映射预测
        - preds_y2x: 从Y到X的映射预测
        - z_x: 域X编码表示
        - z_y: 域Y编码表示
        - z_x_reg_loss: 正交性约束损失（域X编码表示）
        - z_y_reg_loss: 正交性约束损失（域Y编码表示）
        """
        # 编码过程
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)

        # 解码过程
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)

        # 正交映射过程
        mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, z_y)
        preds_x2y = self.decoder_y(mapped_z_x)
        preds_y2x = self.decoder_x(mapped_z_y)

        # 定义正交性约束损失
        z_x_ = torch.matmul(mapped_z_x, torch.transpose(self.orthogonal_w, 1, 0))
        z_y_ = torch.matmul(mapped_z_y, self.orthogonal_w)
        z_x_reg_loss = torch.norm(z_x - z_x_, p=1, dim=1)
        z_y_reg_loss = torch.norm(z_y - z_y_, p=1, dim=1)

        return preds_x, preds_y, preds_x2y, preds_y2x, feature_x, feature_y, z_x_reg_loss, z_y_reg_loss


    def get_user_embedding(self, batch_user_x, batch_user_y):
        """
        获取用户嵌入表示的方法

        参数：
        - batch_user_x: X特征张量
        - batch_user_y: Y特征张量

        返回：
        - h_user_x: X特征的编码表示
        - h_user_y: Y特征的编码表示
        """
        # this is for SIGIR's experiment
        # 编码过程
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        # 返回编码后的用户特征表示
        return h_user_x, h_user_y

    def get_latent_z(self, batch_user, batch_user_x, batch_user_y):
        # this is for clustering visualization
        # 编码过程
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)

        # 返回潜在表示
        return z_x, z_y



# 鉴别器（Discriminator），用于对输入数据进行分类。
class Discriminator(nn.Module):
    def __init__(self, n_fts, dropout):
        # 初始化鉴别器的参数和层结构
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.training = True

        # 定义鉴别器的多层感知机（MLP）结构
        self.disc = nn.Sequential(
            nn.Linear(n_fts, int(n_fts/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_fts/2), 1))

    def forward(self, x):
        # make mlp for discriminator
        # 鉴别器的前向传播过程，通过多层感知机进行处理
        h = self.disc(x)

        # 返回鉴别器的输出结果
        return h

def save_embedding_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_user_embedding(v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)

def save_z_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_latent_z(v_user, v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)