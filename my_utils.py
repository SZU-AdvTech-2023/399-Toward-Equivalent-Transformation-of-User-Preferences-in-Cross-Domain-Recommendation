import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.metrics import auc
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm

# 计算最大均值差异（Maximum Mean Discrepancy，MMD）。用于衡量两个概率分布之间差异的方法。
# 在这个函数中，MMD的计算涉及到了核方法，尤其是径向基函数（RBF）核。
def MMD(X, Y, biased=True):
    # set params to calculate MMD distance
    sigma_list = [1e-2, 1e-1, 1, 10, 100]
    sigma_list = torch.FloatTensor(np.array(sigma_list))
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

# 一个混合径向基函数核（Mixed Radial Basis Function Kernel）的计算函数 _mix_rbf_kernel。该核函数用于计算两个输入数据集 X 和 Y 之间的核矩阵。
def _mix_rbf_kernel(X, Y, sigma_list):
    # 确保输入数据集 X 和 Y 的样本数量相同
    assert(X.size(0) == Y.size(0))
    m = X.size(0) # 样本数量

    # 将两个数据集合并成一个新的数据集 Z
    Z = torch.cat((X, Y), 0)
    # 计算 Z 和 Z 转置的乘积，得到 ZZ^T
    ZZT = torch.mm(Z, Z.t())
    # 提取 ZZ^T 对角线上的元素，得到对角元素组成的列向量
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    # 将对角元素扩展为与 ZZ^T 相同大小的矩阵，得到 Z_norm_sqr
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    # 计算指数部分，得到 exponent
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    # 初始化核矩阵 K
    K = 0.0
    # 遍历给定的径向基函数核宽度参数列表 sigma_list
    for sigma in sigma_list:
        # 计算高斯核的参数 gamma
        gamma = 1.0 / (2 * sigma**2)
        # 将高斯核的贡献累加到核矩阵 K 中
        K += torch.exp(-gamma * exponent)
    # 将核矩阵 K 分割成四个部分，并返回结果
    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

#最大均值差异（Maximum Mean Discrepancy，MMD），用于衡量两个样本集合的相似性。
#MMD的基本思想是通过核方法比较两个分布，核矩阵（Gram 矩阵）来表示样本之间的相似度。
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def evaluation(preds, topk):
    sort = np.argsort(-preds, axis=1)[:, :topk]
    hr_arr = np.zeros(shape=[sort.shape[0]])
    ndcg_arr = np.zeros(shape=[sort.shape[0]])
    mrr_arr = np.zeros(shape=[sort.shape[0]])
    rows = np.where(sort==99)[0]
    cols = np.where(sort==99)[1]
    hr_arr[rows] = 1.0
    ndcg_arr[rows] = np.log(2) / np.log(cols + 2.0)
    mrr_arr[rows] = 1.0 / (cols + 1.0)
    return hr_arr.tolist(), ndcg_arr.tolist(), mrr_arr.tolist()

def test_process(model, train_loader, feed_data, is_cuda, topK,  mode='val'):
    all_hr1_list = []
    all_ndcg1_list = []
    all_mrr1_list = []
    all_hr2_list = []
    all_ndcg2_list = []
    all_mrr2_list = []
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    if mode == 'val':
        movie_nega = feed_data['movie_nega']
        movie_test = feed_data['movie_vali']
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_vali']
    elif mode=='test':
        movie_nega = feed_data['movie_nega']
        movie_test = feed_data['movie_test']
        book_nega = feed_data['book_nega']
        book_test = feed_data['book_test']
    else:
        raise Exception


    for batch_idx, data in enumerate(train_loader):
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

        res = model.forward(v_user, v_item1, v_item2)
        y1 = res[0]
        y2 = res[1]
        if is_cuda:
            y1 = y1.detach().cpu().numpy()
            y2 = y2.detach().cpu().numpy()
        else:
            y1 = y1.detach().numpy()
            y2 = y2.detach().numpy()


        nega_vali1 = np.array([movie_nega[ele] + [movie_test[ele]] for ele in val_user_arr])
        nega_vali2 = np.array([book_nega[ele] + [book_test[ele]] for ele in val_user_arr])
        pred1 = np.stack([y1[xx][nega_vali1[xx]] for xx in range(nega_vali1.shape[0])])
        pred2 = np.stack([y2[xx][nega_vali2[xx]] for xx in range(nega_vali2.shape[0])])
        hr1_list, ndcg1_list, mrr1_list = evaluation(pred1, topK)
        hr2_list, ndcg2_list, mrr2_list = evaluation(pred2, topK)
        all_hr1_list += hr1_list
        all_ndcg1_list += ndcg1_list
        all_mrr1_list += mrr1_list
        all_hr2_list += hr2_list
        all_ndcg2_list += ndcg2_list
        all_mrr2_list += mrr2_list

    avg_hr1 = np.mean(all_hr1_list)
    avg_ndcg1 = np.mean(all_ndcg1_list)
    avg_mrr1 = np.mean(all_mrr1_list)
    avg_hr2 = np.mean(all_hr2_list)
    avg_ndcg2 = np.mean(all_ndcg2_list)
    avg_mrr2 = np.mean(all_mrr2_list)

    return avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2


