import csv
import os.path as osp
import numpy as np


class GetData(object):
    def __init__(self, root, miRNA_num=713, dis_num=447):
        super().__init__()
        self.root = osp.join(root, 'HMDD3.2_processed')
        self.miRNA_num = miRNA_num
        self.dis_num = dis_num
        self.dis_sim, self.type_tensor = self.__get_data__()

    def __get_data__(self):
        type_name = ['target', 'circu', 'epic', 'genetic', 'tissue']
        type_association_matrix = np.zeros((self.miRNA_num, self.dis_num, 5))
        for i in range(5):
            with open(osp.join(self.root, '{}.csv'.format(type_name[i])), 'r') as type_:
                type_mat = csv.reader(type_)
                row = -1

                for line in type_mat:
                    if row >= 0:
                        col = -1
                        for association in line:
                            if col >= 0:
                                type_association_matrix[row, col, i] = eval(association)
                            col = col + 1
                    row = row + 1

        disease_similarity_mat = np.zeros((self.dis_num, self.dis_num))
        with open(osp.join(self.root, 'Dis_sim.csv'), 'r') as dis_sim:
            sim_mat = csv.reader(dis_sim)
            row = -1

            for line in sim_mat:
                if row >= 0:
                    col = -1
                    for sim in line:
                        if col >= 0:
                            disease_similarity_mat[row, col] = eval(sim)
                        col = col + 1
                row = row + 1
        disease_similarity_mat = np.asmatrix(disease_similarity_mat)
        disease_similarity_mat = disease_similarity_mat - np.diag(np.diag(disease_similarity_mat))
        return disease_similarity_mat, type_association_matrix

    def get_functional_sim(self, mir_dis_mat):
        """Vectorized version — tránh O(N²) nested loop Python.

        Wang's miRNA functional similarity:
          sim(m1, m2) = (sum_max_axis0 + sum_max_axis1) / (deg(m1) + deg(m2))
          với mat[d1, d2] = mir(m1, d1) * S_d(d1, d2) * mir(m2, d2)
          sum_max_axis0 = sum over d2 của max over d1
          sum_max_axis1 = sum over d1 của max over d2

        Vectorize bằng cách dùng broadcasting + max-reduce với O(N*D²) memory.
        Với N=713, D=447: 713*447² ≈ 142M floats ≈ 570MB → quá lớn.
        Thay vào đó: lặp m1 nhưng cho mỗi m1 vectorize toàn bộ m2.
        """
        mir_fun_sim_matrix = np.zeros((self.miRNA_num, self.miRNA_num))
        dis_semantic_sim = np.asarray(self.dis_sim) - np.diag(np.diag(np.asarray(self.dis_sim))) + np.eye(self.dis_num)
        mir_dis_mat = np.asarray(mir_dis_mat, dtype=np.float32)  # (N, D)
        link_num = mir_dis_mat.sum(axis=1)  # (N,)

        for m1 in range(self.miRNA_num):
            v1 = mir_dis_mat[m1]  # (D,)
            if link_num[m1] == 0:
                continue
            # M1 = v1[d1] * S(d1, d2) shape (D, D)
            M1 = v1[:, None] * dis_semantic_sim
            # m1_m2_sim[m2, d1, d2] = M1[d1, d2] * mir_dis_mat[m2, d2]
            # axis=0 reduction (over d1): max_axis0[m2, d2] = max over d1 (M1[d1, d2]) * mir_dis_mat[m2, d2]
            max_axis0 = M1.max(axis=0)  # (D,) — không phụ thuộc m2 trong cùng dim 0
            # m1_max_sum[m2] = sum_{d2} max_axis0[d2] * mir_dis_mat[m2, d2]
            m1_max_sum = mir_dis_mat @ max_axis0  # (N,)

            # axis=1 reduction (over d2): max_axis1[d1, m2] = max over d2 (M1[d1, d2] * mir_dis_mat[m2, d2])
            # → (D, D) * mir_dis_mat[m2, d2] = (m2, d1, d2), max over d2 → (m2, d1)
            # Compute với einsum-style: M1 mở rộng × mir_dis_mat → max over d2
            # mem: (N, D, D) = 713*447*447*4 ≈ 570MB. Quá lớn.
            # Workaround: chunk over m2 batches.
            chunk_size = 32
            m2_max_sum = np.zeros(self.miRNA_num, dtype=np.float32)
            for start in range(0, self.miRNA_num, chunk_size):
                end = min(start + chunk_size, self.miRNA_num)
                # M1 (D, D), mir_dis_mat[start:end] (B, D)
                # tmp (B, D, D) = M1[None] * mir_dis_mat[start:end, None, :]
                tmp = M1[None, :, :] * mir_dis_mat[start:end, None, :]
                # max over d2 → (B, D), sum over d1 → (B,)
                m2_max_sum[start:end] = tmp.max(axis=2).sum(axis=1)

            denom = (link_num[m1] + link_num)  # (N,)
            denom_safe = np.where(denom > 0, denom, 1)
            sim_vals = (m1_max_sum + m2_max_sum) / denom_safe
            sim_vals[denom == 0] = 0
            # Chỉ ghi nửa trên + chéo (m2 ≥ m1)
            mir_fun_sim_matrix[m1, m1:] = sim_vals[m1:]
            mir_fun_sim_matrix[m1:, m1] = sim_vals[m1:]

        mir_fun_sim_matrix = mir_fun_sim_matrix - np.diag(np.diag(mir_fun_sim_matrix))
        mir_fun_sim_matrix = np.nan_to_num(mir_fun_sim_matrix)
        return mir_fun_sim_matrix
