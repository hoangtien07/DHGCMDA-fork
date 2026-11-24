# Port HMDD v3.2

Mã nền `06f9ba3`. Chỉ thay kích thước dữ liệu, số loại, ánh xạ nhãn và số đếm lớp. Trình chấm điểm giữ nguyên từng byte. Top-1 cho 5 loại được ghi riêng từ dự đoán đã lưu. Bản phát hành gốc nằm trong `release/`.

```diff
--- hetero_model.py
+++ v32/hetero_model.py
@@ -592,14 +592,14 @@
         self.association_predictor = SimplifiedTypePredictor(
             node_dim=out_channels,
             hidden_dim=128,
-            num_types=4
+            num_types=args.num_association_types
         )
 
         self.dropout = nn.Dropout(args.dropout)
 
         # 4种类型的分布跟踪（基于实际数据）
         self.register_buffer('type_distribution',
-                             torch.tensor([0.219, 0.094, 0.175, 0.407]))
+                             torch.tensor([1311, 155, 2613, 756, 3900], dtype=torch.float32) / 8735)
         self.update_type_counts = True
 
     def _get_association_matrix(self, concat_mi_tensor, concat_dis_tensor):
--- main_experiments_hetero1.py
+++ v32/main_experiments_hetero1.py
@@ -62,7 +62,7 @@
         self.device = device
 
         # 🔥 Effective Number 类别权重 (更激进的平衡)
-        counts = [367, 157, 293, 681]  # 循环, 表观, 靶标, 遗传
+        counts = [1311, 155, 2613, 756, 3900]  # 循环, 表观, 靶标, 遗传
         beta = 0.99999  # 🔥 改进: 更接近1 → 给少数类更高权重
         effective_nums = [(1 - beta ** n) / (1 - beta) for n in counts]
         raw_weights = [1.0 / en for en in effective_nums]
@@ -144,6 +144,7 @@
         type_indices[pos_targets == 2] = 1
         type_indices[pos_targets == 3] = 2
         type_indices[pos_targets == 4] = 3
+        type_indices[pos_targets == 5] = 4
 
         valid_mask = pos_targets > 0
         if valid_mask.sum() == 0:
@@ -301,7 +302,7 @@
         if isinstance(train_data, list) and len(train_data) >= 5:
             association_matrix = train_data[4].to(device).float()
         else:
-            association_matrix = torch.zeros((495, 380), device=device).float()
+            association_matrix = torch.zeros((411, 271), device=device).float()
 
         target_mi_num = association_matrix.shape[0]
         target_dis_num = association_matrix.shape[1]
--- param.py
+++ v32/param.py
@@ -10,8 +10,8 @@
     # Dataset parameters
     parser.add_argument('--dataset',
                         type=str,
-                        default='v2.0_495m383D',
-                        help='Dataset name: v2.0_495m383D')
+                        default='v3.2_spld_paper',
+                        help='Dataset name: v3.2_spld_paper')
 
     # 添加数据路径参数 - 修复 AttributeError
     parser.add_argument('--data_path',
@@ -248,7 +248,7 @@
     if args.class_weights is None:
         # 使用反比例权重，强调少数类以防止类别崩溃
         # Circulation: 367 (24.5%), Epigenetics: 157 (10.5%), Target: 293 (19.6%), Genetics: 681 (45.5%)
-        counts = [367, 157, 293, 681]  # 实际样本数
+        counts = [1311, 155, 2613, 756, 3900]  # 实际样本数
         total = sum(counts)
 
         # 计算反比例权重（样本越少，权重越高）
@@ -266,9 +266,9 @@
         print(f"  [NOTE] Higher weights for minority classes to prevent model from only predicting majority class")
 
     # Validate number of association types
-    if args.num_association_types != 4:
+    if args.num_association_types != 5:
         print(f"[WARNING] num_association_types is set to {args.num_association_types}.")
-        print("  The model expects 4 types: circulation, epigenetics, target, genetics.")
+        print("  The model expects 5 types: circulation, epigenetics, target, genetics, tissue.")
 
     # Validate inter-view CL parameters
     if args.enable_inter_view_cl:
@@ -277,14 +277,14 @@
             args.inter_view_weight = 0.3
 
     # Dataset-specific settings
-    if args.dataset == 'v2.0_495m383D':
-        args.mi_num = 495
-        args.dis_num = 383
-        print(f"[INFO] Using v2.0_495m383D dataset:")
+    if args.dataset == 'v3.2_spld_paper':
+        args.mi_num = 411
+        args.dis_num = 271
+        print(f"[INFO] Using v3.2_spld_paper dataset:")
         print(f"  miRNAs: {args.mi_num}")
         print(f"  Diseases: {args.dis_num}")
-        print(f"  Expected associations: ~1498")
-        print(f"  Types: 4 (Circulation, Epigenetics, Target, Genetics)")
+        print(f"  Expected associations: 8735")
+        print(f"  Types: 5 (Circulation, Epigenetics, Target, Genetics, Tissue)")
     else:
         print(f"[WARNING] Unknown dataset: {args.dataset}")
         print("  Using default dimensions (495 miRNAs, 383 diseases)")
--- prepareData.py
+++ v32/prepareData.py
@@ -67,8 +67,8 @@
         association_types = data.iloc[:, 2].values  # 保持原始类型编码 1,2,3,4
 
         # 创建关联矩阵 - 495个miRNA × 383个疾病
-        max_mirna = 495
-        max_disease = 383
+        max_mirna = 411
+        max_disease = 271
         association_matrix = np.zeros((max_mirna, max_disease), dtype=np.float32)
 
         # 填充关联矩阵（保持原始类型值 1,2,3,4）
@@ -85,7 +85,8 @@
             1: "循环类型 (Circulation)",
             2: "表观遗传学 (Epigenetics)",
             3: "靶标类型 (Target)",
-            4: "遗传学类型 (Genetics)"
+            4: "遗传学类型 (Genetics)",
+            5: "Tissue"
         }
 
         unique_vals, counts = np.unique(association_matrix, return_counts=True)
@@ -247,6 +248,7 @@
             'type2': [],
             'type3': [],
             'type4': [],
+            'type5': [],
             'nonzero': []
         }
 
@@ -263,6 +265,8 @@
                 type_indices['type3'].append([i, j])
             elif value == 4:
                 type_indices['type4'].append([i, j])
+            elif value == 5:
+                type_indices['type5'].append([i, j])
 
         zero_indices_i, zero_indices_j = np.where(association_matrix == 0)
         for i, j in zip(zero_indices_i, zero_indices_j):
@@ -310,6 +314,7 @@
                 'type2': type_indices['type2'],
                 'type3': type_indices['type3'],
                 'type4': type_indices['type4'],
+                'type5': type_indices['type5'],
             }
         }
 
@@ -327,10 +332,10 @@
     dataset = {}
     preprocessor = OptimizedDataPreprocessor()
 
-    print("Loading all similarity and feature files from v2.0_495m383D...")
+    print("Loading all similarity and feature files from v3.2_spld_paper...")
 
     # 1. 加载疾病语义相似性 (D_SSM1.txt) - Disease View 2的基础
-    dd_sem_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'D_SSM1.txt')
+    dd_sem_path = os.path.join(opt.data_path, 'v3.2_spld_paper', 'D_SSM1.txt')
     if os.path.exists(dd_sem_path):
         dd_sem_data = read_txt(dd_sem_path)
         dd_sem_mat = dd_sem_data.numpy().astype(np.float32)
@@ -340,7 +345,7 @@
         return None
 
     # 2. 加载miRNA功能相似性 (M_FSM.txt) - miRNA View 2的基础
-    mm_fun_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'M_FSM.txt')
+    mm_fun_path = os.path.join(opt.data_path, 'v3.2_spld_paper', 'M_FSM.txt')
     if os.path.exists(mm_fun_path):
         mm_fun_data = read_txt(mm_fun_path)
         mm_fun_mat = mm_fun_data.numpy().astype(np.float32)
@@ -350,7 +355,7 @@
         return None
 
     # 3. 加载关联数据并转换为矩阵 (multi_all_mirna_disease_pairs_without_negative.csv)
-    mi_dis_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'multi_all_mirna_disease_pairs_without_negative.csv')
+    mi_dis_path = os.path.join(opt.data_path, 'v3.2_spld_paper', 'multi_all_mirna_disease_pairs_without_negative.csv')
     if os.path.exists(mi_dis_path):
         association_matrix_tensor = read_association_csv(mi_dis_path)
         if association_matrix_tensor is not None:
@@ -367,7 +372,7 @@
     dataset['md_true'] = dataset['md_p']
 
     # 4. 加载疾病-基因特征 (D_SSM2.txt) - Disease View 1的基础
-    d_gs_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'D_SSM2.txt')
+    d_gs_path = os.path.join(opt.data_path, 'v3.2_spld_paper', 'D_SSM2.txt')
     if os.path.exists(d_gs_path):
         d_gs_data = read_txt(d_gs_path)
         dataset['d_gs'] = d_gs_data
@@ -377,7 +382,7 @@
         dataset['d_gs'] = t.FloatTensor(dd_sem_mat)
 
     # 5. 加载miRNA-序列特征 (M_GSM.txt) - miRNA View 1的基础
-    m_ss_path = os.path.join(opt.data_path, 'v2.0_495m383D', 'M_GSM.txt')
+    m_ss_path = os.path.join(opt.data_path, 'v3.2_spld_paper', 'M_GSM.txt')
     if os.path.exists(m_ss_path):
         m_ss_data = read_txt(m_ss_path)
         dataset['m_ss'] = m_ss_data
@@ -412,7 +417,7 @@
 
     # 打印双视图配置摘要
     print("\n" + "=" * 60)
-    print("🎯 DUAL-VIEW CONFIGURATION SUMMARY (v2.0_495m383D)")
+    print("🎯 DUAL-VIEW CONFIGURATION SUMMARY (v3.2_spld_paper)")
     print("=" * 60)
     print("miRNA Dual Views:")
     print(f"  📊 View 1 (Sequence): M_GSM.txt {dataset['m_ss'].shape}")
@@ -421,7 +426,7 @@
     print(f"  📊 View 1 (Gene): D_SSM2.txt {dataset['d_gs'].shape}")
     print(f"  📊 View 2 (Semantic): D_SSM1.txt {dataset['dis_sem'].shape}")
     print(f"\nAssociation Matrix: {dataset['md_p'].shape}")
-    print(f"Data source: v2.0_495m383D (495 miRNAs × 383 Diseases)")
+    print(f"Data source: v3.2_spld_paper (411 miRNAs × 271 Diseases)")
     print("=" * 60)
 
     return dataset
```
