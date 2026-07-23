import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DHGCMDA Google Colab Runner\n",
    "\n",
    "Notebook này được thiết kế đặc biệt để chạy dự án DHGCMDA-fork của bạn trên **Google Colab**.\n",
    "\n",
    "Vì code gốc của bạn đang nằm ở máy local (hoặc Google Drive), cách tốt nhất để chạy trên Colab là mount Google Drive vào Colab, cài đặt các thư viện cần thiết, sau đó thực thi các script."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Mount Google Drive\n",
    "Hãy upload toàn bộ thư mục `DHGCMDA-fork` của bạn lên Google Drive. Sau đó chạy cell dưới đây để kết nối Colab với Google Drive của bạn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Di chuyển vào thư mục dự án\n",
    "Thay đổi đường dẫn `/content/drive/MyDrive/DHGCMDA-fork` thành đường dẫn thực tế nơi bạn lưu thư mục dự án trên Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "# Đổi đường dẫn này thành nơi chứa thư mục dự án của bạn trên Google Drive\n",
    "PROJECT_PATH = '/content/drive/MyDrive/DHGCMDA-fork'\n",
    "\n",
    "os.chdir(PROJECT_PATH)\n",
    "print(\"Thư mục làm việc hiện tại:\", os.getcwd())\n",
    "\n",
    "# Hiển thị danh sách file để kiểm tra xem đã vào đúng thư mục chưa\n",
    "!ls -la"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Cài đặt các thư viện cần thiết (Dependencies)\n",
    "Google Colab đã cài sẵn PyTorch, Numpy, Pandas và Scikit-Learn. Chúng ta chỉ cần cài đặt thêm `torch-geometric` và đảm bảo phiên bản PyTorch tương thích."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install torch-geometric\n",
    "!pip install python-docx  # Cần cho script tạo báo cáo\n",
    "\n",
    "import torch\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Chạy dự án (Cách 1: Chạy trực tiếp script bằng lệnh)\n",
    "Đây là cách đơn giản và ổn định nhất, mô phỏng lại y hệt cách bạn chạy trên máy local bằng file `.ps1` hay `.sh`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chạy file main_experiments với cấu hình tối ưu nhất (K=2, predictor=full_bilinear)\n",
    "!python main_experiments_hetero1.py \\\n",
    "    --device cuda \\\n",
    "    --dataset v2.0_495m383D \\\n",
    "    --K_neigs 2 \\\n",
    "    --predictor_mode full_bilinear \\\n",
    "    --epoch 650 \\\n",
    "    --exist_weight 0.1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Chạy dự án (Cách 2: Chạy Tương tác / Debug trực tiếp trong Colab)\n",
    "Nếu bạn muốn debug, in ra các tensor hay theo dõi loss từng bước thì chạy các cell dưới đây (tương tự như Interactive Runner ở local)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "if PROJECT_PATH not in sys.path:\n",
    "    sys.path.append(PROJECT_PATH)\n",
    "\n",
    "import warnings\n",
    "import torch.optim as optim\n",
    "import numpy as np\n",
    "import random\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "from param import parameter_parser\n",
    "from trainData import get_train_data\n",
    "from hetero_model import HeterogenousGraphCLAMIR\n",
    "from Calculate_Metrics import Metric_fun\n",
    "from main_experiments_hetero1 import (\n",
    "    SimplifiedMultiTypeAssociationLoss, \n",
    "    create_hetero_data_optimized, \n",
    "    constructHW_knn\n",
    ")\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Sử dụng thiết bị: {device}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sys.argv = [''] \n",
    "args = parameter_parser()\n",
    "args.device = str(device)\n",
    "\n",
    "# Cấu hình theo Plan M tối ưu\n",
    "args.K_neigs = [2]\n",
    "args.predictor_mode = 'full_bilinear'\n",
    "args.dataset = 'v2.0_495m383D'\n",
    "args.epoch = 50  # Rút ngắn để chạy thử\n",
    "\n",
    "print(\"Đang tải dữ liệu...\")\n",
    "train_data = get_train_data(args)\n",
    "dis_sim = train_data[0].float().to(device)\n",
    "mi_sim = train_data[1].float().to(device)\n",
    "association_matrix = train_data[4].float().to(device)\n",
    "\n",
    "model = HeterogenousGraphCLAMIR(args).to(device)\n",
    "optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)\n",
    "criterion = SimplifiedMultiTypeAssociationLoss(args, model).to(device)\n",
    "print(\"Mô hình đã khởi tạo!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "hetero_data = create_hetero_data_optimized(train_data)\n",
    "mi_H = constructHW_knn(train_data[1].cpu().numpy(), args.K_neigs, is_probH=False).to(device)\n",
    "dis_H = constructHW_knn(train_data[0].cpu().numpy(), args.K_neigs, is_probH=False).to(device)\n",
    "\n",
    "pos_indices = torch.nonzero(association_matrix > 0, as_tuple=False)\n",
    "neg_indices = torch.nonzero(association_matrix == 0, as_tuple=False)\n",
    "idx = torch.randperm(neg_indices.size(0))[:pos_indices.size(0)]\n",
    "sampled_neg_indices = neg_indices[idx]\n",
    "\n",
    "print(\"Bắt đầu huấn luyện...\")\n",
    "for epoch in range(1, args.epoch + 1):\n",
    "    model.train()\n",
    "    optimizer.zero_grad()\n",
    "    score, _, _, _ = model(hetero_data, mi_H, dis_H, mi_sim, dis_sim)\n",
    "    loss = criterion(pos_indices, sampled_neg_indices, score, association_matrix)\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    if epoch % 10 == 0 or epoch == 1:\n",
    "        print(f\"Epoch {epoch:03d}/{args.epoch:03d} - Loss: {loss.item():.4f}\")\n",
    "\n",
    "print(\"Hoàn tất huấn luyện!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("d:\\VsCode\\DHGCMDA-fork\\DHGCMDA_Colab_Runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
