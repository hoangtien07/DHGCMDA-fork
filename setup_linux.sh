#!/usr/bin/env bash
# Dựng môi trường Python cho DHGCMDA trên Linux (Ubuntu) — dùng uv (user-level, không cần sudo).
# Tái lập venv giống hệt lần port sang Linux (branch linux-run).
#
# Cách dùng:
#     ./setup_linux.sh
#
# Sau khi xong:
#     source venv/bin/activate    # hoặc dùng trực tiếp venv/bin/python
#     python main_experiments_hetero1.py --device cpu --epoch 3 --validation 2   # smoke test
set -uo pipefail
cd "$(dirname "$0")"

# 1) Cài uv nếu chưa có (standalone, vào ~/.local/bin — không đụng hệ thống)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[setup] Cài uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "[setup] uv $(uv --version)"

# 2) CPython 3.12 do uv quản lý (KHÔNG dùng /usr/bin/python3.12 của Debian — bản đó
#    patch site.py làm venv không nạp site-packages, torch sẽ không import được).
uv python install 3.12

# 3) Tạo venv với python uv-managed
uv venv --python 3.12 --python-preference only-managed venv

# 4) Cài deps. torch CPU nằm ở index PyTorch; phần còn lại ở PyPI.
#    --index-strategy unsafe-best-match để uv xét cả 2 index (lấy torch==2.5.1+cpu).
uv pip install --python venv/bin/python \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    -r requirements_linux.txt

echo ""
echo "[setup] XONG. Kiểm tra:"
venv/bin/python -c "import torch, torch_geometric, docx, matplotlib; print('torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda', torch.cuda.is_available())"
echo "[setup] Chạy thử: ./run_ablations.sh --smoke"
