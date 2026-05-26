#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 一键为本项目配置专属 Python 虚拟隔离环境
# 用法: bash scripts/setup_venv.sh
# ──────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

echo "=== Beyesian 虚拟环境配置 ==="
echo "项目目录: $PROJECT_ROOT"
echo "Python 版本: $(python3 --version)"

# 1. 创建虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "虚拟环境已存在: $VENV_DIR"
    read -rp "是否删除并重建? [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        rm -rf "$VENV_DIR"
        echo "已删除旧环境"
    else
        echo "保留现有环境，仅更新依赖..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[1/2] 创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# 2. 安装依赖
echo ""
echo "[2/2] 安装依赖..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_ROOT/requirements.txt"
deactivate

echo ""
echo "──────────────────────────────────────────"
echo "  配置完成!"
echo ""
echo "  激活环境:"
echo "    source venv/bin/activate"
echo ""
echo "  激活后运行:"
echo "    python3 main.py --model asia"
echo "    python3 main.py --model alarm --pop-size 100 --generations 500"
echo "──────────────────────────────────────────"
