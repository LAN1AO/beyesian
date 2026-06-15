#!/bin/bash
# 安装 R 运行时 + bnlearn 包 (Ubuntu/Debian)
# bnlearn 安装到项目本地 venv/R_libs，无需 sudo 权限
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
R_LIBS="$ROOT/venv/R_libs"

# R 运行时 (需要 sudo)
if ! command -v Rscript &> /dev/null; then
    echo "安装 R 运行时..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq r-base
fi

# bnlearn 安装到本地目录
mkdir -p "$R_LIBS"
echo "安装 bnlearn → $R_LIBS"
Rscript -e "install.packages('bnlearn', lib='$R_LIBS', repos='https://cloud.r-project.org/')"

echo "完成:"
Rscript -e ".libPaths('$R_LIBS'); cat('  bnlearn', as.character(packageVersion('bnlearn')), '\n')"
