#!/bin/bash
# 安装 R (≥4.4) + bnlearn 包 (Ubuntu/Debian)
# R 从 CRAN 官方源安装（系统源可能版本过低）
# bnlearn 安装到项目本地 venv/R_libs，无需 sudo
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
R_LIBS="$ROOT/venv/R_libs"

# 检查 R 版本是否 >= 4.4
install_r=false
if ! command -v Rscript &> /dev/null; then
    install_r=true
elif Rscript -e 'if (getRversion() < "4.4.0") quit(status=1)' 2>/dev/null; [ $? -ne 0 ]; then
    echo "当前 R 版本过低 ($(R --version | head -1)), 需要 >= 4.4.0"
    install_r=true
fi

if [ "$install_r" = true ]; then
    echo "从 CRAN 源安装 R..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common dirmngr
    # 添加 CRAN GPG key 和源
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
        sudo tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc > /dev/null
    sudo add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    sudo apt-get update -qq
    sudo apt-get install -y -qq r-base
fi

echo "R 版本: $(Rscript -e 'cat(as.character(getRversion()))')"

# bnlearn 安装到本地目录
mkdir -p "$R_LIBS"
echo "安装 bnlearn → $R_LIBS"
Rscript -e "install.packages('bnlearn', lib='$R_LIBS', repos='https://cloud.r-project.org/')"

echo "完成:"
Rscript -e ".libPaths('$R_LIBS'); cat('  bnlearn', as.character(packageVersion('bnlearn')), '\n')"
