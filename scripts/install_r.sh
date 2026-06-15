#!/bin/bash
# 安装 R 运行时 + bnlearn 包 (Ubuntu/Debian)
set -e
echo "安装 R 运行时..."
sudo apt-get update -qq
sudo apt-get install -y -qq r-base
echo "安装 bnlearn 包..."
Rscript -e 'install.packages("bnlearn", repos="https://cloud.r-project.org/")'
echo "完成: R $(R --version | head -1) + bnlearn"
Rscript -e 'cat("bnlearn", as.character(packageVersion("bnlearn")), "\n")'
