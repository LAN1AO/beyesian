# baseline_bnlearn.R — 从 CSV 读取数据，跑 5 个结构学习算法，输出边表
# 用法: Rscript baseline_bnlearn.R <input.csv> <output.csv>
suppressPackageStartupMessages(library(bnlearn))

args <- commandArgs(trailingOnly = TRUE)
input_csv  <- args[1]
output_csv <- args[2]

df <- read.csv(input_csv)
df[] <- lapply(df, as.factor)

algos <- list(
  hc   = function(d) hc(d, score = "bic"),
  tabu = function(d) tabu(d, score = "bic"),
  mmhc = function(d) mmhc(d),
  pc   = function(d) pc.stable(d, test = "mi"),
  iamb = function(d) inter.iamb(d, test = "mi")
)

results <- data.frame()

for (name in names(algos)) {
  t0 <- proc.time()
  net <- algos[[name]](df)
  # 约束类算法可能返回 PDAG (含无向边)，cextend 扩展为完整 DAG
  net <- cextend(net)
  dt <- (proc.time() - t0)["elapsed"]

  edges <- arcs(net)
  if (nrow(edges) > 0) {
    block <- data.frame(
      algorithm   = name,
      from        = edges[, "from"],
      to          = edges[, "to"],
      runtime_sec = round(dt, 2),
      stringsAsFactors = FALSE
    )
  } else {
    # 零边时仍输出一行 (from/to 为空字符串)，保证 Python 端能读到此算法
    block <- data.frame(
      algorithm   = name,
      from        = "",
      to          = "",
      runtime_sec = round(dt, 2),
      stringsAsFactors = FALSE
    )
  }
  results <- rbind(results, block)
}

write.csv(results, output_csv, row.names = FALSE)
