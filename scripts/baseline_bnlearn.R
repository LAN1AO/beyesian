# baseline_bnlearn.R — 从 CSV 读取数据，跑 5 个结构学习算法，输出边表
# 用法: Rscript baseline_bnlearn.R <input.csv> <output.csv>
suppressPackageStartupMessages(library(bnlearn))

args <- commandArgs(trailingOnly = TRUE)
input_csv  <- args[1]
output_csv <- args[2]

df <- read.csv(input_csv)

# 转为 factor，确保每列至少 2 个 level (bnlearn 要求)
df[] <- lapply(df, function(col) {
  f <- as.factor(col)
  if (nlevels(f) < 2) {
    levels(f) <- c(levels(f), as.character(max(as.integer(levels(f))) + 1))
  }
  f
})

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
  net <- tryCatch(algos[[name]](df), error = function(e) NULL)
  if (is.null(net)) {
    # 算法本身失败，跳过
    results <- rbind(results, data.frame(
      algorithm = name, from = "", to = "", runtime_sec = 0,
      stringsAsFactors = FALSE))
    next
  }

  # 约束类算法返回 PDAG，尝试 cextend 转为 DAG
  net <- tryCatch(cextend(net), error = function(e) net)
  dt <- (proc.time() - t0)["elapsed"]

  edges <- arcs(net)
  # 过滤掉双向边 (PDAG 中无向边表示为双向)，仅保留单向
  if (nrow(edges) > 0) {
    edge_pairs <- paste(edges[, "from"], edges[, "to"], sep = "->")
    reverse_pairs <- paste(edges[, "to"], edges[, "from"], sep = "->")
    is_undirected <- edge_pairs %in% reverse_pairs
    edges <- edges[!is_undirected, , drop = FALSE]
  }

  if (nrow(edges) > 0) {
    block <- data.frame(
      algorithm   = name,
      from        = edges[, "from"],
      to          = edges[, "to"],
      runtime_sec = round(dt, 2),
      stringsAsFactors = FALSE
    )
  } else {
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
