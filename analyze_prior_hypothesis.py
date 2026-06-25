import sys; sys.path.insert(0, '/home/langxiao/beyesian')
import csv, pickle, os
from collections import defaultdict
import numpy as np

CSV = '/home/langxiao/beyesian/output/experiments/summary.csv'
PRIOR_DIR = '/home/langxiao/beyesian/data/priors'

# ---------- 读取先验元信息 ----------
prior_info = {}  # (network, prior) -> dict(E_prior, shd_from_gt, n_nodes)
for fn in os.listdir(PRIOR_DIR):
    if not fn.endswith('.pkl'):
        continue
    net, pri = fn[:-4].rsplit('_', 1)
    d = pickle.load(open(os.path.join(PRIOR_DIR, fn), 'rb'))
    g = d['graph']
    prior_info[(net, pri)] = {
        'E_prior': len(g.get_edges()),
        'shd_from_gt': d['shd_from_gt'],
        'n_nodes': g.n_nodes,
    }

# ---------- 读取实验结果 ----------
rows = list(csv.DictReader(open(CSV)))
# 按 (network, prior, n_samples) 聚合
grp = defaultdict(list)  # -> list of row dicts
for r in rows:
    key = (r['network'], r['prior'], int(r['n_samples']))
    grp[key].append(r)

def fmean(rs, col):
    return float(np.mean([float(x[col]) for x in rs]))

networks = sorted(set(r['network'] for r in rows))
prior_order_label = ['gt', 'mild', 'moderate', 'severe', 'random']

# ============================================================
# 分析 A: 结果是否靠近先验？  ratio = mean_sdiff / E_prior
# ============================================================
print('=' * 78)
print('分析 A — 结果相对先验的靠近度  ratio = mean_sdiff / E_prior（越小越贴先验）')
print('=' * 78)
A = defaultdict(dict)  # (net) -> {(prior,ns): ratio}
all_ratios = []
per_net_ratios = defaultdict(list)
for net in networks:
    nss = sorted(set(int(r['n_samples']) for r in rows if r['network'] == net))
    pis = prior_info  # alias
    print(f'\n[{net}]  n_nodes={prior_info[(net,"gt")]["n_nodes"]}')
    print(f'  {"prior":<9}{"n_samp":>7}{"E_prior":>8}{"mean_sdiff":>11}{"ratio":>8}{"mean_shd_skel":>14}{"mean_f1_skel":>13}')
    for pri in prior_order_label:
        for ns in nss:
            rs = grp.get((net, pri, ns))
            if not rs:
                continue
            ms = fmean(rs, 'sdiff')
            E = prior_info[(net, pri)]['E_prior']
            ratio = ms / E if E else float('nan')
            mshd = fmean(rs, 'shd_skel')
            mf1 = fmean(rs, 'f1_skel')
            all_ratios.append(ratio)
            per_net_ratios[net].append(ratio)
            A[net][(pri, ns)] = ratio
            print(f'  {pri:<9}{ns:>7}{E:>8}{ms:>11.2f}{ratio:>8.3f}{mshd:>14.2f}{mf1:>13.4f}')

print('\n--- A 汇总: 每网络 ratio (= mean_sdiff/E_prior) 的范围与中位 ---')
print(f'  {"network":<12}{"min":>7}{"median":>8}{"max":>7}{"<0.3占比":>10}')
for net in networks:
    v = np.array(per_net_ratios[net])
    print(f'  {net:<12}{v.min():>7.3f}{np.median(v):>8.3f}{v.max():>7.3f}{np.mean(v<0.3):>10.0%}')
allr = np.array(all_ratios)
print(f'\n  全局 ratio: min={allr.min():.3f} median={np.median(allr):.3f} '
      f'mean={allr.mean():.3f} max={allr.max():.3f}')
print(f'  ratio<0.10 的比例: {np.mean(allr<0.10):.0%}   '
      f'ratio<0.30 的比例: {np.mean(allr<0.30):.0%}   '
      f'ratio<0.50 的比例: {np.mean(allr<0.50):.0%}')

# ============================================================
# 分析 B: 结果质量是否随先验质量单调变化？
# ============================================================
def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # 处理并列：相同值取平均秩
    uniq = np.unique(a)
    for u in uniq:
        m = a == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float('nan')
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float('nan')

print('\n' + '=' * 78)
print('分析 B — 结果质量是否随先验质量(shd_from_gt)单调变差？')
print('  期望: 先验越差(shd_from_gt↑) → shd_skel↑(ρ>0) 且 f1_skel↓(ρ<0)')
print('=' * 78)

rho_shd_list, rho_f1_list = [], []
gt_vs_rand_shd = []  # (gt结果shd, random结果shd) 两端对比
gt_vs_rand_f1 = []
for net in networks:
    nss = sorted(set(int(r['n_samples']) for r in rows if r['network'] == net))
    print(f'\n[{net}]')
    for ns in nss:
        pts = []
        for pri in prior_order_label:
            rs = grp.get((net, pri, ns))
            if not rs:
                continue
            pts.append((pri,
                        prior_info[(net, pri)]['shd_from_gt'],
                        fmean(rs, 'shd_skel'),
                        fmean(rs, 'f1_skel')))
        # 按 shd_from_gt 排序
        pts.sort(key=lambda t: t[1])
        xs = [p[1] for p in pts]
        ys_shd = [p[2] for p in pts]
        ys_f1 = [p[3] for p in pts]
        rho_shd = spearman(xs, ys_shd)
        rho_f1 = spearman(xs, ys_f1)
        rho_shd_list.append(rho_shd); rho_f1_list.append(rho_f1)
        chain = '  '.join(f'{p[0]}(g{p[1]}):shd{p[2]:.1f}/f1{p[3]:.3f}' for p in pts)
        print(f'  n={ns:<6} ρ_shd={rho_shd:+.2f} ρ_f1={rho_f1:+.2f} | {chain}')
        # 两端对比
        d = dict((p[0], p) for p in pts)
        if 'gt' in d and 'random' in d:
            gt_vs_rand_shd.append((d['gt'][2], d['random'][2]))
            gt_vs_rand_f1.append((d['gt'][3], d['random'][3]))

rs_shd = np.array([r for r in rho_shd_list if not np.isnan(r)])
rs_f1 = np.array([r for r in rho_f1_list if not np.isnan(r)])
print('\n--- B 汇总: 各 (network,n_samples) 组的 Spearman ρ ---')
print(f'  ρ(shd_from_gt → shd_skel): mean={rs_shd.mean():+.2f} median={np.median(rs_shd):+.2f} '
      f'  正向(ρ>0)占比={np.mean(rs_shd>0):.0%}  强正向(ρ>0.5)占比={np.mean(rs_shd>0.5):.0%}')
print(f'  ρ(shd_from_gt → f1_skel ): mean={rs_f1.mean():+.2f} median={np.median(rs_f1):+.2f} '
      f'  负向(ρ<0)占比={np.mean(rs_f1<0):.0%}  强负向(ρ<-0.5)占比={np.mean(rs_f1<-0.5):.0%}')

# 两端对比：gt 先验 vs random 先验
gs = np.array(gt_vs_rand_shd); gf = np.array(gt_vs_rand_f1)
print('\n--- B 两端对比: gt 先验 vs random 先验 (跨所有 network×n_samples 组) ---')
print(f'  shd_skel:  gt先验 mean={gs[:,0].mean():.2f}  →  random先验 mean={gs[:,1].mean():.2f}  '
      f'(random更差占比={np.mean(gs[:,1]>gs[:,0]):.0%})')
print(f'  f1_skel :  gt先验 mean={gf[:,0].mean():.3f}  →  random先验 mean={gf[:,1].mean():.3f}  '
      f'(random更差占比={np.mean(gf[:,1]<gf[:,0]):.0%})')
