#!/usr/bin/env python3
"""凸/凹判定: (sdiff, mdl) Pareto 前沿能否被 weighted-sum 覆盖。
两目标均最小化。weighted-sum 仅能命中 lower convex hull 上的支持解;
凹(非支持)中间点会被整段跳过。"""
import csv, glob, os, collections

NODES = {"asia":8,"alarm":37,"hailfinder":56,"win95pts":76,"munin1":186,"andes":223}

def load_front(path):
    rows=[]
    for r in csv.DictReader(open(path)):
        try:
            rows.append((int(round(float(r["sdiff"]))), float(r["mdl"])))
        except (ValueError, KeyError):
            continue
    return rows

def dedup_sort(rows):
    """同一 sdiff 取最小 mdl(下包络), 按 sdiff 升序。"""
    best={}
    for s,m in rows:
        if s not in best or m<best[s]:
            best[s]=m
    return sorted(best.items())  # [(sdiff, mdl)] asc by sdiff

def cross(o,a,b):
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

def lower_convex_hull(pts, rel_eps=1e-9):
    """返回支持解(凸包顶点)的索引集合。
    min-min 递减前沿 => mdl(sdiff) 凸 <=> 全部点在下凸包上。
    pop 中间点当且仅当它严格落在邻接弦之上(凹/被跳过)。"""
    if len(pts)<=2:
        return set(range(len(pts)))
    # 尺度: 用于带容差的共线判定, 避免浮点噪声误判凹
    ys=[p[1] for p in pts]; xs=[p[0] for p in pts]
    scale=(max(ys)-min(ys))*(max(xs)-min(xs))
    tol=rel_eps*scale if scale>0 else 1e-6
    hull=[]  # 存(point, orig_index)
    for i,p in enumerate(pts):
        while len(hull)>=2 and cross(hull[-2][0],hull[-1][0],p) < -tol:
            # cross<0 => 中间点 hull[-1] 在 O->p 弦的上方(凹/非支持) => 移除
            # cross>0 => 凸(保留); |cross|<=tol => 共线, 弱支持(保留)
            hull.pop()
        hull.append((p,i))
    return {idx for _,idx in hull}

def vgap(pts, supported):
    """对每个非支持点, 计算它在 mdl 方向上高出下凸包多少(绝对 & 相对)。"""
    sup=sorted(supported)
    gaps=[]
    for i,(x,y) in enumerate(pts):
        if i in supported: continue
        # 找相邻的两个支持点把 x 夹住, 线性插值出包络 mdl
        left=max(j for j in sup if pts[j][0]<=x)
        right=min(j for j in sup if pts[j][0]>=x)
        (xl,yl),(xr,yr)=pts[left],pts[right]
        env = yl if xr==xl else yl+(yr-yl)*(x-xl)/(xr-xl)
        gaps.append((y-env, (y-env)/abs(env) if env else 0.0))
    return gaps

def main():
    files=sorted(glob.glob("/home/langxiao/beyesian/output/severe_random/*/run_42/pareto_front.csv"))
    print(f"{'combo':<28}{'nodes':>6}{'pts':>5}{'sup':>5}{'skip':>5}{'skip%':>7}  shape")
    print("-"*78)
    agg=[]
    for f in files:
        combo=f.split("/severe_random/")[1].split("/run_42")[0]
        net=combo.split("_")[0]; prior=combo.split("_")[1]
        raw=load_front(f)
        pts=dedup_sort(raw)
        n=len(pts)
        sup=lower_convex_hull(pts)
        nskip=n-len(sup)
        frac=nskip/n if n else 0
        shape = "CONVEX(全支持)" if nskip==0 else f"混合(凹点={nskip})"
        if n<=3: shape="点太少, 判定意义有限"
        gaps=vgap(pts,sup) if nskip else []
        maxrel=max((g[1] for g in gaps), default=0.0)
        print(f"{combo:<28}{NODES[net]:>6}{n:>5}{len(sup):>5}{nskip:>5}{frac*100:>6.1f}%  {shape}"
              + (f"  max凹深={maxrel*100:.3f}%mdl" if gaps else ""))
        agg.append(dict(combo=combo,net=net,prior=prior,nodes=NODES[net],
                        n=n,sup=len(sup),skip=nskip,frac=frac,maxrel=maxrel,small=NODES[net]<=76))

    print("\n=== 汇总 ===")
    tot_pts=sum(a["n"] for a in agg); tot_skip=sum(a["skip"] for a in agg)
    print(f"全部 {len(agg)} 个前沿: 总点数={tot_pts}, weighted-sum 跳过={tot_skip} "
          f"({tot_skip/tot_pts*100:.2f}%)")
    nconvex=sum(1 for a in agg if a["skip"]==0 and a["n"]>3)
    nmixed =sum(1 for a in agg if a["skip"]>0)
    nsmall =sum(1 for a in agg if a["n"]<=3)
    print(f"严格凸(全支持)前沿: {nconvex}  含凹点前沿: {nmixed}  点太少无法判定: {nsmall}")

    for label,keep in [("小图(<=76节点)",lambda a:a["small"]),("大图(>76节点)",lambda a:not a["small"]),
                       ("severe先验",lambda a:a["prior"]=="severe"),("random先验",lambda a:a["prior"]=="random")]:
        sub=[a for a in agg if keep(a)]
        p=sum(a["n"] for a in sub); s=sum(a["skip"] for a in sub)
        print(f"  {label:<16}: 前沿={len(sub):>2}  总点={p:>4}  跳过={s:>3} ({s/p*100:>5.2f}%)  "
              f"含凹前沿={sum(1 for a in sub if a['skip']>0)}/{len(sub)}")

if __name__=="__main__":
    main()
