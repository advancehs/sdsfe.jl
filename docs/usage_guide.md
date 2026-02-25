# sdsfe 使用指南

> 空间随机前沿模型 (Spatial Stochastic Frontier) 全模型估计、反事实分析与 Henderson 45度诊断图

## 目录

- [环境准备](#环境准备)
- [数据加载与预处理](#数据加载与预处理)
- [模型对照表](#模型对照表)
- [1. sfmodel_fit — 模型估计](#1-sfmodel_fit--模型估计)
  - [1.1 KU 系列](#11-ku-系列--kutlu-2020)
  - [1.2 KK 系列](#12-kk-系列--kutlu-2017)
  - [1.3 OA 系列](#13-oa-系列--orea--álvarez-2019--oad-2024)
- [2. sfmodel_counterfactual — 反事实分析](#2-sfmodel_counterfactual--反事实分析)
  - [2.1 场景类型](#21-场景类型)
  - [2.2 KU 系列反事实](#22-ku-系列反事实)
  - [2.3 KK 系列反事实](#23-kk-系列反事实)
  - [2.4 OA 系列反事实](#24-oa-系列反事实)
  - [2.5 水平值分解 (C_level)](#25-水平值分解)
- [3. sfmodel_henderson45 — 非效率边际效应 Henderson 图](#3-sfmodel_henderson45--非效率边际效应-henderson-图)
  - [3.1 KU 系列 Henderson 图](#31-ku-系列-henderson-图)
  - [3.2 KK 系列 Henderson 图](#32-kk-系列-henderson-图)
  - [3.3 OA 系列 Henderson 图](#33-oa-系列-henderson-图)
  - [3.4 批量画图](#34-批量画图)
- [4. sfmodel_henderson45_y — 前沿边际效应 Henderson 图](#4-sfmodel_henderson45_y--前沿边际效应-henderson-图)
  - [4.1 KU 系列前沿 Henderson 图](#41-ku-系列前沿-henderson-图)
  - [4.2 KK 系列前沿 Henderson 图](#42-kk-系列前沿-henderson-图)
  - [4.3 OA 系列前沿 Henderson 图](#43-oa-系列前沿-henderson-图)

---

## 环境准备

```julia
using Pkg
Pkg.activate(@__DIR__)

using sdsfe
using CSV, DataFrames, Statistics, JLD2, LinearAlgebra
using Optim: LineSearches
```

## 数据加载与预处理

```julia
# 辅助函数: 行标准化空间权重矩阵
function normw(W)
    rowsums = sum(W, dims=2)
    rowsums[rowsums .== 0] .= 1
    W .= W ./ rowsums
    return W
end

# 加载数据并排序（必须按 [year, city_code] 排序！）
dat = CSV.read("your_data.csv", DataFrame)
dat[!, :_cons] .= 1.0
dat = sort(dat, [:year, :city_code])

# 空间权重矩阵
w_df = CSV.read("your_weight_matrix.csv", DataFrame)
w_mat = normw(Matrix(w_df)[:, 2:end])
Wx = Array{Matrix}(undef, 1, 1); Wx[1] = w_mat

# 构建 translog 变量 + 去均值化变量
# （参见 examples/sdsfe_usage_guide.jl 中的 translog2 函数）
```

## 模型对照表

| 模型ID | 面板函数 | 分布 | 内生性 | Wy | Wu | Wv | cfindices |
|--------|----------|------|--------|----|----|----|----|
| SSFKUH | `SSF_KU2020` | half | ✗ | ✓ | ✗ | ✗ | `Dict(1=>0.0)` |
| SSFKUT | `SSF_KU2020` | trun | ✗ | ✓ | ✗ | ✗ | `Dict(1=>0.0)` |
| SSFKUEH | `SSF_KUE2020` | half | ✓ | ✓ | ✗ | ✗ | `Dict(1=>0.0)` |
| SSFKUET | `SSF_KUE2020` | trun | ✓ | ✓ | ✗ | ✗ | `Dict(1=>0.0)` |
| SSFKKH | `SSF_KK2017` | half | ✗ | ✗ | ✗ | ✗ | `Dict(1=>6.3)` |
| SSFKKT | `SSF_KK2017` | trun | ✗ | ✗ | ✗ | ✗ | `Dict(1=>6.3)` |
| SSFKKEH | `SSF_KKE2017` | half | ✓ | ✗ | ✗ | ✗ | `Dict(1=>6.3)` |
| SSFKKET | `SSF_KKE2017` | trun | ✓ | ✗ | ✗ | ✗ | `Dict(1=>6.3)` |
| SSFOAH | `SSF_OA2019` | half | ✗ | ✓ | ✓ | ✗ | `Dict(1=>0.0)` |
| SSFOAT | `SSF_OA2019` | trun | ✗ | ✓ | ✓ | ✗ | `Dict(1=>0.0)` |
| SSFOADH | `SSF_OAD2024` | half | ✓ | ✓ | ✓ | ✓ | `Dict(1=>0.0)` |
| SSFOADT | `SSF_OAD2024` | trun | ✓ | ✓ | ✓ | ✓ | `Dict(1=>0.0)` |
| SSFWHH | `SSF_WH2010` | half | ✗ | ✗ | ✗ | ✗ | 暂不支持反事实 |
| SSFWHT | `SSF_WH2010` | trun | ✗ | ✗ | ✗ | ✗ | 暂不支持反事实 |
| SSFWHEH | `SSF_WHE2010` | half | ✓ | ✗ | ✗ | ✗ | 暂不支持反事实 |
| SSFWHET | `SSF_WHE2010` | trun | ✓ | ✗ | ✗ | ✗ | 暂不支持反事实 |

**关键区别:**
- **半正态 (half)**: 不需要 `@μ`
- **截断正态 (trun)**: 需要 `@μ(_cons)`
- **有内生性 (E)**: 需要 `@envar` + `@ivvar`，面板函数用 E 版本
- **KK 系列**: 无空间权重，优化器推荐 `NelderMead` + `finite`
- **KU 系列**: 含 Wy，优化器推荐 `BFGS` + `forward`
- **OA 系列**: 含 Wy+Wu(+Wv)，优化器推荐 `BFGS` + `forward`

---

## 1. sfmodel_fit — 模型估计

### 1.1 KU 系列 — Kutlu (2020)

特点: 含 Wy, 可含 Wx, 逐观测计算 JLMS。

**SSFKUH — 半正态 + 无内生性:**

```julia
sfmodel_spec(sfpanel(SSF_KU2020), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_kuh = sfmodel_fit(useData(Td))
```

**SSFKUEH — 半正态 + 有内生性:**

```julia
sfmodel_spec(sfpanel(SSF_KUE2020), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2),          # 内生变量
    @ivvar(ivkind22),      # 工具变量
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(1000),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward), margeffu(true),
    cfindices(Dict(1 => 0.0)))

res_kueh = sfmodel_fit(useData(Td))
```

**SSFKUT — 截断正态 + 无内生性:**

```julia
sfmodel_spec(sfpanel(SSF_KU2020), sftype(cost), sfdist(trun),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @μ(_cons),             # 截断正态需要 μ
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_kut = sfmodel_fit(useData(Td))
```

### 1.2 KK 系列 — Kutlu (2017)

特点: 无空间权重，按个体内积计算。优化器推荐 `NelderMead` + `finite`。

**SSFKKH — 半正态 + 无内生性:**

```julia
sfmodel_spec(sfpanel(SSF_KK2017), sftype(cost), sfdist(half),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(2000),
    main_solver(NelderMead()),
    main_maxIT(3000000),
    tolerance(1e-6), autodiff_mode(finite),
    cfindices(Dict(1 => 6.304568)))

res_kkh = sfmodel_fit(useData(Td))
```

**SSFKKEH — 半正态 + 有内生性:**

```julia
sfmodel_spec(sfpanel(SSF_KKE2017), sftype(cost), sfdist(half),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2), @ivvar(ivkind22),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(3000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 6.304568)))

res_kkeh = sfmodel_fit(useData(Td))
```

> 截断正态变体 (SSFKKT / SSFKKET): 将 `sfdist(half)` 改为 `sfdist(trun)` 并添加 `@μ(_cons)` 即可。

### 1.3 OA 系列 — Orea & Álvarez (2019) / OAD (2024)

特点: 含 Wy + Wu，可含 Wv，按时间段计算。OAD 支持内生性 + Wv。

**SSFOAH — 半正态 + 无内生性 (Wy+Wu):**

```julia
sfmodel_spec(sfpanel(SSF_OA2019), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx), wu(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_oah = sfmodel_fit(useData(Td))
```

**SSFOADH — 半正态 + 有内生性 (Wy+Wu+Wv):**

```julia
sfmodel_spec(sfpanel(SSF_OAD2024), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx), wu(Wx), wv(Wx),   # 完整四矩阵
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2), @ivvar(ivkind22),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(2000),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_oadh = sfmodel_fit(useData(Td))
```

> 截断正态变体 (SSFOAT / SSFOADT): 将 `sfdist(half)` 改为 `sfdist(trun)` 并添加 `@μ(_cons)` 即可。

---

## 2. sfmodel_counterfactual — 反事实分析

支持 KU(4) + KK(4) + OA(4) = 12 种模型。WH 系列暂不支持。

### 2.1 场景类型

`scenarios` 参数为 `Dict{String, Any}`，key 为 hscale 变量名，value 支持：

| 类型 | 示例 | 含义 |
|------|------|------|
| `Float64` | `"agg2" => 0.0` | 设为常数值 |
| `:quantile => p` | `"agg2" => :quantile => 0.75` | 设为第 p 百分位数 |
| `:shift => k` | `"agg2" => :shift => 1.0` | 原值 + k 个标准差 |
| `:multiply => k` | `"agg2" => :multiply => 0.5` | 原值 × k |
| `Symbol` | `"agg2" => :other_col` | 用另一列替换 |
| `Vector` | `"agg2" => my_vec` | 直接传入替换向量 |

### 2.2 KU 系列反事实

特点: 含 Wy，需传 `Wy_mat`。有内生性 (E) 版本需额外传 `envar` + `ivvar`。

**SSFKUH — 半正态 + 无内生性:**

```julia
cf_kuh = sfmodel_counterfactual(res_kuh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1])
```

**SSFKUT — 截断正态 + 无内生性:**

```julia
cf_kut = sfmodel_counterfactual(res_kut;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1])
```

**SSFKUEH — 半正态 + 有内生性:**

```julia
cf_kueh = sfmodel_counterfactual(res_kueh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")
```

**SSFKUET — 截断正态 + 有内生性:**

```julia
cf_kuet = sfmodel_counterfactual(res_kuet;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")
```

### 2.3 KK 系列反事实

特点: 无空间权重，不需要传 `Wy_mat`/`Wu_mat`。有内生性 (E) 版本需传 `envar` + `ivvar`。

**SSFKKH — 半正态 + 无内生性:**

```julia
cf_kkh = sfmodel_counterfactual(res_kkh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568))
```

**SSFKKT — 截断正态 + 无内生性:**

```julia
cf_kkt = sfmodel_counterfactual(res_kkt;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568))
```

**SSFKKEH — 半正态 + 有内生性:**

```julia
cf_kkeh = sfmodel_counterfactual(res_kkeh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568),
    envar="agg2", ivvar="ivkind22")
```

**SSFKKET — 截断正态 + 有内生性:**

```julia
cf_kket = sfmodel_counterfactual(res_kket;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568),
    envar="agg2", ivvar="ivkind22")
```

### 2.4 OA 系列反事实

特点: 含 Wy + Wu，需传 `Wy_mat` + `Wu_mat`。OAD 版本额外含 Wv + 内生性，需传 `Wv_mat` + `envar` + `ivvar`。

**SSFOAH — 半正态 + 无内生性 (Wy+Wu):**

```julia
cf_oah = sfmodel_counterfactual(res_oah;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1])
```

**SSFOAT — 截断正态 + 无内生性 (Wy+Wu):**

```julia
cf_oat = sfmodel_counterfactual(res_oat;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1])
```

**SSFOADH — 半正态 + 有内生性 (Wy+Wu+Wv):**

```julia
cf_oadh = sfmodel_counterfactual(res_oadh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1], Wv_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")
```

**SSFOADT — 截断正态 + 有内生性 (Wy+Wu+Wv):**

```julia
cf_oadt = sfmodel_counterfactual(res_oadt;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1], Wv_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")
```

### 2.5 水平值分解

传入 `C_level` 参数可获得水平值（非对数）的前沿/效率通道分解：

```julia
cf = sfmodel_counterfactual(res_kueh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22",
    C_level=Td.emission)       # 传入碳排放水平值（非对数）

# 返回值:
cf.delta_lnC_frontier    # 前沿通道 ΔlnC
cf.delta_lnC_efficiency  # 效率通道 ΔlnC
cf.delta_lnC_total       # 总 ΔlnC = frontier + efficiency
cf.ΔC_frontier           # 前沿通道 ΔC 水平值
cf.ΔC_efficiency         # 效率通道 ΔC 水平值
cf.ΔC_total              # 总 ΔC 水平值
cf.C_cf                  # 反事实碳排放 = C + ΔC_total
cf.te_cf_total           # 反事实 CEE (total)
cf.te_cf_direct          # 反事实 CEE (direct)
cf.te_cf_indirect        # 反事实 CEE (indirect)
```

---

## 3. sfmodel_henderson45 — 非效率边际效应 Henderson 图

Henderson & Parmeter (2012) 45度诊断图，用于检验 hscale 变量对 E(u) 的观测层面边际效应异质性。

**参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `res` | NamedTuple | `sfmodel_fit` 返回的结果 |
| `dat` | DataFrame | 按 `[year, city_code]` 排序后的数据 |
| `target_var` | String | hscale 中的目标变量名 |
| `Wy_mat` | Matrix | Wy 空间权重矩阵（KU/OA 需要） |
| `Wu_mat` | Matrix | Wu 空间权重矩阵（OA 需要） |
| `save_dir` | String | 图片保存目录 |
| `B` | Int | MC 模拟次数（默认 499） |
| `confidence_level` | Float64 | 置信水平（默认 0.95） |
| `dpi` | Int | 图片分辨率（默认 600） |

**返回值:** `.results`（Dict）、`.raw_margeff`、`.direct`/`.indirect`/`.total`、`.mc_direct`/`.mc_indirect`/`.mc_total`、`.config`

### 3.1 KU 系列 Henderson 图

特点: 含 Wy，需传 `Wy_mat`，输出 direct/indirect/total 三张图。

**SSFKUH — 半正态 + 无内生性:**

```julia
h45_kuh = sfmodel_henderson45(res_kuh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kuh", B=499, dpi=600)
```

**SSFKUT — 截断正态 + 无内生性:**

```julia
h45_kut = sfmodel_henderson45(res_kut;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kut", B=499)
```

**SSFKUEH — 半正态 + 有内生性:**

```julia
h45_kueh = sfmodel_henderson45(res_kueh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kueh", B=499)
```

**SSFKUET — 截断正态 + 有内生性:**

```julia
h45_kuet = sfmodel_henderson45(res_kuet;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kuet", B=499)
```

### 3.2 KK 系列 Henderson 图

特点: 无空间权重，不需要传空间矩阵，只输出一张 total 图。

**SSFKKH — 半正态 + 无内生性:**

```julia
h45_kkh = sfmodel_henderson45(res_kkh;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkh", B=499)
```

**SSFKKT — 截断正态 + 无内生性:**

```julia
h45_kkt = sfmodel_henderson45(res_kkt;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkt", B=499)
```

**SSFKKEH — 半正态 + 有内生性:**

```julia
h45_kkeh = sfmodel_henderson45(res_kkeh;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkeh", B=499)
```

**SSFKKET — 截断正态 + 有内生性:**

```julia
h45_kket = sfmodel_henderson45(res_kket;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kket", B=499)
```

### 3.3 OA 系列 Henderson 图

特点: 含 Wy + Wu，需传 `Wy_mat` + `Wu_mat`，输出 direct/indirect/total 三张图。Henderson 图不区分内生性，OA 和 OAD 调用方式相同。

**SSFOAH — 半正态 + 无内生性 (Wy+Wu):**

```julia
h45_oah = sfmodel_henderson45(res_oah;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oah", B=499)
```

**SSFOAT — 截断正态 + 无内生性 (Wy+Wu):**

```julia
h45_oat = sfmodel_henderson45(res_oat;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oat", B=499)
```

**SSFOADH — 半正态 + 有内生性 (Wy+Wu+Wv):**

```julia
h45_oadh = sfmodel_henderson45(res_oadh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oadh", B=499)
```

**SSFOADT — 截断正态 + 有内生性 (Wy+Wu+Wv):**

```julia
h45_oadt = sfmodel_henderson45(res_oadt;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oadt", B=499)
```

### 3.4 批量画图

```julia
for var in ["agg2", "indus2", "lagfdi2", "human2", "lnpgdp2", "roadpc2"]
    sfmodel_henderson45(res_kueh;
        dat=Td, target_var=var,
        Wy_mat=Wx[1],
        save_dir="result/henderson_45/kueh_$var", B=499)
end
```

---

## 4. sfmodel_henderson45_y — 前沿边际效应 Henderson 图

用于 translog 成本/生产函数中 ∂lnC/∂lnL、∂lnC/∂lnK 等观测层面前沿边际效应。

**核心参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `frontier_deriv` | `Dict{Symbol,Any}` | 每个前沿变量对目标的偏导数 |
| `wx_deriv` | `Dict{Symbol,Any}` | WX 变量的导数（key 为去掉 `W*` 前缀的变量名） |

`frontier_deriv` 值的类型：
- `Float64` → 常数导数（如线性项 = 1.0）
- `Symbol` → 导数等于该变量的观测值（如交叉项、平方项）
- `Vector` → 直接传入导数向量
- 未列出的变量 → 导数为 0

**Translog 导数参考:**

translog: `lnC = ... + β_L·lnL + β_LK·lnL·lnK + β_LY·lnL·lnY + 0.5·β_LL·lnL²`

| 目标 | frontier_deriv | 说明 |
|------|---------------|------|
| ∂lnC/∂lnL | `:lnl22=>1.0, :lnl_lnk22=>:lnk22, :lnl_lny22=>:lny22, :lnl2_05=>:lnl22` | 劳动弹性 |
| ∂lnC/∂lnK | `:lnk22=>1.0, :lnl_lnk22=>:lnl22, :lnk_lny22=>:lny22, :lnk2_05=>:lnk22` | 资本弹性 |
| ∂lnC/∂lnY | `:lny22=>1.0, :lnl_lny22=>:lnl22, :lnk_lny22=>:lnk22, :lny2_05=>:lny22` | 产出弹性 |
| ∂lnC/∂AGG | `:agg2=>1.0` + `wx_deriv=Dict(:agg2=>1.0)` | 含 WX 变量 |

### 4.1 KU 系列前沿 Henderson 图

特点: 含 Wy，输出 direct/indirect/total 三张图。含 `@frontierWx` 时需传 `wx_deriv`。

**SSFKUH — 半正态 + 无内生性 (∂lnC/∂lnL):**

```julia
sfmodel_henderson45_y(res_kuh; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,
        :lnl_lnk22 => :lnk22,
        :lnl_lny22 => :lny22,
        :lnl2_05   => :lnl22,
    ),
    save_dir="result/henderson_45_y/kuh_lnL", B=499)
```

**SSFKUT — 截断正态 + 无内生性 (∂lnC/∂lnK):**

```julia
sfmodel_henderson45_y(res_kut; dat=Td, target_var="lnK",
    frontier_deriv = Dict{Symbol,Any}(
        :lnk22     => 1.0,
        :lnl_lnk22 => :lnl22,
        :lnk_lny22 => :lny22,
        :lnk2_05   => :lnk22,
    ),
    save_dir="result/henderson_45_y/kut_lnK", B=499)
```

**SSFKUEH — 半正态 + 有内生性 (∂lnC/∂lnY):**

```julia
sfmodel_henderson45_y(res_kueh; dat=Td, target_var="lnY",
    frontier_deriv = Dict{Symbol,Any}(
        :lny22     => 1.0,
        :lnl_lny22 => :lnl22,
        :lnk_lny22 => :lnk22,
        :lny2_05   => :lny22,
    ),
    save_dir="result/henderson_45_y/kueh_lnY", B=499)
```

**SSFKUET — 截断正态 + 有内生性 (∂lnC/∂AGG, 含 WX):**

```julia
sfmodel_henderson45_y(res_kuet; dat=Td, target_var="AGG",
    frontier_deriv = Dict{Symbol,Any}(:agg2 => 1.0),
    wx_deriv = Dict{Symbol,Any}(:agg2 => 1.0),
    save_dir="result/henderson_45_y/kuet_AGG", B=499)
```

### 4.2 KK 系列前沿 Henderson 图

特点: 无空间权重，只输出一张 total 图。无 `@frontierWx`，不需要 `wx_deriv`。

**SSFKKH — 半正态 + 无内生性 (∂lnC/∂lnL):**

```julia
sfmodel_henderson45_y(res_kkh; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,
        :lnl_lnk22 => :lnk22,
        :lnl_lny22 => :lny22,
        :lnl2_05   => :lnl22,
    ),
    save_dir="result/henderson_45_y/kkh_lnL", B=499)
```

**SSFKKT — 截断正态 + 无内生性 (∂lnC/∂lnK):**

```julia
sfmodel_henderson45_y(res_kkt; dat=Td, target_var="lnK",
    frontier_deriv = Dict{Symbol,Any}(
        :lnk22     => 1.0,
        :lnl_lnk22 => :lnl22,
        :lnk_lny22 => :lny22,
        :lnk2_05   => :lnk22,
    ),
    save_dir="result/henderson_45_y/kkt_lnK", B=499)
```

**SSFKKEH — 半正态 + 有内生性 (∂lnC/∂lnY):**

```julia
sfmodel_henderson45_y(res_kkeh; dat=Td, target_var="lnY",
    frontier_deriv = Dict{Symbol,Any}(
        :lny22     => 1.0,
        :lnl_lny22 => :lnl22,
        :lnk_lny22 => :lnk22,
        :lny2_05   => :lny22,
    ),
    save_dir="result/henderson_45_y/kkeh_lnY", B=499)
```

**SSFKKET — 截断正态 + 有内生性 (∂lnC/∂lnL):**

```julia
sfmodel_henderson45_y(res_kket; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,
        :lnl_lnk22 => :lnk22,
        :lnl_lny22 => :lny22,
        :lnl2_05   => :lnl22,
    ),
    save_dir="result/henderson_45_y/kket_lnL", B=499)
```

### 4.3 OA 系列前沿 Henderson 图

特点: 含 Wy，输出 direct/indirect/total 三张图。含 `@frontierWx` 时需传 `wx_deriv`。Henderson 图不区分内生性，OA 和 OAD 调用方式相同。

**SSFOAH — 半正态 + 无内生性 (∂lnC/∂lnL):**

```julia
sfmodel_henderson45_y(res_oah; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,
        :lnl_lnk22 => :lnk22,
        :lnl_lny22 => :lny22,
        :lnl2_05   => :lnl22,
    ),
    save_dir="result/henderson_45_y/oah_lnL", B=499)
```

**SSFOAT — 截断正态 + 无内生性 (∂lnC/∂lnK):**

```julia
sfmodel_henderson45_y(res_oat; dat=Td, target_var="lnK",
    frontier_deriv = Dict{Symbol,Any}(
        :lnk22     => 1.0,
        :lnl_lnk22 => :lnl22,
        :lnk_lny22 => :lny22,
        :lnk2_05   => :lnk22,
    ),
    save_dir="result/henderson_45_y/oat_lnK", B=499)
```

**SSFOADH — 半正态 + 有内生性 (∂lnC/∂lnY):**

```julia
sfmodel_henderson45_y(res_oadh; dat=Td, target_var="lnY",
    frontier_deriv = Dict{Symbol,Any}(
        :lny22     => 1.0,
        :lnl_lny22 => :lnl22,
        :lnk_lny22 => :lnk22,
        :lny2_05   => :lny22,
    ),
    save_dir="result/henderson_45_y/oadh_lnY", B=499)
```

**SSFOADT — 截断正态 + 有内生性 (∂lnC/∂AGG, 含 WX):**

```julia
sfmodel_henderson45_y(res_oadt; dat=Td, target_var="AGG",
    frontier_deriv = Dict{Symbol,Any}(:agg2 => 1.0),
    wx_deriv = Dict{Symbol,Any}(:agg2 => 1.0),
    save_dir="result/henderson_45_y/oadt_AGG", B=499)
```
