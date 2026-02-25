#########################################################
#  sdsfe 全模型使用指南
#  包含: sfmodel_fit, sfmodel_counterfactual,
#        sfmodel_henderson45, sfmodel_henderson45_y
#  覆盖 16 种模型 (KU/KK/OA/WH × 半正态/截断 × 有无内生性)
#########################################################

# ============================================================
# 0. 环境准备与数据加载
# ============================================================

using Pkg
Pkg.activate(@__DIR__)

using sdsfe
using CSV, DataFrames, Statistics, JLD2, LinearAlgebra
using Optim: LineSearches

# --- 辅助函数 ---
function normw(W)
    rowsums = sum(W, dims=2)
    rowsums[rowsums .== 0] .= 1
    W .= W ./ rowsums
    return W
end

function translog2(xs, ys, c, normalize=0, T=nothing)
    lnxs = normalize != 0 ? log.(xs) .- repeat(mean(log.(xs), dims=1), size(xs, 1)) : log.(xs)
    lnys = normalize != 0 ? log.(ys) .- repeat(mean(log.(ys), dims=1), size(ys, 1)) : log.(ys)
    lnc  = normalize != 0 ? log.(c) .- mean(log.(c)) : log.(c)
    lnys = lnys .+ lnc
    lnxs = hcat(lnxs, lnys)
    xvar = lnxs
    for i in 1:size(lnxs,2)-1, j in i+1:size(lnxs,2)
        xvar = hcat(xvar, lnxs[:,i] .* lnxs[:,j])
    end
    for i in 1:size(lnxs,2)
        xvar = hcat(xvar, 0.5 .* lnxs[:,i] .* lnxs[:,i])
    end
    if T !== nothing
        t = normalize != 0 ? (T .- minimum(T) .+ 1) .- mean(T .- minimum(T) .+ 1) : (T .- minimum(T) .+ 1)
        xvar = hcat(xvar, t)
        for i in 1:size(lnxs,2)
            xvar = hcat(xvar, t .* lnxs[:,i])
        end
        xvar = hcat(xvar, 0.5 * t .* t)
    end
    return xvar
end

# --- 加载数据 ---
dat = CSV.read(raw"你的数据路径.csv", DataFrame)
dat[!, :_cons] .= 1.0
dat = sort(dat, [:year, :city_code])  # 必须排序！

# --- 空间权重矩阵 ---
w_df = CSV.read(raw"你的权重矩阵路径.csv", DataFrame)
w_mat = normw(Matrix(w_df)[:, 2:end])
Wx = Array{Matrix}(undef, 1, 1); Wx[1] = w_mat

# --- 构建 translog 变量 ---
xs = Matrix(dat[!, [:labor, :cap_stock]])
ys = Matrix(dat[!, [:gdp]])
c  = Matrix(dat[!, [:emission]])
trans = translog2(xs, ys, c, 1, dat[!, :year])
trans_df = DataFrame(trans, [:lnl22,:lnk22,:lny22,
    :lnl_lnk22,:lnl_lny22,:lnk_lny22,
    :lnl2_05,:lnk2_05,:lny2_05,:t,:t_lnl,:t_lnk,:t_lny,:t2_05])
dat = hcat(dat, trans_df)

# 去均值化变量
cols_to_dm = [:lnc,:agg,:human,:indus,:lagfdi,:roadpc,:lnpgdp,:ivkind2]
xs2 = Matrix(dat[!, cols_to_dm])
dm = xs2 .- mean(xs2, dims=1)
dm_names = Symbol.(string.(cols_to_dm) .* "2")
dat = hcat(dat, DataFrame(dm, dm_names), makeunique=true)

Td = dat  # 后续统一用 Td

# ============================================================
# 1. KU 系列 — Kutlu (2020) 空间随机前沿
#    特点: 含 Wy, 可含 Wx, 逐观测计算
#    面板函数: SSF_KU2020 (无内生性), SSF_KUE2020 (有内生性)
#    分布: half (半正态), trun (截断正态)
#    模型ID: SSFKUH, SSFKUT, SSFKUEH, SSFKUET
# ============================================================

# --- 1.1 SSFKUH: KU + 半正态 + 无内生性 ---
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

# --- 1.2 SSFKUT: KU + 截断正态 + 无内生性 ---
sfmodel_spec(sfpanel(SSF_KU2020), sftype(cost), sfdist(trun),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @μ(_cons),                    # 截断正态需要 μ
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_kut = sfmodel_fit(useData(Td))

# --- 1.3 SSFKUEH: KU + 半正态 + 有内生性 ---
sfmodel_spec(sfpanel(SSF_KUE2020), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2),                 # 内生变量
    @ivvar(ivkind22),             # 工具变量
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(1000),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward), margeffu(true),
    cfindices(Dict(1 => 0.0)))

res_kueh = sfmodel_fit(useData(Td))

# --- 1.4 SSFKUET: KU + 截断正态 + 有内生性 ---
sfmodel_spec(sfpanel(SSF_KUE2020), sftype(cost), sfdist(trun),
    wy(Wx), wx(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2), @ivvar(ivkind22),
    @μ(_cons),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(1000),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_kuet = sfmodel_fit(useData(Td))

# ============================================================
# 2. KK 系列 — Kutlu (2017) 随机前沿
#    特点: 无空间权重, 按个体内积计算
#    面板函数: SSF_KK2017 (无内生性), SSF_KKE2017 (有内生性)
#    分布: half (半正态), trun (截断正态)
#    模型ID: SSFKKH, SSFKKT, SSFKKEH, SSFKKET
# ============================================================

# --- 2.1 SSFKKH: KK + 半正态 + 无内生性 ---
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
    main_solver(NelderMead()),        # KK 用 NelderMead 更稳定
    main_maxIT(3000000),
    tolerance(1e-6), autodiff_mode(finite),
    cfindices(Dict(1 => 6.304568)))   # KK 的 cfindices 不同

res_kkh = sfmodel_fit(useData(Td))

# --- 2.2 SSFKKT: KK + 截断正态 + 无内生性 ---
sfmodel_spec(sfpanel(SSF_KK2017), sftype(cost), sfdist(trun),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @μ(_cons),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(2000),
    main_solver(NelderMead()),
    main_maxIT(3000000),
    tolerance(1e-6), autodiff_mode(finite),
    cfindices(Dict(1 => 6.304568)))

res_kkt = sfmodel_fit(useData(Td))

# --- 2.3 SSFKKEH: KK + 半正态 + 有内生性 ---
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

# --- 2.4 SSFKKET: KK + 截断正态 + 有内生性 ---
sfmodel_spec(sfpanel(SSF_KKE2017), sftype(cost), sfdist(trun),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2), @ivvar(ivkind22),
    @μ(_cons),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(3000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 6.304568)))

res_kket = sfmodel_fit(useData(Td))

# ============================================================
# 3. OA 系列 — Orea & Álvarez (2019) / OAD (2024) 空间随机前沿
#    特点: 含 Wy + Wu, 可含 Wv, 按时间段计算
#    面板函数: SSF_OA2019 (无内生性), SSF_OAD2024 (有内生性)
#    分布: half (半正态), trun (截断正态)
#    模型ID: SSFOAH, SSFOAT, SSFOADH, SSFOADT
#    注意: OA 系列支持 wu(), wv() 参数
# ============================================================

# --- 3.1 SSFOAH: OA + 半正态 + 无内生性 (Wy+Wu) ---
sfmodel_spec(sfpanel(SSF_OA2019), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx), wu(Wx),          # Wy + Wx + Wu
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

# --- 3.2 SSFOAT: OA + 截断正态 + 无内生性 (Wy+Wu) ---
sfmodel_spec(sfpanel(SSF_OA2019), sftype(cost), sfdist(trun),
    wy(Wx), wx(Wx), wu(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @μ(_cons),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(600),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_oat = sfmodel_fit(useData(Td))

# --- 3.3 SSFOADH: OAD + 半正态 + 有内生性 (Wy+Wu+Wv) ---
sfmodel_spec(sfpanel(SSF_OAD2024), sftype(cost), sfdist(half),
    wy(Wx), wx(Wx), wu(Wx), wv(Wx),  # 完整四矩阵
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

# --- 3.4 SSFOADT: OAD + 截断正态 + 有内生性 (Wy+Wu+Wv) ---
sfmodel_spec(sfpanel(SSF_OAD2024), sftype(cost), sfdist(trun),
    wy(Wx), wx(Wx), wu(Wx), wv(Wx),
    @timevar(tt), @idvar(id),
    @depvar(lnc2),
    @frontier(constant, lnl22, lnk22, lny22,
              lnl_lnk22, lnl_lny22, lnk_lny22,
              lnl2_05, lnk2_05, lny2_05, agg2),
    @frontierWx(agg2),
    @hscale(agg2, indus2, lagfdi2, human2, lnpgdp2, roadpc2),
    @envar(agg2), @ivvar(ivkind22),
    @μ(_cons),
    @σᵤ²(_cons), @σᵥ²(_cons), message=true)

sfmodel_opt(warmstart_solver(NelderMead()),
    warmstart_maxIT(2000),
    main_solver(BFGS(linesearch=LineSearches.BackTracking())),
    main_maxIT(1000),
    tolerance(1e-6), autodiff_mode(forward),
    cfindices(Dict(1 => 0.0)))

res_oadt = sfmodel_fit(useData(Td))

# ============================================================
# 4. sfmodel_counterfactual — 反事实分析
#    支持: KU(4) + KK(4) + OA(4) = 12 种模型
#    WH 系列暂不支持（原始 sdsfe 也未实现）
# ============================================================
#
# 场景类型 (scenarios 参数):
#   Float64          → 设为常数值, 如 0.0 表示均值
#   :quantile => 0.75 → 设为第75百分位数
#   :shift => 1.0    → 原值 + 1个标准差
#   :multiply => 0.5 → 原值 × 0.5
#   Symbol           → 用另一列替换, 如 :other_col
#   Vector           → 直接传入替换向量
#
# 返回值:
#   .te_cf_total / .te_cf_direct / .te_cf_indirect  — 反事实 CEE
#   .counterfacttotal / .counterfactdire / .counterfactindire — 反事实 JLMS
#   .delta_lnC_frontier / .delta_lnC_efficiency / .delta_lnC_total — 两通道分解
#   .ΔC_frontier / .ΔC_efficiency / .ΔC_total / .C_cf — 水平值分解(需传 C_level)

# --- 4.1 KU 系列反事实 (含 Wy) ---

# SSFKUH: 无内生性, 不需要 envar/ivvar
cf_kuh = sfmodel_counterfactual(res_kuh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1])                    # 显式传 Wy_mat

# SSFKUEH: 有内生性, 必须传 envar + ivvar
cf_kueh = sfmodel_counterfactual(res_kueh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")

# SSFKUT / SSFKUET: 同上, 截断正态自动识别
cf_kut = sfmodel_counterfactual(res_kut;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1])

cf_kuet = sfmodel_counterfactual(res_kuet;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")

# --- 4.2 KK 系列反事实 (无空间权重) ---

# SSFKKH: 不需要 Wy_mat/Wu_mat
cf_kkh = sfmodel_counterfactual(res_kkh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568))

# SSFKKEH: 有内生性
cf_kkeh = sfmodel_counterfactual(res_kkeh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568),
    envar="agg2", ivvar="ivkind22")

# SSFKKT / SSFKKET: 截断正态
cf_kkt = sfmodel_counterfactual(res_kkt;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568))

cf_kket = sfmodel_counterfactual(res_kket;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 6.304568),
    envar="agg2", ivvar="ivkind22")

# --- 4.3 OA 系列反事实 (含 Wy + Wu, 可含 Wv) ---

# SSFOAH: 无内生性, 传 Wy_mat + Wu_mat
cf_oah = sfmodel_counterfactual(res_oah;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1])

# SSFOAT: 截断正态
cf_oat = sfmodel_counterfactual(res_oat;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1])

# SSFOADH: 有内生性 + Wv, 传全部三个矩阵
cf_oadh = sfmodel_counterfactual(res_oadh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1], Wv_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")

# SSFOADT: 截断正态 + 内生性 + Wv
cf_oadt = sfmodel_counterfactual(res_oadt;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0),
    Wy_mat=Wx[1], Wu_mat=Wx[1], Wv_mat=Wx[1],
    envar="agg2", ivvar="ivkind22")

# --- 4.4 带 C_level 的水平值分解 ---
cf_with_level = sfmodel_counterfactual(res_kueh;
    dat=Td, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    Wy_mat=Wx[1],
    envar="agg2", ivvar="ivkind22",
    C_level=Td.emission)              # 传入碳排放水平值

# 访问水平值分解结果:
# cf_with_level.ΔC_frontier    — 前沿通道 ΔC
# cf_with_level.ΔC_efficiency  — 效率通道 ΔC
# cf_with_level.ΔC_total       — 总 ΔC
# cf_with_level.C_cf           — 反事实碳排放 = C + ΔC_total

# ============================================================
# 5. sfmodel_henderson45 — 非效率 E(u) 边际效应 Henderson 45度图
#    支持: KU(4) + KK(4) + OA(4) = 12 种模型
#    WH 系列暂不支持
#    自动检测模型类型, 自动进行空间分解 (direct/indirect/total)
# ============================================================
#
# 参数说明:
#   res          — sfmodel_fit 返回的结果
#   dat          — 按 [year, city_code] 排序后的 DataFrame
#   target_var   — hscale 中的目标变量名 (如 "agg2")
#   Wy_mat       — Wy 空间权重矩阵 (KU/OA 模型需要)
#   Wu_mat       — Wu 空间权重矩阵 (OA 模型需要)
#   save_dir     — 图片保存目录
#   B            — MC 模拟次数 (默认 499)
#   confidence_level — 置信水平 (默认 0.95)
#   dpi          — 图片分辨率 (默认 600)
#
# 返回值:
#   .results     — Dict, 含 "direct"/"indirect"/"total" 子结果
#   .raw_margeff — 原始边际效应向量
#   .direct / .indirect / .total — 空间分解后的边际效应
#   .mc_direct / .mc_indirect / .mc_total — MC 模拟矩阵
#   .config      — 模型配置信息

# --- 5.1 KU 系列 Henderson 45度图 ---

# SSFKUH / SSFKUEH: 含 Wy, 输出 direct/indirect/total 三张图
h45_kuh = sfmodel_henderson45(res_kuh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kuh", B=499, dpi=600)

h45_kueh = sfmodel_henderson45(res_kueh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kueh", B=499)

# SSFKUT / SSFKUET: 截断正态, 同样含 Wy
h45_kut = sfmodel_henderson45(res_kut;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kut", B=499)

h45_kuet = sfmodel_henderson45(res_kuet;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1],
    save_dir="result/henderson_45/kuet", B=499)

# --- 5.2 KK 系列 Henderson 45度图 ---

# SSFKKH / SSFKKEH: 无空间权重, 只输出一张 total 图
h45_kkh = sfmodel_henderson45(res_kkh;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkh", B=499)

h45_kkeh = sfmodel_henderson45(res_kkeh;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkeh", B=499)

# SSFKKT / SSFKKET: 截断正态
h45_kkt = sfmodel_henderson45(res_kkt;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kkt", B=499)

h45_kket = sfmodel_henderson45(res_kket;
    dat=Td, target_var="agg2",
    save_dir="result/henderson_45/kket", B=499)

# --- 5.3 OA 系列 Henderson 45度图 ---

# SSFOAH: 含 Wy + Wu
h45_oah = sfmodel_henderson45(res_oah;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oah", B=499)

# SSFOAT: 截断正态
h45_oat = sfmodel_henderson45(res_oat;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oat", B=499)

# SSFOADH / SSFOADT: 有内生性 (Henderson 图不区分内生性, 用法相同)
h45_oadh = sfmodel_henderson45(res_oadh;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oadh", B=499)

h45_oadt = sfmodel_henderson45(res_oadt;
    dat=Td, target_var="agg2",
    Wy_mat=Wx[1], Wu_mat=Wx[1],
    save_dir="result/henderson_45/oadt", B=499)

# --- 5.4 对多个 hscale 变量分别画图 ---
for var in ["agg2", "indus2", "lagfdi2", "human2", "lnpgdp2", "roadpc2"]
    sfmodel_henderson45(res_kueh;
        dat=Td, target_var=var,
        Wy_mat=Wx[1],
        save_dir="result/henderson_45/kueh_$var", B=499)
end

# ============================================================
# 6. sfmodel_henderson45_y — 前沿（因变量）边际效应 Henderson 45度图
#    用于 translog 成本/生产函数中 ∂lnC/∂lnL, ∂lnC/∂lnK 等
#    支持: 所有含 Wy 的模型 (KU/OA 系列)
#    KK 系列无空间权重, 只输出 total 图
# ============================================================
#
# 参数说明:
#   frontier_deriv — Dict{Symbol,Any}: 每个前沿变量对目标的偏导数
#     值为 Float64  → 常数导数 (如线性项 = 1.0)
#     值为 Symbol   → 导数等于该变量的观测值 (如交叉项、平方项)
#     值为 Vector   → 直接传入导数向量
#   wx_deriv       — Dict{Symbol,Any}: WX 变量的导数 (key 为去掉 "W*" 前缀的变量名)
#   Wy_mat         — Wy 空间权重矩阵

# --- 6.1 ∂lnC/∂lnL (劳动弹性) ---
# translog: lnC = ... + β_L·lnL + β_LK·lnL·lnK + β_LY·lnL·lnY + 0.5·β_LL·lnL²
# ∂lnC/∂lnL = β_L + β_LK·lnK + β_LY·lnY + β_LL·lnL
sfmodel_henderson45_y(res_kueh; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,       # ∂(lnL)/∂(lnL) = 1
        :lnl_lnk22 => :lnk22,   # ∂(lnL·lnK)/∂(lnL) = lnK
        :lnl_lny22 => :lny22,   # ∂(lnL·lnY)/∂(lnL) = lnY
        :lnl2_05   => :lnl22,   # ∂(0.5·lnL²)/∂(lnL) = lnL
    ),
    save_dir="result/henderson_45_y/lnL", B=499)

# --- 6.2 ∂lnC/∂lnK (资本弹性) ---
sfmodel_henderson45_y(res_kueh; dat=Td, target_var="lnK",
    frontier_deriv = Dict{Symbol,Any}(
        :lnk22     => 1.0,
        :lnl_lnk22 => :lnl22,   # ∂(lnL·lnK)/∂(lnK) = lnL
        :lnk_lny22 => :lny22,   # ∂(lnK·lnY)/∂(lnK) = lnY
        :lnk2_05   => :lnk22,   # ∂(0.5·lnK²)/∂(lnK) = lnK
    ),
    save_dir="result/henderson_45_y/lnK", B=499)

# --- 6.3 ∂lnC/∂lnY* (产出弹性) ---
sfmodel_henderson45_y(res_kueh; dat=Td, target_var="lnY",
    frontier_deriv = Dict{Symbol,Any}(
        :lny22     => 1.0,
        :lnl_lny22 => :lnl22,   # ∂(lnL·lnY)/∂(lnY) = lnL
        :lnk_lny22 => :lnk22,   # ∂(lnK·lnY)/∂(lnY) = lnK
        :lny2_05   => :lny22,   # ∂(0.5·lnY²)/∂(lnY) = lnY
    ),
    save_dir="result/henderson_45_y/lnY", B=499)

# --- 6.4 ∂lnC/∂AGG (含 WX 变量) ---
# 当 agg2 同时在 @frontier 和 @frontierWx 中时:
# ∂lnC/∂AGG = β_agg + θ_Wagg·W (需要 wx_deriv)
sfmodel_henderson45_y(res_kueh; dat=Td, target_var="AGG",
    frontier_deriv = Dict{Symbol,Any}(:agg2 => 1.0),
    wx_deriv = Dict{Symbol,Any}(:agg2 => 1.0),  # W*agg 对 agg 的导数 = 1
    save_dir="result/henderson_45_y/AGG", B=499)

# --- 6.5 KK 系列 (无空间权重, 只输出 total) ---
sfmodel_henderson45_y(res_kkeh; dat=Td, target_var="lnL",
    frontier_deriv = Dict{Symbol,Any}(
        :lnl22     => 1.0,
        :lnl_lnk22 => :lnk22,
        :lnl_lny22 => :lny22,
        :lnl2_05   => :lnl22,
    ),
    save_dir="result/henderson_45_y/kk_lnL", B=499)

# --- 6.6 OA 系列 (含 Wy, 输出 direct/indirect/total) ---
sfmodel_henderson45_y(res_oadh; dat=Td, target_var="lnK",
    frontier_deriv = Dict{Symbol,Any}(
        :lnk22     => 1.0,
        :lnl_lnk22 => :lnl22,
        :lnk_lny22 => :lny22,
        :lnk2_05   => :lnk22,
    ),
    save_dir="result/henderson_45_y/oa_lnK", B=499)

# ============================================================
# 7. 模型对照表 (快速参考)
# ============================================================
#
# ┌──────────┬──────────┬──────┬──────┬──────┬──────┬──────┬──────────┐
# │ 模型ID   │ 面板函数  │ 分布 │ 内生 │  Wy  │  Wu  │  Wv  │ cfindices│
# ├──────────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────────┤
# │ SSFKUH   │ SSF_KU2020│ half │  ✗  │  ✓  │  ✗  │  ✗  │ Dict(1=>0)│
# │ SSFKUT   │ SSF_KU2020│ trun │  ✗  │  ✓  │  ✗  │  ✗  │ Dict(1=>0)│
# │ SSFKUEH  │ SSF_KUE2020│half │  ✓  │  ✓  │  ✗  │  ✗  │ Dict(1=>0)│
# │ SSFKUET  │ SSF_KUE2020│trun │  ✓  │  ✓  │  ✗  │  ✗  │ Dict(1=>0)│
# ├──────────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────────┤
# │ SSFKKH   │ SSF_KK2017│ half │  ✗  │  ✗  │  ✗  │  ✗  │ Dict(1=>6.3)│
# │ SSFKKT   │ SSF_KK2017│ trun │  ✗  │  ✗  │  ✗  │  ✗  │ Dict(1=>6.3)│
# │ SSFKKEH  │ SSF_KKE2017│half │  ✓  │  ✗  │  ✗  │  ✗  │ Dict(1=>6.3)│
# │ SSFKKET  │ SSF_KKE2017│trun │  ✓  │  ✗  │  ✗  │  ✗  │ Dict(1=>6.3)│
# ├──────────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────────┤
# │ SSFOAH   │ SSF_OA2019│ half │  ✗  │  ✓  │  ✓  │  ✗  │ Dict(1=>0)│
# │ SSFOAT   │ SSF_OA2019│ trun │  ✗  │  ✓  │  ✓  │  ✗  │ Dict(1=>0)│
# │ SSFOADH  │ SSF_OAD2024│half │  ✓  │  ✓  │  ✓  │  ✓  │ Dict(1=>0)│
# │ SSFOADT  │ SSF_OAD2024│trun │  ✓  │  ✓  │  ✓  │  ✓  │ Dict(1=>0)│
# ├──────────┼──────────┼──────┼──────┼──────┼──────┼──────┼──────────┤
# │ SSFWHH   │ SSF_WH2010│ half │  ✗  │  ✗  │  ✗  │  ✗  │ (暂不支持)│
# │ SSFWHT   │ SSF_WH2010│ trun │  ✗  │  ✗  │  ✗  │  ✗  │ (暂不支持)│
# │ SSFWHEH  │ SSF_WHE2010│half │  ✓  │  ✗  │  ✗  │  ✗  │ (暂不支持)│
# │ SSFWHET  │ SSF_WHE2010│trun │  ✓  │  ✗  │  ✗  │  ✗  │ (暂不支持)│
# └──────────┴──────────┴──────┴──────┴──────┴──────┴──────┴──────────┘
#
# 关键区别:
#   半正态 (half): 不需要 @μ
#   截断正态 (trun): 需要 @μ(_cons)
#   有内生性 (E): 需要 @envar + @ivvar, 面板函数用 E 版本
#   KK 系列: 无空间权重, 优化器推荐 NelderMead + finite
#   KU 系列: 含 Wy, 优化器推荐 BFGS + forward
#   OA 系列: 含 Wy+Wu(+Wv), 优化器推荐 BFGS + forward
#
# sfmodel_counterfactual 调用要点:
#   KU: 传 Wy_mat; 有内生性传 envar+ivvar
#   KK: 不传空间矩阵; 有内生性传 envar+ivvar
#   OA: 传 Wy_mat+Wu_mat; OAD 额外传 Wv_mat+envar+ivvar
#
# sfmodel_henderson45 调用要点:
#   target_var 必须在 @hscale 变量列表中
#   KU: 传 Wy_mat
#   KK: 不传空间矩阵
#   OA: 传 Wy_mat + Wu_mat
#
# sfmodel_henderson45_y 调用要点:
#   frontier_deriv: 只填有导数的变量, 其余自动为 0
#   wx_deriv: 只在变量同时出现在 @frontier 和 @frontierWx 时需要

