#########################################################
#  Henderson & Parmeter (2012) 45度诊断图               #
#  集成到 sdsfe 包，适配全部 16 种模型类型              #
#########################################################

# ============================================================
# 1. E(u) 计算函数（供 ForwardDiff 自动微分）
# ============================================================

"""半正态, 无 Wu"""
function _h45_Eu_half(qw, nq, τ, δ)
    q = qw[1:nq]; w = qw[nq+1:end]
    h  = exp(dot(q, τ))
    σᵤ = exp(0.5 * dot(w, δ))
    Λ  = 0.0 / σᵤ
    return h * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
end

"""截断正态, 无 Wu"""
function _h45_Eu_trun(qwz, nq, nw, τ, δ, δz)
    q = qwz[1:nq]; w = qwz[nq+1:nq+nw]; z = qwz[nq+nw+1:end]
    h  = exp(dot(q, τ))
    σᵤ = exp(0.5 * dot(w, δ))
    μ  = dot(z, δz)
    Λ  = μ / σᵤ
    return h * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
end

"""半正态, 有 Wu"""
function _h45_Eu_half_wu(qw, nq, τ, δ, mii)
    q = qw[1:nq]; w = qw[nq+1:end]
    h  = exp(dot(q, τ))
    σᵤ = exp(0.5 * dot(w, δ))
    hs = mii * h
    Λ  = 0.0 / σᵤ
    return hs * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
end

"""截断正态, 有 Wu"""
function _h45_Eu_trun_wu(qwz, nq, nw, τ, δ, δz, mii)
    q = qwz[1:nq]; w = qwz[nq+1:nq+nw]; z = qwz[nq+nw+1:end]
    h  = exp(dot(q, τ))
    σᵤ = exp(0.5 * dot(w, δ))
    μ  = dot(z, δz)
    hs = mii * h
    Λ  = μ / σᵤ
    return hs * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
end

# ============================================================
# 2. 空间参数工具函数
# ============================================================

"""从变换后的空间参数还原原始参数"""
_h45_recover_raw(rho, rmin, rmax) = log((rho - rmin) / (rmax - rho))

"""从原始参数变换回实际空间参数"""
_h45_transform(g, rmin, rmax) = rmin / (1 + exp(g)) + rmax * exp(g) / (1 + exp(g))

"""计算特征值边界"""
function _h45_eigbounds(W)
    ev = real.(eigvals(W))
    return 1.0 / minimum(ev), 1.0  # rymax 硬编码为 1.0，与 sdsfe 一致
end

# ============================================================
# 3. 模型类型检测
# ============================================================

function _h45_detect(modelid)
    wh_set   = (SSFWHH, SSFWHT, SSFWHEH, SSFWHET)
    half_set = (SSFOAH, SSFOADH, SSFKUH, SSFKUEH, SSFKKH, SSFKKEH, SSFWHH, SSFWHEH)
    wu_set   = (SSFOAH, SSFOAT, SSFOADH, SSFOADT)
    wy_set   = (SSFOAH, SSFOAT, SSFOADH, SSFOADT, SSFKUH, SSFKUT, SSFKUEH, SSFKUET)
    is_half = modelid in half_set
    has_Wu  = modelid in wu_set
    has_Wy  = modelid in wy_set
    has_mu  = !is_half
    is_wh   = modelid in wh_set
    return (is_half=is_half, has_Wu=has_Wu, has_Wy=has_Wy, has_mu=has_mu, is_wh=is_wh)
end

# ============================================================
# 4. Henderson 45度图绘图函数
# ============================================================

"""
    henderson_45degree(ghat, ghat_se; save_path, parametric_mean, ...)

Henderson & Parmeter (2012) 45度诊断图。

# 参数
- `ghat::Vector`: 观测级别边际效应估计值
- `ghat_se::Vector`: 对应的标准误
- `save_path::String`: 图片保存路径
- `parametric_mean::Float64`: 参数化平均效应（参考线）
- `confidence_level::Float64=0.95`: 置信水平
- `title::String=""`: 图标题
- `dpi::Int=600`: 分辨率
"""
function henderson_45degree(
    ghat::Vector{Float64}, ghat_se::Vector{Float64};
    save_path::String,
    parametric_mean::Float64=mean(ghat),
    confidence_level::Float64=0.95,
    title::String="",
    show_insignificant::Bool=true,
    dpi::Int=600
)
    n = length(ghat)
    z_crit = confidence_level ≈ 0.95 ? 1.96 : (confidence_level ≈ 0.90 ? 1.645 : 1.96)
    ci_upper = ghat .+ z_crit .* ghat_se
    ci_lower = ghat .- z_crit .* ghat_se
    sig_idx   = findall(sign.(ci_lower) .== sign.(ci_upper))
    insig_idx = setdiff(1:n, sig_idx)
    n_sig = length(sig_idx)

    # 坐标范围
    all_vals = vcat(ghat, ci_upper, ci_lower)
    v_min, v_max = minimum(all_vals), maximum(all_vals)
    margin = (v_max - v_min) * 0.08
    lims = (v_min - margin, v_max + margin)

    gr()
    p = plot(size=(700,700), fontfamily="Times",
        xlabel="Marginal Effect Estimate", ylabel="Marginal Effect / CI Bounds",
        xlim=lims, ylim=lims, legend=:topleft,
        title=(title == "" ? "45° Diagnostic (Henderson & Parmeter 2012)" : title),
        dpi=dpi, grid=true, framestyle=:box, aspect_ratio=:equal)

    # 45度线 + 零线 + 参考线
    plot!(p, [lims[1],lims[2]], [lims[1],lims[2]], line=(:solid,2,:black), label="45° line", alpha=0.8)
    hline!(p, [0], line=(:dot,1,:gray60), label="", alpha=0.6)
    vline!(p, [0], line=(:dot,1,:gray60), label="", alpha=0.6)
    hline!(p, [parametric_mean], line=(:dash,2,:blue), label="", alpha=0.7)
    vline!(p, [parametric_mean], line=(:dash,2,:blue),
           label="Param mean=$(round(parametric_mean,digits=4))", alpha=0.7)

    # 不显著点
    if show_insignificant && !isempty(insig_idx)
        scatter!(p, ghat[insig_idx], ci_upper[insig_idx],
                marker=(:utriangle,3,:gray70,stroke(0)), label="", alpha=0.3)
        scatter!(p, ghat[insig_idx], ci_lower[insig_idx],
                marker=(:dtriangle,3,:gray70,stroke(0)), label="", alpha=0.3)
        scatter!(p, ghat[insig_idx], ghat[insig_idx],
                marker=(:circle,3,:gray70,stroke(0)), label="Insignificant ($(n-n_sig))", alpha=0.3)
    end
    # 显著点 CI 线段
    for i in sig_idx
        plot!(p, [ghat[i],ghat[i]], [ci_lower[i],ci_upper[i]],
              line=(:solid,0.8,:steelblue), label="", alpha=0.25)
    end
    # 显著点标记
    scatter!(p, ghat[sig_idx], ci_upper[sig_idx],
            marker=(:utriangle,4,:red,stroke(0,:red)), label="CI upper", alpha=0.6)
    scatter!(p, ghat[sig_idx], ci_lower[sig_idx],
            marker=(:dtriangle,4,:green,stroke(0,:green)), label="CI lower", alpha=0.6)
    scatter!(p, ghat[sig_idx], ghat[sig_idx],
            marker=(:circle,4,:black,stroke(0)), label="Estimate ($n_sig sig.)", alpha=0.8)

    mkpath(dirname(save_path))
    savefig(p, save_path)
    println("  图形已保存: $save_path")

    sig_pct = round(n_sig/n*100, digits=1)
    println("  均值=$(round(mean(ghat),digits=6)), 显著比例=$(sig_pct)%")
    return Dict(:n_obs=>n, :n_significant=>n_sig, :sig_pct=>sig_pct,
                :mean_effect=>mean(ghat), :parametric_mean=>parametric_mean, :plot=>p)
end

# ============================================================
# 5. 主入口函数 sfmodel_henderson45
# ============================================================

"""
    sfmodel_henderson45(res; dat, target_var, Wy_mat, Wu_mat, save_dir, B, ...)

Henderson & Parmeter (2012) 45度诊断图，适配 sdsfe 全部模型类型。
在 sfmodel_fit 之后调用，自动检测模型类型并计算空间分解。

# 参数
- `res`: sfmodel_fit 返回的结果
- `dat::DataFrame`: 按 [year, city_code] 排序后的 DataFrame
- `target_var::String`: 目标变量名（必须在 hscale 变量中）
- `Wy_mat=nothing`: Wy 空间权重矩阵（默认从模型规格中获取）
- `Wu_mat=nothing`: Wu 空间权重矩阵（默认从模型规格中获取）
- `save_dir::String`: 图片保存目录
- `B::Int=499`: MC 模拟次数
- `confidence_level::Float64=0.95`: 置信水平
- `dpi::Int=600`: 图片分辨率

# 示例
```julia
res = sfmodel_fit(useData(dat))
sfmodel_henderson45(res; dat=dat_sorted, target_var="agg2",
    save_dir="result/henderson_45", B=499)
```
"""
function sfmodel_henderson45(res;
    dat::DataFrame,
    target_var::String,
    Wy_mat=nothing, Wu_mat=nothing,
    save_dir::String="result/henderson_45",
    B::Int=499, confidence_level::Float64=0.95, dpi::Int=600)

    modelid = res[:modelid]
    cfg = _h45_detect(modelid)
    cfg.is_wh && error("WH 模型暂不支持 Henderson 45度图（原始 sdsfe 也未实现）")
    eqpo = res[:eqpo]
    coef = res[:coeff]
    vcov = res[:var_cov_mat]

    println(">>> Henderson 45度图 — 模型: $modelid")
    println("    半正态=$(cfg.is_half), Wu=$(cfg.has_Wu), Wy=$(cfg.has_Wy)")

    # --- 提取参数索引和系数 ---
    idx_q = eqpo[:coeff_log_hscale]
    idx_w = eqpo[:coeff_log_σᵤ²]
    τ_coef = coef[idx_q]
    δ_coef = coef[idx_w]
    nq = length(τ_coef); nw = length(δ_coef)

    idx_z = nothing; δz_coef = nothing; nz = 0
    if cfg.has_mu && haskey(eqpo, :coeff_μ)
        idx_z = eqpo[:coeff_μ]; δz_coef = coef[idx_z]; nz = length(δz_coef)
    end

    # --- 空间参数 ---
    rho_y = nothing; idx_gamma = nothing
    if cfg.has_Wy && haskey(eqpo, :coeff_γ)
        idx_gamma = eqpo[:coeff_γ]
        rho_y = coef[first(idx_gamma)]  # 已变换，直接使用
    end
    tau_u = nothing; idx_tau = nothing
    if cfg.has_Wu && haskey(eqpo, :coeff_τ)
        idx_tau = eqpo[:coeff_τ]
        tau_u = coef[first(idx_tau)]
    end

    # --- 获取空间权重矩阵 ---
    if Wy_mat === nothing && cfg.has_Wy
        Wy_raw = _dicM[:wy]
        Wy_mat = (Wy_raw !== nothing && Wy_raw !== Nothing) ? Wy_raw[1] : nothing
    end
    if Wu_mat === nothing && cfg.has_Wu
        Wu_raw = _dicM[:wu]
        Wu_mat = (Wu_raw !== nothing && Wu_raw !== Nothing) ? Wu_raw[1] : nothing
    end

    # --- 获取变量名并构建数据矩阵 ---
    hscale_names = Symbol.(res[:hscale])
    w_names = Symbol.(res[:σᵤ²])
    z_names = cfg.has_mu && haskey(res, :μ) && res[:μ] !== nothing ? Symbol.(res[:μ]) : Symbol[]

    Q_mat = Float64.(Matrix(dat[!, hscale_names]))
    W_mat = Float64.(Matrix(dat[!, w_names]))
    Z_mat = nz > 0 ? Float64.(Matrix(dat[!, z_names])) : nothing

    # --- 面板结构 ---
    idvar = Symbol(res[:idvar][1])
    city_codes = sort(unique(dat[!, idvar]))
    city_map = Dict(c => i for (i,c) in enumerate(city_codes))
    obs_ci = [city_map[c] for c in dat[!, idvar]]
    N_cities = length(city_codes)
    nobs = nrow(dat)

    # --- 目标变量位置 ---
    target_sym = Symbol(target_var)
    target_k = findfirst(==(target_sym), hscale_names)
    target_k === nothing && error("目标变量 $target_var 不在 hscale 变量中")
    println("    目标变量: $target_var (第 $target_k 个), 观测=$nobs, 城市=$N_cities")

    # --- 特征值边界 ---
    rymin = 0.0; rymax = 1.0; rumin = 0.0; rumax = 1.0
    if haskey(res, :eigvalu) && res[:eigvalu] !== nothing
        ev = res[:eigvalu]
        hasfield(typeof(ev), :rymin) && (rymin = ev.rymin)
        hasfield(typeof(ev), :rymax) && (rymax = ev.rymax)
        hasfield(typeof(ev), :rumin) && (rumin = ev.rumin)
        hasfield(typeof(ev), :rumax) && (rumax = ev.rumax)
    else
        if cfg.has_Wy && Wy_mat !== nothing
            rymin, rymax = _h45_eigbounds(Wy_mat)
        end
        if cfg.has_Wu && Wu_mat !== nothing
            rumin, rumax = _h45_eigbounds(Wu_mat)
        end
    end

    # --- 预计算 Mtau (OA/OAD 系列) ---
    Mtau = nothing
    if cfg.has_Wu && tau_u !== nothing && Wu_mat !== nothing
        Mtau = inv(I(N_cities) - tau_u * Wu_mat)
    end

    # --- 计算观测级别边际效应 ---
    println(">>> 计算观测级别边际效应...")
    raw_me = zeros(nobs)
    for i in 1:nobs
        ci = obs_ci[i]
        if cfg.has_mu && nz > 0
            qwz = vcat(Q_mat[i,:], W_mat[i,:], Z_mat[i,:])
            if Mtau !== nothing
                grad = ForwardDiff.gradient(v -> _h45_Eu_trun_wu(v, nq, nw, τ_coef, δ_coef, δz_coef, Mtau[ci]), qwz)
            else
                grad = ForwardDiff.gradient(v -> _h45_Eu_trun(v, nq, nw, τ_coef, δ_coef, δz_coef), qwz)
            end
        else
            qw = vcat(Q_mat[i,:], W_mat[i,:])
            if Mtau !== nothing
                grad = ForwardDiff.gradient(v -> _h45_Eu_half_wu(v, nq, τ_coef, δ_coef, Mtau[ci]), qw)
            else
                grad = ForwardDiff.gradient(v -> _h45_Eu_half(v, nq, τ_coef, δ_coef), qw)
            end
        end
        raw_me[i] = grad[target_k]
    end
    println("    原始边际效应均值: $(round(mean(raw_me), digits=6))")

    # --- 空间分解 ---
    has_spatial = cfg.has_Wy && rho_y !== nothing && Wy_mat !== nothing
    if has_spatial
        println(">>> 空间分解...")
        A = inv(I(N_cities) - rho_y * Wy_mat)
        A_diag = diag(A); A_colsum = vec(sum(A, dims=1))
        direct_me   = [A_diag[obs_ci[i]] * raw_me[i] for i in 1:nobs]
        total_me    = [A_colsum[obs_ci[i]] * raw_me[i] for i in 1:nobs]
        indirect_me = total_me .- direct_me
    else
        direct_me = raw_me; indirect_me = zeros(nobs); total_me = raw_me
    end

    # --- MC 模拟构建参数索引 ---
    println(">>> MC 模拟 (B=$B)...")
    Random.seed!(12345)
    mc_idx = Int[]; append!(mc_idx, collect(idx_q)); append!(mc_idx, collect(idx_w))
    off_τ = 1:nq; off_δ = (nq+1):(nq+nw)
    off_z = nothing
    if cfg.has_mu && idx_z !== nothing
        append!(mc_idx, collect(idx_z)); off_z = (nq+nw+1):(nq+nw+nz)
    end
    n_nonspatial = nq + nw + nz
    off_gamma = nothing; gammap_raw = nothing
    if cfg.has_Wy && idx_gamma !== nothing
        append!(mc_idx, collect(idx_gamma))
        off_gamma = n_nonspatial + 1
        gammap_raw = _h45_recover_raw(rho_y, rymin, rymax)
    end
    off_tau = nothing; taup_raw = nothing
    if cfg.has_Wu && idx_tau !== nothing
        append!(mc_idx, collect(idx_tau))
        off_tau = length(mc_idx)
        taup_raw = _h45_recover_raw(tau_u, rumin, rumax)
    end

    sub_vcov = vcov[mc_idx, mc_idx]
    L = cholesky(Symmetric(sub_vcov)).L
    mc_direct = zeros(nobs, B); mc_indirect = zeros(nobs, B); mc_total = zeros(nobs, B)

    # --- MC 模拟循环 ---
    for b in 1:B
        perturb = L * randn(size(L,1))
        τ_b = τ_coef .+ perturb[off_τ]
        δ_b = δ_coef .+ perturb[off_δ]
        δz_b = off_z !== nothing ? δz_coef .+ perturb[off_z] : nothing

        # 空间参数扰动
        A_diag_b = ones(N_cities); A_colsum_b = ones(N_cities)
        if has_spatial && off_gamma !== nothing
            g_b = gammap_raw + perturb[off_gamma]
            rho_b = _h45_transform(g_b, rymin, rymax)
            A_b = inv(I(N_cities) - rho_b * Wy_mat)
            A_diag_b = diag(A_b); A_colsum_b = vec(sum(A_b, dims=1))
        end
        Mtau_b = nothing
        if cfg.has_Wu && off_tau !== nothing && Wu_mat !== nothing
            t_b = taup_raw + perturb[off_tau]
            tau_b = _h45_transform(t_b, rumin, rumax)
            Mtau_b = inv(I(N_cities) - tau_b * Wu_mat)
        end

        for i in 1:nobs
            ci = obs_ci[i]
            if cfg.has_mu && nz > 0
                qwz = vcat(Q_mat[i,:], W_mat[i,:], Z_mat[i,:])
                if Mtau_b !== nothing
                    grad = ForwardDiff.gradient(v -> _h45_Eu_trun_wu(v, nq, nw, τ_b, δ_b, δz_b, Mtau_b[ci]), qwz)
                else
                    grad = ForwardDiff.gradient(v -> _h45_Eu_trun(v, nq, nw, τ_b, δ_b, δz_b), qwz)
                end
            else
                qw = vcat(Q_mat[i,:], W_mat[i,:])
                if Mtau_b !== nothing
                    grad = ForwardDiff.gradient(v -> _h45_Eu_half_wu(v, nq, τ_b, δ_b, Mtau_b[ci]), qw)
                else
                    grad = ForwardDiff.gradient(v -> _h45_Eu_half(v, nq, τ_b, δ_b), qw)
                end
            end
            me_raw = grad[target_k]
            if has_spatial
                mc_direct[i,b]   = A_diag_b[ci] * me_raw
                mc_total[i,b]    = A_colsum_b[ci] * me_raw
                mc_indirect[i,b] = mc_total[i,b] - mc_direct[i,b]
            else
                mc_direct[i,b] = me_raw; mc_indirect[i,b] = 0.0; mc_total[i,b] = me_raw
            end
        end
        b % 100 == 0 && println("  MC draw $b / $B")
    end
    println(">>> MC 模拟完成")

    # --- 生成 Henderson 45度图 ---
    println(">>> 生成 Henderson 45度图...")
    mkpath(save_dir)
    results = Dict{String, Any}()

    if has_spatial
        for (eff_name, eff_vec, mc_mat) in [
            ("direct", direct_me, mc_direct),
            ("indirect", indirect_me, mc_indirect),
            ("total", total_me, mc_total)]

            se_vec = vec(std(mc_mat, dims=2))
            path = joinpath(save_dir, "henderson_$(target_var)_$(eff_name).png")
            r = henderson_45degree(Float64.(eff_vec), se_vec;
                save_path=path, parametric_mean=mean(eff_vec),
                confidence_level=confidence_level, dpi=dpi,
                title="45° Diagnostic: $(uppercase(target_var)) $(titlecase(eff_name)) Effect on E(u)")
            results[eff_name] = r
        end
    else
        se_vec = vec(std(mc_direct, dims=2))
        path = joinpath(save_dir, "henderson_$(target_var).png")
        r = henderson_45degree(Float64.(raw_me), se_vec;
            save_path=path, parametric_mean=mean(raw_me),
            confidence_level=confidence_level, dpi=dpi,
            title="45° Diagnostic: $(uppercase(target_var)) Marginal Effect on E(u)")
        results["total"] = r
    end

    # --- 汇总 ---
    println("\n", "="^60)
    println("Henderson 45度图汇总 — $target_var 对 E(u) 的边际效应")
    println("模型: $modelid | 分布: $(cfg.is_half ? "半正态" : "截断正态")")
    println("空间: Wy=$(cfg.has_Wy), Wu=$(cfg.has_Wu)")
    println("="^60)

    return (results=results, raw_margeff=raw_me,
            direct=direct_me, indirect=indirect_me, total=total_me,
            mc_direct=mc_direct, mc_indirect=mc_indirect, mc_total=mc_total,
            config=cfg)
end

# ============================================================
# 6. 前沿（因变量）边际效应 Henderson 45度图
# ============================================================

"""
    sfmodel_henderson45_y(res; dat, target_var, frontier_deriv, ...)

Henderson & Parmeter (2012) 45度诊断图 — 前沿（因变量）边际效应。
用于 translog 成本/生产函数中 ∂lnC/∂lnL、∂lnC/∂lnK 等观测层面变化的边际效应。

# 示例
```julia
sfmodel_henderson45_y(res; dat=Td22, target_var="lnL",
    frontier_deriv = Dict(
        :lnl22     => 1.0,      # ∂lnl22/∂lnL = 1
        :lnl_lnk22 => :lnk22,   # ∂(lnL*lnK)/∂lnL = lnK
        :lnl_lny22 => :lny22,   # ∂(lnL*lnY)/∂lnL = lnY
        :lnl2_05   => :lnl22,   # ∂(0.5*lnL²)/∂lnL = lnL
    ),
    save_dir="result/henderson_45_y/lnL", B=499)
```

# 参数
- `res`: sfmodel_fit 返回的结果
- `dat::DataFrame`: 按 [year, city_code] 排序后的 DataFrame
- `target_var::String`: 目标变量名称（用于图标题）
- `frontier_deriv::Dict{Symbol,Any}`: 每个前沿变量对目标的偏导数
  - 值为 `Float64` 表示常数导数（如线性项 = 1.0）
  - 值为 `Symbol` 表示导数等于该变量的观测值（如交叉项、平方项）
  - 值为 `AbstractVector` 表示直接传入导数向量
- `wx_deriv::Dict{Symbol,Any}`: WX 变量的导数（格式同上，key 为去掉 "W*" 前缀的变量名）
- `Wy_mat`, `save_dir`, `B`, `confidence_level`, `dpi`: 同 sfmodel_henderson45
"""
function sfmodel_henderson45_y(res;
    dat::DataFrame,
    target_var::String,
    frontier_deriv::Dict{Symbol, <:Any},
    wx_deriv::Dict{Symbol, <:Any}=Dict{Symbol, Any}(),
    Wy_mat=nothing,
    save_dir::String="result/henderson_45_y",
    B::Int=499, confidence_level::Float64=0.95, dpi::Int=600)

    modelid = res[:modelid]
    cfg = _h45_detect(modelid)
    cfg.is_wh && error("WH 模型暂不支持 Henderson 45度图（原始 sdsfe 也未实现）")
    coef = res[:coeff]
    vcov = res[:var_cov_mat]
    eqpo = res[:eqpo]

    println(">>> Henderson 45度图 (前沿) — 模型: $modelid, 目标: $target_var")

    # --- 前沿系数和变量名 ---
    idx_frontier = eqpo[:coeff_frontier]
    β_frontier = coef[idx_frontier]
    nf = length(idx_frontier)  # 总前沿系数数（含 WX）

    # 构建完整的前沿变量名列表（res[:frontier] 不含 WX 变量名）
    frontier_base_names = Symbol.(res[:frontier])
    n_base = length(frontier_base_names)
    n_wx_coefs = nf - n_base

    frontier_names = Vector{Symbol}(undef, nf)
    frontier_names[1:n_base] .= frontier_base_names
    # 从 table_show 中提取 WX 变量名（以 "W*" 前缀存储）
    if n_wx_coefs > 0
        ts = res[:table_show]
        wx_count = 0
        for r in 1:size(ts, 1)
            nm_str = string(ts[r, 2])
            if startswith(nm_str, "W*")
                wx_count += 1
                if wx_count <= n_wx_coefs
                    frontier_names[n_base + wx_count] = Symbol(nm_str)
                end
            end
        end
    end

    # --- 识别 WX 和非 WX 变量 ---
    wx_names = Symbol[]
    wx_indices_in_frontier = Int[]
    non_wx_names = Symbol[]
    non_wx_indices = Int[]
    for (k, nm) in enumerate(frontier_names)
        s = string(nm)
        if startswith(s, "W*") || startswith(s, "Wx")
            push!(wx_names, nm)
            push!(wx_indices_in_frontier, k)
        else
            push!(non_wx_names, nm)
            push!(non_wx_indices, k)
        end
    end
    println("    前沿变量($n_base): $non_wx_names")
    println("    WX变量($n_wx_coefs): $wx_names")

    # --- 面板结构 ---
    idvar = Symbol(res[:idvar][1])
    city_codes = sort(unique(dat[!, idvar]))
    city_map = Dict(c => i for (i,c) in enumerate(city_codes))
    obs_ci = [city_map[c] for c in dat[!, idvar]]
    N_cities = length(city_codes)
    nobs = nrow(dat)

    # --- 获取空间权重矩阵 ---
    if Wy_mat === nothing && cfg.has_Wy
        Wy_raw = _dicM[:wy]
        Wy_mat = (Wy_raw !== nothing && Wy_raw !== Nothing) ? Wy_raw[1] : nothing
    end
    # --- 空间参数 ---
    rho_y = nothing; idx_gamma = nothing
    if cfg.has_Wy && haskey(eqpo, :coeff_γ)
        idx_gamma = eqpo[:coeff_γ]
        rho_y = coef[first(idx_gamma)]
    end

    # --- 特征值边界 ---
    rymin = 0.0; rymax = 1.0
    if haskey(res, :eigvalu) && res[:eigvalu] !== nothing
        ev = res[:eigvalu]
        hasfield(typeof(ev), :rymin) && (rymin = ev.rymin)
        hasfield(typeof(ev), :rymax) && (rymax = ev.rymax)
    elseif cfg.has_Wy && Wy_mat !== nothing
        rymin, rymax = _h45_eigbounds(Wy_mat)
    end

    # --- 构建导数向量 ---
    # frontier_deriv: Dict(:lnl22 => 1.0, :lnl_lnk22 => :lnk22, ...)
    # 对每个前沿变量，构建 nobs 长度的导数向量
    deriv_mat = zeros(nobs, nf)  # deriv_mat[i, k] = ∂x_k/∂z for obs i
    for (k, nm) in enumerate(frontier_names)
        s = string(nm)
        # 检查是否是 WX 变量，去掉 "W*" 前缀后查找
        base_nm = startswith(s, "W*") ? Symbol(s[3:end]) : nm
        d = nothing
        if haskey(frontier_deriv, nm)
            d = frontier_deriv[nm]
        elseif startswith(s, "W*") && haskey(wx_deriv, base_nm)
            d = wx_deriv[base_nm]
        end
        if d !== nothing
            if d isa Real
                deriv_mat[:, k] .= Float64(d)
            elseif d isa Symbol
                deriv_mat[:, k] .= Float64.(dat[!, d])
            elseif d isa AbstractVector
                deriv_mat[:, k] .= Float64.(d)
            end
        end
    end

    # --- 计算观测级别原始边际效应 ---
    println(">>> 计算观测级别前沿边际效应...")
    raw_me = zeros(nobs)
    for i in 1:nobs
        raw_me[i] = sum(β_frontier[k] * deriv_mat[i, k] for k in 1:nf)
    end
    println("    原始边际效应均值: $(round(mean(raw_me), digits=6))")
    # --- 空间分解 ---
    has_spatial = cfg.has_Wy && rho_y !== nothing && Wy_mat !== nothing
    if has_spatial
        println(">>> 空间分解...")
        A = inv(I(N_cities) - rho_y * Wy_mat)
        AW = A * Wy_mat
        A_diag = diag(A); A_colsum = vec(sum(A, dims=1))
        AW_diag = diag(AW); AW_colsum = vec(sum(AW, dims=1))

        # 分离非WX和WX部分的边际效应
        raw_me_nonwx = zeros(nobs)
        raw_me_wx = zeros(nobs)
        for i in 1:nobs
            for k in 1:nf
                if k in wx_indices_in_frontier
                    raw_me_wx[i] += β_frontier[k] * deriv_mat[i, k]
                else
                    raw_me_nonwx[i] += β_frontier[k] * deriv_mat[i, k]
                end
            end
        end

        direct_me  = [A_diag[obs_ci[i]] * raw_me_nonwx[i] + AW_diag[obs_ci[i]] * raw_me_wx[i] for i in 1:nobs]
        total_me   = [A_colsum[obs_ci[i]] * raw_me_nonwx[i] + AW_colsum[obs_ci[i]] * raw_me_wx[i] for i in 1:nobs]
        indirect_me = total_me .- direct_me
    else
        direct_me = raw_me; indirect_me = zeros(nobs); total_me = raw_me
    end

    # --- MC 模拟 ---
    println(">>> MC 模拟 (B=$B)...")
    Random.seed!(12345)
    mc_idx = collect(idx_frontier)
    off_beta = 1:nf
    n_mc = nf
    off_gamma = nothing; gammap_raw = nothing
    if has_spatial && idx_gamma !== nothing
        append!(mc_idx, collect(idx_gamma))
        off_gamma = n_mc + 1
        gammap_raw = _h45_recover_raw(rho_y, rymin, rymax)
        n_mc += 1
    end

    sub_vcov = vcov[mc_idx, mc_idx]
    L = cholesky(Symmetric(sub_vcov)).L
    mc_direct = zeros(nobs, B); mc_indirect = zeros(nobs, B); mc_total = zeros(nobs, B)
    for b in 1:B
        perturb = L * randn(size(L,1))
        β_b = β_frontier .+ perturb[off_beta]

        # 空间参数扰动
        A_diag_b = ones(N_cities); A_colsum_b = ones(N_cities)
        AW_diag_b = zeros(N_cities); AW_colsum_b = zeros(N_cities)
        if has_spatial && off_gamma !== nothing
            g_b = gammap_raw + perturb[off_gamma]
            rho_b = _h45_transform(g_b, rymin, rymax)
            A_b = inv(I(N_cities) - rho_b * Wy_mat)
            AW_b = A_b * Wy_mat
            A_diag_b = diag(A_b); A_colsum_b = vec(sum(A_b, dims=1))
            AW_diag_b = diag(AW_b); AW_colsum_b = vec(sum(AW_b, dims=1))
        end

        for i in 1:nobs
            ci = obs_ci[i]
            me_nonwx = 0.0; me_wx = 0.0
            for k in 1:nf
                if k in wx_indices_in_frontier
                    me_wx += β_b[k] * deriv_mat[i, k]
                else
                    me_nonwx += β_b[k] * deriv_mat[i, k]
                end
            end
            if has_spatial
                mc_direct[i,b]   = A_diag_b[ci] * me_nonwx + AW_diag_b[ci] * me_wx
                mc_total[i,b]    = A_colsum_b[ci] * me_nonwx + AW_colsum_b[ci] * me_wx
                mc_indirect[i,b] = mc_total[i,b] - mc_direct[i,b]
            else
                mc_direct[i,b] = me_nonwx + me_wx
                mc_total[i,b]  = me_nonwx + me_wx
            end
        end
        b % 100 == 0 && println("  MC draw $b / $B")
    end
    println(">>> MC 模拟完成")
    # --- 生成 Henderson 45度图 ---
    println(">>> 生成 Henderson 45度图...")
    mkpath(save_dir)
    results = Dict{String, Any}()

    if has_spatial
        for (eff_name, eff_vec, mc_mat) in [
            ("direct", direct_me, mc_direct),
            ("indirect", indirect_me, mc_indirect),
            ("total", total_me, mc_total)]

            se_vec = vec(std(mc_mat, dims=2))
            path = joinpath(save_dir, "henderson_y_$(target_var)_$(eff_name).png")
            r = henderson_45degree(Float64.(eff_vec), se_vec;
                save_path=path, parametric_mean=mean(eff_vec),
                confidence_level=confidence_level, dpi=dpi,
                title="45° Diagnostic: ∂lnC/∂$(target_var) $(titlecase(eff_name)) Effect")
            results[eff_name] = r
        end
    else
        se_vec = vec(std(mc_direct, dims=2))
        path = joinpath(save_dir, "henderson_y_$(target_var).png")
        r = henderson_45degree(Float64.(raw_me), se_vec;
            save_path=path, parametric_mean=mean(raw_me),
            confidence_level=confidence_level, dpi=dpi,
            title="45° Diagnostic: ∂lnC/∂$(target_var) Marginal Effect")
        results["total"] = r
    end

    println("\n", "="^60)
    println("Henderson 45度图汇总 — ∂lnC/∂$(target_var) 前沿边际效应")
    println("模型: $modelid | 空间: Wy=$(cfg.has_Wy)")
    println("direct均值: $(round(mean(direct_me), digits=6))")
    println("indirect均值: $(round(mean(indirect_me), digits=6))")
    println("total均值: $(round(mean(total_me), digits=6))")
    println("="^60)

    return (results=results, raw_margeff=raw_me,
            direct=direct_me, indirect=indirect_me, total=total_me,
            mc_direct=mc_direct, mc_indirect=mc_indirect, mc_total=mc_total)
end

# ============================================================
# 7. 反事实分析（支持变量名 + 多场景）
# ============================================================

"""
    sfmodel_counterfactual(res; dat, depvar, scenarios, Wy_mat, Wu_mat, Wv_mat, envar, ivvar, C_level)

反事实分析：支持按变量名指定、多种场景，适配 KU/KK/OA 全部模型类型。
WH 模型暂不支持（原始 sdsfe 也未实现）。

自动进行两通道分解：当反事实变量同时出现在 @frontier 和 @hscale 中时，
返回前沿通道 (delta_lnC_frontier) 和效率通道 (delta_lnC_efficiency) 的分解结果。
若提供 C_level，还会计算水平值的 ΔC 分解。

# 支持的模型
- KU 系列 (SSFKUH, SSFKUT, SSFKUEH, SSFKUET): 逐观测计算，含 Wy
- KK 系列 (SSFKKH, SSFKKT, SSFKKEH, SSFKKET): 按个体内积计算，无空间权重
- OA 系列 (SSFOAT, SSFOAH, SSFOADT, SSFOADH): 按时间段计算，含 Wu/Wy

# 参数
- `res`: sfmodel_fit 返回的结果
- `dat::DataFrame`: 按 [year, city_code] 排序后的 DataFrame
- `depvar::Union{String,Symbol}`: 因变量名（必须提供，如 "lnc2"）
- `scenarios::Dict{String, Any}`: 反事实场景，key 为 hscale 变量名，value 为：
  - `Float64`: 将该变量设为常数值（如 0.0 表示均值）
  - `Symbol`: 使用 dat 中另一列的值替换
  - `Vector`: 直接传入替换向量
  - `Pair{Symbol,Float64}`: 如 `:quantile => 0.25` 表示设为第25百分位数
  - `Pair{Symbol,Float64}`: 如 `:shift => 1.0` 表示在原值基础上加一个标准差
- `Wy_mat`: Wy 空间权重矩阵（KU/OA 模型，默认从模型内部获取）
- `Wu_mat`: Wu 空间权重矩阵（OA 模型，默认从模型内部获取）
- `Wv_mat`: Wv 空间权重矩阵（OA 模型，预留）
- `envar`: 内生变量名（如 "agg2"），有内生性模型必须提供
- `ivvar`: 工具变量名（如 "ivkind22" 或 ["ivkind22"]），有内生性模型必须提供
- `C_level::Union{Nothing, AbstractVector}=nothing`: 碳排放水平值向量（注意：传入水平值，
  非对数值，如 `dat.emission`）。提供后自动计算 ΔC 前沿/效率/总计分解（单位与传入值一致）。

# 返回值（新增字段）
- `delta_lnC_frontier`: 前沿通道的 ΔlnC（对数变化）
- `delta_lnC_efficiency`: 效率通道的 ΔlnC（对数变化）
- `delta_lnC_total`: 总 ΔlnC = frontier + efficiency
- `ΔC_frontier`: 前沿通道的 ΔC 水平值（仅当提供 C_level 时）
- `ΔC_efficiency`: 效率通道的 ΔC 水平值（仅当提供 C_level 时）
- `ΔC_total`: 总 ΔC 水平值（仅当提供 C_level 时）
- `C_cf`: 反事实碳排放水平值 = C_level + ΔC_total（仅当提供 C_level 时）

# 示例
```julia
# KU 无内生性模型（仅对数分解）
r1 = sfmodel_counterfactual(res; dat=Td22, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0))

# KU 有内生性模型 + CO2 水平值分解
r2 = sfmodel_counterfactual(res; dat=Td22, depvar="lnc2",
    scenarios=Dict("agg2" => :quantile => 0.75),
    envar="agg2", ivvar="ivkind22",
    C_level=Td22.emission)
# r2.ΔC_frontier   — 前沿通道导致的碳排放变化（万吨）
# r2.ΔC_efficiency  — 效率通道导致的碳排放变化（万吨）
# r2.C_cf           — 反事实碳排放水平值

# KK 模型（无空间权重）
r3 = sfmodel_counterfactual(res_kk; dat=Td22, depvar="lnc2",
    scenarios=Dict("agg2" => 0.0))
```
"""
function sfmodel_counterfactual(res;
    dat::DataFrame,
    depvar::Union{String, Symbol},
    scenarios::Dict{String, <:Any},
    Wy_mat=nothing,
    Wu_mat=nothing,
    Wv_mat=nothing,
    envar=nothing,
    ivvar=nothing,
    C_level::Union{Nothing, AbstractVector}=nothing)

    modelid = res[:modelid]
    # --- 模型名称字符串检测（兼容 JLD2.UnknownType）---
    modelid_str = string(modelid)
    function _match_model(s, names)
        for n in names
            occursin(n, s) && return true
        end
        return false
    end
    is_ku = _match_model(modelid_str, ["SSFKUH", "SSFKUT", "SSFKUEH", "SSFKUET"])
    is_kk = _match_model(modelid_str, ["SSFKKH", "SSFKKT", "SSFKKEH", "SSFKKET"])
    is_oa = _match_model(modelid_str, ["SSFOAH", "SSFOAT", "SSFOADH", "SSFOADT"])
    is_wh = _match_model(modelid_str, ["SSFWHH", "SSFWHT", "SSFWHEH", "SSFWHET"])

    if is_wh
        error("WH 模型暂不支持反事实分析（原始 sdsfe 也未实现）")
    end

    # 检测分布和空间特征（字符串匹配，兼容 JLD2 加载）
    is_half = _match_model(modelid_str, ["SSFKUH", "SSFKUEH", "SSFKKH", "SSFKKEH", "SSFOAH", "SSFOADH", "SSFWHH", "SSFWHEH"])
    has_Wu  = _match_model(modelid_str, ["SSFOAH", "SSFOAT", "SSFOADH", "SSFOADT"])
    has_Wy  = _match_model(modelid_str, ["SSFOAH", "SSFOAT", "SSFOADH", "SSFOADT", "SSFKUH", "SSFKUT", "SSFKUEH", "SSFKUET"])
    has_mu  = !is_half
    cfg = (is_half=is_half, has_Wu=has_Wu, has_Wy=has_Wy, has_mu=has_mu)

    coef = res[:coeff]
    eqpo = res[:eqpo]
    PorC = res[:PorC]

    println(">>> 反事实分析 — 模型: $modelid_str ($(is_ku ? "KU" : is_kk ? "KK" : "OA"))")

    # --- 提取系数 ---
    idx_frontier = eqpo[:coeff_frontier]
    β = coef[idx_frontier]
    idx_q = eqpo[:coeff_log_hscale]
    τ = coef[idx_q]
    idx_w = eqpo[:coeff_log_σᵤ²]
    δ2 = coef[first(idx_w)]
    idx_v = eqpo[:coeff_log_σᵥ²]
    γ_val = coef[first(idx_v)]

    σᵤ² = exp(δ2)
    σᵥ² = exp(γ_val)
    μ = 0.0  # 半正态
    if cfg.has_mu && haskey(eqpo, :coeff_μ)
        idx_z = eqpo[:coeff_μ]
        μ = coef[first(idx_z)]
    end

    # --- 空间参数 ---
    gamma = nothing
    if cfg.has_Wy && haskey(eqpo, :coeff_γ)
        gamma = coef[first(eqpo[:coeff_γ])]
    end

    # --- Wu 空间参数 (OA 模型) ---
    tau_u = nothing
    if cfg.has_Wu && haskey(eqpo, :coeff_τ)
        tau_u = coef[first(eqpo[:coeff_τ])]
    end

    # --- Wv 空间参数 (OA 模型) ---
    rho_v = nothing
    if is_oa && haskey(eqpo, :coeff_ρ) && length(eqpo[:coeff_ρ]) > 0
        rho_v = coef[first(eqpo[:coeff_ρ])]
    end

    # --- 获取 Wy ---
    if Wy_mat === nothing && cfg.has_Wy
        Wy_raw = _dicM[:wy]
        Wy_mat = (Wy_raw !== nothing && Wy_raw !== Nothing) ? Wy_raw[1] : nothing
    end
    # 处理 Array{Matrix} 包装类型（用户可能传入 Wx 而非 Wx[1]）
    if Wy_mat !== nothing && !(Wy_mat isa AbstractMatrix{<:Real})
        Wy_mat = Wy_mat[1]
    end

    # --- 获取 Wu (OA 模型) ---
    if Wu_mat === nothing && cfg.has_Wu
        Wu_raw = _dicM[:wu]
        Wu_mat = (Wu_raw !== nothing && Wu_raw !== Nothing) ? Wu_raw[1] : nothing
    end
    if Wu_mat !== nothing && !(Wu_mat isa AbstractMatrix{<:Real})
        Wu_mat = Wu_mat[1]
    end

    # --- 获取 Wv (OA 模型) ---
    if Wv_mat === nothing && is_oa && rho_v !== nothing
        Wv_raw = _dicM[:wv]
        Wv_mat = (Wv_raw !== nothing && Wv_raw !== Nothing) ? Wv_raw[1] : nothing
    end
    if Wv_mat !== nothing && !(Wv_mat isa AbstractMatrix{<:Real})
        Wv_mat = Wv_mat[1]
    end

    # --- Wv 活跃标志 ---
    has_wv_active = is_oa && rho_v !== nothing && Wv_mat !== nothing
    if has_wv_active
        println("    Wv 空间权重已启用 (ρᵥ=$(round(rho_v, digits=6)))")
    end

    # --- 面板结构 ---
    idvar = Symbol(res[:idvar][1])
    timevar = Symbol(res[:timevar][1])

    # KK 模型内部按 [idvar, timevar] 排序，其他模型按 [timevar, idvar] 排序
    # 记录原始行顺序，计算完后排回去
    unsort_perm = nothing
    if is_kk
        dat = copy(dat)
        dat[!, :_orig_row_] = 1:nrow(dat)
        sort!(dat, [idvar, timevar])
        unsort_perm = sortperm(dat[!, :_orig_row_])
        select!(dat, Not(:_orig_row_))
    end

    tvar = dat[!, timevar]
    ivar = dat[!, idvar]
    years = sort(unique(tvar))
    T = length(years)
    city_codes = sort(unique(ivar))
    N = length(city_codes)
    nobs = nrow(dat)

    # --- 构建 rowIDT（与 sdsfe 内部一致）---
    rowIDT = Array{Any}(undef, T, 2)
    for (tt, yr) in enumerate(years)
        rowIDT[tt, 1] = findall(tvar .== yr)
        rowIDT[tt, 2] = length(rowIDT[tt, 1])
    end

    # --- 构建 rowIDI（按个体索引，KK 模型需要）---
    rowIDI = nothing
    if is_kk
        rowIDI = Array{Any}(undef, N, 2)
        for (ii, cc) in enumerate(city_codes)
            rowIDI[ii, 1] = findall(ivar .== cc)
            rowIDI[ii, 2] = length(rowIDI[ii, 1])
        end
    end

    # --- 构建变量矩阵 ---
    hscale_names = Symbol.(res[:hscale])
    frontier_names = Symbol.(res[:frontier])
    w_names = Symbol.(res[:σᵤ²])
    v_names = Symbol.(res[:σᵥ²])

    Q_orig = Float64.(Matrix(dat[!, hscale_names]))
    # 构建 X 矩阵，处理常数项（sdsfe 内部命名为 _consssssss）
    X_cols = Matrix{Float64}(undef, nobs, length(frontier_names))
    for (j, nm) in enumerate(frontier_names)
        if hasproperty(dat, nm)
            X_cols[:, j] .= Float64.(dat[!, nm])
        elseif occursin("cons", string(nm))
            X_cols[:, j] .= 1.0
        else
            error("前沿变量 $nm 不在 DataFrame 中")
        end
    end
    X_mat = X_cols
    nq = length(hscale_names)

    # --- 处理 WX 变量（如果有）---
    n_base = length(frontier_names)
    n_frontier_total = length(idx_frontier)
    if n_frontier_total > n_base && cfg.has_Wy && Wy_mat !== nothing
        # 有 WX 变量，需要构建完整的 X 矩阵
        # 从 table_show 提取 WX 变量并构建
        ts = res[:table_show]
        wx_data = zeros(nobs, n_frontier_total - n_base)
        wx_count = 0
        for r in 1:size(ts, 1)
            nm_str = string(ts[r, 2])
            if startswith(nm_str, "W*")
                wx_count += 1
                if wx_count <= n_frontier_total - n_base
                    base_nm = Symbol(nm_str[3:end])
                    base_col = Float64.(dat[!, base_nm])
                    # WX = Wy * X，按时间块计算
                    for (tt, yr) in enumerate(years)
                        idx = findall(tvar .== yr)
                        wx_data[idx, wx_count] = Wy_mat * base_col[idx]
                    end
                end
            end
        end
        X_mat = hcat(X_mat, wx_data)
    end
    # --- 应用反事实场景：修改 Q 矩阵 ---
    Q_cf = copy(Q_orig)
    for (var_name, scenario) in scenarios
        var_sym = Symbol(var_name)
        k = findfirst(==(var_sym), hscale_names)
        k === nothing && error("变量 $var_name 不在 hscale 变量列表 $hscale_names 中")

        orig_col = Q_orig[:, k]
        if scenario isa Real
            # 常数替换
            Q_cf[:, k] .= Float64(scenario)
            println("    $var_name → 常数 $(scenario)")
        elseif scenario isa Pair
            mode, val = scenario
            if mode == :quantile
                q_val = quantile(orig_col, val)
                Q_cf[:, k] .= q_val
                println("    $var_name → 第$(val*100)百分位数 = $(round(q_val, digits=4))")
            elseif mode == :shift
                sd_val = std(orig_col)
                Q_cf[:, k] .= orig_col .+ val * sd_val
                println("    $var_name → 原值 + $(val)σ (σ=$(round(sd_val, digits=4)))")
            elseif mode == :multiply
                Q_cf[:, k] .= orig_col .* val
                println("    $var_name → 原值 × $(val)")
            else
                error("未知场景模式: $(mode), 支持 :quantile, :shift, :multiply")
            end
        elseif scenario isa Symbol
            Q_cf[:, k] .= Float64.(dat[!, scenario])
            println("    $var_name → 使用列 $scenario 的值")
        elseif scenario isa AbstractVector
            length(scenario) == nobs || error("向量长度 $(length(scenario)) ≠ 观测数 $nobs")
            Q_cf[:, k] .= Float64.(scenario)
            println("    $var_name → 自定义向量")
        else
            error("不支持的场景类型: $(typeof(scenario))")
        end
    end

    # --- 计算前沿变化 (两通道分解: 前沿通道) ---
    delta_frontier_raw = zeros(nobs)

    # 提取 WX 变量名和对应系数位置
    wx_var_base_names = Symbol[]   # WX 变量的基础名（去掉 W* 前缀）
    wx_coef_local_idx = Int[]      # WX 系数在 β 中的局部位置
    if n_frontier_total > n_base
        ts = res[:table_show]
        wx_cnt = 0
        for r in 1:size(ts, 1)
            nm_str = string(ts[r, 2])
            if startswith(nm_str, "W*")
                wx_cnt += 1
                if wx_cnt <= n_frontier_total - n_base
                    push!(wx_var_base_names, Symbol(nm_str[3:end]))
                    push!(wx_coef_local_idx, n_base + wx_cnt)
                end
            end
        end
    end

    for (var_name, _) in scenarios
        var_sym = Symbol(var_name)
        k_hscale = findfirst(==(var_sym), hscale_names)
        delta_var = Q_cf[:, k_hscale] .- Q_orig[:, k_hscale]

        # 基础前沿系数 (非 WX 部分)
        j_base = findfirst(==(var_sym), frontier_names)
        if j_base !== nothing
            delta_frontier_raw .+= β[j_base] .* delta_var
        end

        # WX 前沿系数: θ × W × Δvar (按时间块)
        j_wx = findfirst(==(var_sym), wx_var_base_names)
        if j_wx !== nothing && cfg.has_Wy && Wy_mat !== nothing
            β_wx = β[wx_coef_local_idx[j_wx]]
            for tt in 1:T
                idx_t = rowIDT[tt, 1]
                delta_frontier_raw[idx_t] .+= β_wx .* (Wy_mat * delta_var[idx_t])
            end
        end
    end
    println("    前沿变化 (structural): 均值=$(round(mean(delta_frontier_raw), digits=6))")

    # --- 计算反事实 JLMS ---
    depvar_sym = Symbol(depvar)
    hi_cf = exp.(Q_cf * τ)
    ϵ = PorC * (Float64.(dat[!, depvar_sym]) - X_mat * β)

    # --- 内生性修正（SSFKUEH, SSFKUET 等含 E 的模型）---
    has_endo = haskey(eqpo, :coeff_ϕ) && haskey(eqpo, :coeff_η) &&
               length(eqpo[:coeff_ϕ]) > 0 && length(eqpo[:coeff_η]) > 0
    # 提升到函数作用域，供 OA+Wv 分支使用
    endo_eps_mat = nothing
    endo_eta_vec = nothing
    if has_endo
        if envar === nothing || ivvar === nothing
            @warn "模型含内生性但未提供 envar/ivvar，反事实结果可能不准确"
        else
            en_names = envar isa AbstractString ? [Symbol(envar)] : Symbol.(envar)
            iv_names_list = ivvar isa AbstractString ? [Symbol(ivvar)] : Symbol.(ivvar)

            # 构建第一阶段 IV 矩阵: [constant, frontier_excl_EN, hscale_excl_EN, iv_vars]
            IV_cols = Vector{Vector{Float64}}()
            for nm in frontier_names
                nm in en_names && continue
                if hasproperty(dat, nm)
                    push!(IV_cols, Float64.(dat[!, nm]))
                elseif occursin("cons", string(nm))
                    push!(IV_cols, ones(nobs))
                end
            end
            for nm in hscale_names
                nm in en_names && continue
                push!(IV_cols, Float64.(dat[!, nm]))
            end
            for nm in iv_names_list
                push!(IV_cols, Float64.(dat[!, nm]))
            end
            IV_mat = hcat(IV_cols...)

            phi_vec = Float64.(coef[eqpo[:coeff_ϕ]])
            eta_vec = Float64.(coef[eqpo[:coeff_η]])
            nofeta = length(eta_vec)
            phi_mat = reshape(phi_vec, :, nofeta)

            EN_mat = hcat([Float64.(dat[!, nm]) for nm in en_names]...)
            eps_mat = EN_mat - IV_mat * phi_mat

            # 内生性修正统一延迟到各模型分支内按需处理
            endo_eps_mat = eps_mat
            endo_eta_vec = eta_vec
            println("    内生性修正将在模型分支内应用 (EN=$(en_names), IV=$(iv_names_list))")
        end
    end

    if is_kk
        # === KK 模型: 按个体内积计算，无空间权重 ===
        # 内生性修正 (KK 无空间权重，全局修正)
        if has_endo && endo_eps_mat !== nothing
            ϵ .-= PorC * (endo_eps_mat * endo_eta_vec)
        end
        invPi = 1.0 / σᵥ²
        jlms_total = zeros(nobs)

        for idid in 1:N
            ind = rowIDI[idid, 1]
            hi_ind = hi_cf[ind]
            eps_ind = ϵ[ind]

            sig2_id = 1.0 / (dot(hi_ind, hi_ind) * invPi + 1.0 / σᵤ²)
            mu_id = (μ / σᵤ² - dot(eps_ind, hi_ind) * invPi) * sig2_id

            ratio = mu_id / sqrt(sig2_id)
            jlms_total[ind] .= hi_ind .* (mu_id + sqrt(sig2_id) * normpdf(ratio) / normcdf(ratio))
        end

        jlms_direct = copy(jlms_total)
        jlms_indirect = zeros(nobs)

    elseif is_oa
        # === OA 模型: 按时间段计算，含 Wu/Wy/Wv (8种组合) ===
        Mtau = (cfg.has_Wu && tau_u !== nothing && Wu_mat !== nothing) ?
               inv(I(N) - tau_u * Wu_mat) : Matrix{Float64}(I, N, N)
        Mgamma = (cfg.has_Wy && gamma !== nothing && Wy_mat !== nothing) ?
                 inv(I(N) - gamma * Wy_mat) : Matrix{Float64}(I, N, N)
        MgammaMtau = Mgamma * Mtau

        # invPi: 含 Wv 时为 Pi⁻¹ 矩阵，否则为 (1/σᵥ²)·I
        # 对应原始 sdsfe: Mrho=(I-ρᵥWv)⁻¹, Pi=σᵥ²(Mrho·Mrho'), invPi=Pi⁻¹
        Mrho_oa = nothing
        if has_wv_active
            Mrho_oa = inv(I(N) - rho_v * Wv_mat)
            Pi = σᵥ² * (Mrho_oa * Mrho_oa')
            invPi_mat = inv(Pi)
        else
            invPi_mat = (1.0 / σᵥ²) * Matrix{Float64}(I, N, N)
        end

        jlms_total = zeros(nobs)
        jlms_direct = zeros(nobs)

        for tt in 1:T
            ind = rowIDT[tt, 1]
            hi_ind = hi_cf[ind]
            hitau_ind = Mtau * hi_ind

            # Wy 修正: ϵ -= PorC·γ·Wy·y
            if cfg.has_Wy && gamma !== nothing && Wy_mat !== nothing
                y_ind = Float64.(dat[ind, depvar_sym])
                ϵ[ind] .-= PorC * gamma * Wy_mat * y_ind
            end

            # 内生性修正 (在时间循环内，Wy 修正之后)
            if endo_eps_mat !== nothing
                if has_wv_active
                    # OA+Wv: ϵ -= PorC·Mrho·(eps·η)
                    ϵ[ind] .-= PorC * Mrho_oa * (endo_eps_mat[ind, :] * endo_eta_vec)
                else
                    # OA-Wv: ϵ -= PorC·(eps·η)
                    ϵ[ind] .-= PorC * (endo_eps_mat[ind, :] * endo_eta_vec)
                end
            end

            eps_ind = ϵ[ind]
            # sig2_t = 1/(hitau'·invPi·hitau + 1/σᵤ²)
            sig2_t = 1.0 / (hitau_ind' * invPi_mat * hitau_ind + 1.0 / σᵤ²)
            # mu_t = (μ/σᵤ² - eps'·invPi·hitau)·sig2_t
            mu_t = (μ / σᵤ² - eps_ind' * invPi_mat * hitau_ind) * sig2_t

            ratio = mu_t / sqrt(sig2_t)
            base = hi_ind .* (mu_t + sqrt(sig2_t) * normpdf(ratio) / normcdf(ratio))

            jlms_total[ind] .= MgammaMtau * base
            jlms_direct[ind] .= Diagonal(diag(MgammaMtau)) * base
        end

        jlms_indirect = jlms_total .- jlms_direct

    elseif cfg.has_Wy && gamma !== nothing && Wy_mat !== nothing
        # === KU 模型 (空间): 逐观测计算，含 Wy ===
        Wyt = kron(I(T), Wy_mat)
        Mgamma = inv(I(N) - gamma * Wy_mat)
        Mgammat = kron(I(T), Mgamma)

        y_vec = Float64.(dat[!, depvar_sym])
        ϵ_spatial = ϵ .- PorC * gamma * Wyt * y_vec
        # 内生性修正: ϵ -= PorC·(eps·η)
        if has_endo && endo_eps_mat !== nothing
            ϵ_spatial .-= PorC * (endo_eps_mat * endo_eta_vec)
        end
        invPi = 1.0 / σᵥ²

        sigs2 = @. 1.0 / (hi_cf^2 * invPi + 1.0 / σᵤ²)
        mus = @. (μ / σᵤ² - ϵ_spatial * hi_cf * invPi) * sigs2

        jlms1 = @. hi_cf * (mus + sqrt(sigs2) *
            normpdf(mus / sqrt(sigs2)) / normcdf(mus / sqrt(sigs2)))
        jlms_total = Mgammat * jlms1
        jlms_direct = Diagonal(diag(Mgammat)) * jlms1
        jlms_indirect = jlms_total - jlms_direct
    else
        # === KU 模型 (非空间) 或其他无空间权重情况 ===
        # 内生性修正: ϵ -= PorC·(eps·η)
        if has_endo && endo_eps_mat !== nothing
            ϵ .-= PorC * (endo_eps_mat * endo_eta_vec)
        end
        invPi = 1.0 / σᵥ²
        sigs2 = @. 1.0 / (hi_cf^2 * invPi + 1.0 / σᵤ²)
        mus = @. (μ / σᵤ² - ϵ * hi_cf * invPi) * sigs2

        jlms_total = @. hi_cf * (mus + sqrt(sigs2) *
            normpdf(mus / sqrt(sigs2)) / normcdf(mus / sqrt(sigs2)))
        jlms_direct = jlms_total
        jlms_indirect = zeros(nobs)
    end

    # --- 前沿变化: 应用空间乘数 (I-ρW)⁻¹ ---
    has_spatial_wy = cfg.has_Wy && gamma !== nothing && Wy_mat !== nothing
    if has_spatial_wy
        Mgamma_f = inv(I(N) - gamma * Wy_mat)
        delta_lnC_frontier = zeros(nobs)
        for tt in 1:T
            ind = rowIDT[tt, 1]
            delta_lnC_frontier[ind] = Mgamma_f * delta_frontier_raw[ind]
        end
    else
        delta_lnC_frontier = copy(delta_frontier_raw)
    end

    # --- 效率变化: Δu = u_cf - u_orig ---
    jlms_orig = vec(Float64.(res[:jlms]))
    delta_lnC_efficiency = vec(jlms_total) .- jlms_orig
    delta_lnC_total = delta_lnC_frontier .+ delta_lnC_efficiency

    # --- 计算 CEE ---
    te_cf_total = exp.(-jlms_total)
    te_cf_direct = exp.(-jlms_direct)
    te_cf_indirect = exp.(-jlms_indirect)

    # --- KK 模型：将结果排回原始输入顺序 ---
    if unsort_perm !== nothing
        jlms_total = jlms_total[unsort_perm]
        jlms_direct = jlms_direct[unsort_perm]
        jlms_indirect = jlms_indirect[unsort_perm]
        te_cf_total = te_cf_total[unsort_perm]
        te_cf_direct = te_cf_direct[unsort_perm]
        te_cf_indirect = te_cf_indirect[unsort_perm]
        delta_lnC_frontier = delta_lnC_frontier[unsort_perm]
        delta_lnC_efficiency = delta_lnC_efficiency[unsort_perm]
        delta_lnC_total = delta_lnC_total[unsort_perm]
    end

    # --- 汇总 ---
    println(">>> 反事实分析完成")
    println("    反事实 CEE (total):    均值=$(round(mean(te_cf_total), digits=4)), 中位数=$(round(median(te_cf_total), digits=4))")
    println("    反事实 CEE (direct):   均值=$(round(mean(te_cf_direct), digits=4)), 中位数=$(round(median(te_cf_direct), digits=4))")
    println("    ΔlnC 前沿通道:        均值=$(round(mean(delta_lnC_frontier), digits=6))")
    println("    ΔlnC 效率通道:        均值=$(round(mean(delta_lnC_efficiency), digits=6))")
    println("    ΔlnC 总计:            均值=$(round(mean(delta_lnC_total), digits=6))")

    # --- 与原始 CEE 对比（如果 res 中有）---
    if haskey(res, :te)
        te_orig = res[:te]
        diff_total = te_cf_total .- te_orig
        println("    原始 CEE (total):      均值=$(round(mean(te_orig), digits=4))")
        println("    CEE 变化 (反事实-原始): 均值=$(round(mean(diff_total), digits=4))")
    end

    # --- CO2 分解（如果提供了 C_level）---
    ΔC_frontier = nothing
    ΔC_efficiency = nothing
    ΔC_total = nothing
    C_cf = nothing
    if C_level !== nothing
        length(C_level) == length(delta_lnC_total) ||
            error("C_level 长度 $(length(C_level)) ≠ 观测数 $(length(delta_lnC_total))")
        C = Float64.(C_level)
        ΔC_frontier   = C .* (exp.(delta_lnC_frontier) .- 1)
        ΔC_efficiency = C .* exp.(delta_lnC_frontier) .* (exp.(delta_lnC_efficiency) .- 1)
        ΔC_total      = ΔC_frontier .+ ΔC_efficiency
        C_cf          = C .+ ΔC_total
        println("    ΔC 前沿通道 (水平值):  总量=$(round(sum(ΔC_frontier), digits=2)), 均值=$(round(mean(ΔC_frontier), digits=4))")
        println("    ΔC 效率通道 (水平值):  总量=$(round(sum(ΔC_efficiency), digits=2)), 均值=$(round(mean(ΔC_efficiency), digits=4))")
        println("    ΔC 总计 (水平值):      总量=$(round(sum(ΔC_total), digits=2)), 均值=$(round(mean(ΔC_total), digits=4))")
    end

    return (counterfacttotal=vec(jlms_total),
            counterfactdire=vec(jlms_direct),
            counterfactindire=vec(jlms_indirect),
            te_cf_total=vec(te_cf_total),
            te_cf_direct=vec(te_cf_direct),
            te_cf_indirect=vec(te_cf_indirect),
            delta_lnC_frontier=vec(delta_lnC_frontier),
            delta_lnC_efficiency=vec(delta_lnC_efficiency),
            delta_lnC_total=vec(delta_lnC_total),
            ΔC_frontier=ΔC_frontier,
            ΔC_efficiency=ΔC_efficiency,
            ΔC_total=ΔC_total,
            C_cf=C_cf,
            scenarios=scenarios)
end
