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
    half_set = (SSFOAH, SSFOADH, SSFKUH, SSFKUEH, SSFKKH, SSFKKEH, SSFWHH, SSFWHEH)
    wu_set   = (SSFOAH, SSFOAT, SSFOADH, SSFOADT)
    wy_set   = (SSFOAH, SSFOAT, SSFOADH, SSFOADT, SSFKUH, SSFKUT, SSFKUEH, SSFKUET)
    is_half = modelid in half_set
    has_Wu  = modelid in wu_set
    has_Wy  = modelid in wy_set
    has_mu  = !is_half
    return (is_half=is_half, has_Wu=has_Wu, has_Wy=has_Wy, has_mu=has_mu)
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
        rymin = ev.rymin; rymax = ev.rymax
        rumin = ev.rumin; rumax = ev.rumax
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
