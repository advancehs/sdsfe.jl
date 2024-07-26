using Test
using sdsfe

using XLSX
using MAT
using CSV
using DataFrames
using Statistics # 添加这一行
using PrettyTables              

using Optim: LineSearches

function translog(xs, normalize=0, T=nothing)

    # Handle normalization
    lnxs = normalize != 0 ? log.(xs) .- repeat(mean(log.(xs), dims=1), size(xs, 1)) : log.(xs)
    
    # Initialize xvar with lnK, lnL
    xvar = lnxs
    
    # Construct lnK*lnL
    for i in 1:size(lnxs, 2)-1
        for j in i+1:size(lnxs, 2)
            xvar = hcat(xvar, lnxs[:, i] .* lnxs[:, j])
        end
    end

    # Construct 0.5*lnK^2, 0.5*lnL^2
    for i in 1:size(lnxs, 2)
        xvar = hcat(xvar, 0.5 .* lnxs[:, i] .* lnxs[:, i] )
    end

    # If time variable T is provided
    if T !== nothing
        t = normalize != 0 ? (T .- minimum(T) .+ 1) .- mean(T .- minimum(T) .+ 1) : (T .- minimum(T) .+ 1)
        
        xvar = hcat(xvar, t)
        
        # Construct t*lnK, t*lnL
        for i in 1:size(lnxs, 2)
            xvar = hcat(xvar, t .* lnxs[:, i])
        end
        
        xvar = hcat(xvar, 0.5 * t .* t)
    end
    
    return xvar
end

function normw(W)
    # PURPOSE: normalize a spatial weight matrix
    # to have row sums of unity

    # Check if W is a square matrix or a 3D matrix
    if ndims(W) == 2 && size(W, 1) != size(W, 2)
        error("normw: W matrix must be square")
    end

    # Normalize matrix by its row sums or third dimension sums
    if ndims(W) == 2
        # For 2D matrices, normalize by row sums
        rowsums = sum(W, dims=2)
        # Protect against division by zero
        rowsums[rowsums .== 0] .= 1
        W .= W ./ rowsums
    elseif ndims(W) == 3
        # For 3D matrices, normalize across the third dimension
        dimsums = sum(W, dims=3)
        # Protect against division by zero
        dimsums[dimsums .== 0] .= 1
        for i in 1:size(W, 1), j in 1:size(W, 2)
            W[i, j, :] ./= dimsums[i, j, :]
        end
    else
        error("normw: W must be a 2D or 3D matrix")
    end

    return W
end


dat = CSV.read(raw"D:\Pythonwork\j高铁\高铁数据.csv", DataFrame)
dat[!, :_cons] .= 1.0;
# 显示前几行以确认数据已正确加载
# println(first(dat, 5))



wawaw_df = CSV.read(raw"D:\Pythonwork\j高铁\空间权重矩阵\w_queen_df.csv", DataFrame);

# 将DataFrame转换为数组，并进行归一化处理
# 这里我们跳过了第一行和第一列，与MATLAB代码中的 wawaw(2:end,2:end) 相对应
wawaw_matrix = Matrix(wawaw_df)[1:end, 2:end];
normalized_wawaw = normw(wawaw_matrix);

Wx = Array{Matrix}(undef, 1, 1);

# 创建一个数组来存储归一化后的矩阵
# Julia中没有单元格数组的概念，通常我们使用数组的数组或字典来处理类似的数据结构
Wx[1] = normalized_wawaw;

Tdatayuv = dat

T = Tdatayuv[!, :year]
xs = Matrix(Tdatayuv[!, [:labor, :cap_stock]])
trans_fake = translog(xs, 1, T)
trans_fake_t = DataFrame(trans_fake, [:lnl22, :lnk22, :lnl_lnk22, :lnl2_05, :lnk2_05, :t, :t_lnl, :t_lnk, :t2_05])
Tdatayuv2 = hcat(Tdatayuv, trans_fake_t);


xs2 = Matrix(Tdatayuv[!, [:lngdp, :agg, :human, :indus, :fdi, :land, :czczl, :roadpc, :rhgdzc, :rjys, :jjhl, :gjmy, :lndis1,:ivdis]])
trans_fake2 = xs2 .- mean(xs2, dims=1)
trans_fake_t2 = DataFrame(trans_fake2, [:lngdp2, :agg2, :human2, :indus2, :fdi2, :land2, :czczl2, :roadpc2, :rhgdzc2, :rjys2, :jjhl2, :gjmy2, :lndis2,:ivdis2])
Tdatayuv22 = hcat(Tdatayuv2, trans_fake_t2, makeunique=true)

# convert(Matrix, Tdatayuv[!, [:lngdp, :agg, :human, :indus, :fdi, :land, :czczl, :roadpc, :rhgdzc, :rjys, :jjhl, :gjmy, :lndis1]])

# println(first(Tdatayuv22, 5))


sfmodel_spec(sfpanel(SSF_OAD2024),sftype(prod), sfdist(trun), wy(Wx),  wu(Wx),  wv(Wx),wx(Wx),
                    @timevar(tt), @idvar(id),
                    @depvar(lngdp2),
                    @frontier(constant   ,lnl22, lnk22,lnl_lnk22, lnl2_05, lnk2_05,  agg2 , indus2 ,human2, roadpc2, lndis2),
                    @frontierWx(agg2 ,  indus2 , human2, roadpc2, lndis2),
                    @hscale( lndis2,   agg2,  indus2 ,human2,roadpc2),                # h(.) function
                    @envar(lndis2),
                    @ivvar(ivdis2),
                    @μ(_cons),
                    @σᵤ²(_cons),
                    @σᵥ²(_cons), message = true);    
                    
                    
sfmodel_opt(warmstart_solver(NelderMead()),   
                    warmstart_maxIT(600),
                    main_solver(BFGS(linesearch=LineSearches.BackTracking())), 
                    main_maxIT(300000), 
                    tolerance(1e-6),table_format(text),autodiff_mode(forward));

res_yuvx = sfmodel_fit(useData(Tdatayuv22)) 



