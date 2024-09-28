using Test
using sdsfe
using Optim:LineSearches

# include("..\\src\\sdsfe.jl")


using XLSX
using MAT
using CSV
using DataFrames
using StatFiles
using LinearAlgebra

# 读取Stata数据文件
w1 = DataFrame(load(raw"D:\Program Files\Stata17\ado\personal\W_kerry.dta"))
w1 = Matrix(w1)
# 显示数据
Wx = Array{Matrix}(undef, 1, 1);

# 创建一个数组来存储归一化后的矩阵
# Julia中没有单元格数组的概念，通常我们使用数组的数组或字典来处理类似的数据结构
Wx[1] = w1;


dat = DataFrame(load(raw"D:\Program Files\Stata17\ado\personal\xtsfsp_ex1.dta"))
 # 显示前几行以确认数据已正确加载
dat[!, :_cons] .= 1.0;

dat
dat = sort(dat,  [:t, :id])



sfmodel_spec(sfpanel(SSF_OA2019),sftype(prod), sfdist(half), wy(Wx), wx(Wx),  wu(Wx), wv(Wx), 
                    @timevar(t), @idvar(id),
                    @depvar(y),
                    @frontier(constant ,x),
                    @frontierWx( x),
                    @hscale( z ),  
                    # @μ(_cons),
                    @σᵤ²(_cons),
                    @σᵥ²(_cons), message = true);    
                    

# sfmodel_spec(sfpanel(SSF_OA2019),sftype(prod), sfdist(trun), wy(Wx), wx(Wx), wu(Wx), wv(Wx), 
# @timevar(tt), @idvar(id),
# @depvar(lngdp),
# @frontier(constant,lnl22, lnk22,lnl_lnk22, lnl2_05, lnk2_05,  agg ,  indus ,human, roadpc,    lndis),
# @frontierWx(agg ,  indus , human, roadpc, lndis),
# @hscale( lndis,   agg,  indus ,human,roadpc),                # h(.) function
# @μ(_cons),
# @σᵤ²(_cons),
# @σᵥ²(_cons), message = true);    
                         
                    
sfmodel_opt(warmstart_solver(NelderMead()),   
                    warmstart_maxIT(600),
                    main_solver(BFGS(linesearch=LineSearches.BackTracking())), 
                    main_maxIT(3000), 
                    tolerance(1e-6),autodiff_mode(finite),cfindices(Dict(1=>6.304568)) );

res_yuvx_noen = sfmodel_fit(useData(dat)) 


# println(res_yuvx_noen[:totalemat])
# println(res_yuvx_noen[:totalematu])

# println(res_yuvx_noen[:jlms_direct])