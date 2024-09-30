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

using StatFiles
using DataFrames

# 读取 Stata 文件
# df = DataFrame(load("D:\\BaiduSyncdisk\\software\\sdsfe\\test_sdsfe\\xtsfkkprod.dta"))

# 读取 Excel 文件
df = XLSX.readtable("D:\\OneDrive\\deabook\\EJOR_ENERGY\\Monte Carlo\\datayuv.xlsx", "Sheet1") |> DataFrame
df[!, :_cons] .= 1.0;
df = sort(df,  [:id, :t])

# 打印数据框
println(first(df, 5))





    
sfmodel_spec(sfpanel(SSF_KKE2017),sftype(prod), sfdist(half),
                    @timevar(t), @idvar(id),
                    @depvar(y),
                    @frontier(constant ,s1, qf),
                    # @frontierWx( agg2, indus2 ,human2 ,fdi2, roadpc2, lndis2),
                    @hscale( s2,qu ),               # h(.) function
                    @envar(qf,qu),
                    @ivvar(z3,z4),
                    # @μ(_cons),
                    @σᵤ²(_cons),
                    @σᵥ²(_cons), message = true);                   

                    
sfmodel_opt(warmstart_solver(NelderMead()),   
                    warmstart_maxIT(600),
                    main_solver(BFGS(linesearch=LineSearches.BackTracking())), 
                    main_maxIT(3000), 
                    tolerance(1e-6),autodiff_mode(forward),
                    
                    margeffu(false),
                    mareffx(false),
                    counterfact(false),
                    ineff_index(false)                    
                    
                    
                    );

res_2017 = sfmodel_fit(useData(df)) 




    
# sfmodel_spec(sfpanel(SSF_KKE2017),sftype(prod), sfdist(trun), 
#                     @timevar(year), @idvar(firm),
#                     @depvar(y),
#                     @frontier(constant ,x1, x2,x3,z1,),
#                     # @frontierWx( agg2, indus2 ,human2 ,fdi2, roadpc2, lndis2),
#                     @hscale( z2 ),             # h(.) function
#                     @envar(z1 , z2),
#                     @ivvar(iv1,iv2),
#                     @μ(_cons),
#                     @σᵤ²(_cons),
#                     @σᵥ²(_cons), message = true);                   

                    
# sfmodel_opt(warmstart_solver(NelderMead()),   
#                     warmstart_maxIT(600),
#                     main_solver(BFGS(linesearch=LineSearches.BackTracking())), 
#                     main_maxIT(3000), 
#                     tolerance(1e-6),autodiff_mode(forward),cfindices(Dict(1=>6.304568)) );

# res_en2017 = sfmodel_fit(useData(df)) 


