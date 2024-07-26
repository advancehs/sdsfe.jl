using Test
using sdsfe

# include("..\\src\\sdsfe.jl")


using XLSX
using MAT
using CSV
using DataFrames



dat=DataFrame(XLSX.readtable("C:\\Users\\10197\\yuv_nox\\spxtsfayuv.xlsx", "Sheet1"  #filename
           ));
dat[!, :_cons] .= 1.0;
file = matopen("C:\\Users\\10197\\yuv_nox\\wm.mat");
W = read(file, "wm"); # note that this does NOT introduce a variable ``varname`` into scope
Wx = Array{Matrix}(undef, 1, 1)
Wx[1]= W;



sfmodel_spec(sfpanel(SSF_OA2019),sftype(prod), sfdist(half), wy(Wx) ,wu(Wx),wv(Wx),
                    @timevar(t), @idvar(id),
                    @depvar(y),
                    @frontier(x),
                    @hscale(z),                # h(.) function
                    @σᵤ²(_cons),
                    @σᵥ²(_cons), message = true);    
                    
                    
sfmodel_opt(warmstart_solver(NelderMead()),   
                    warmstart_maxIT(600),
                    main_solver(BFGS()), 
                    main_maxIT(2000), 
                    tolerance(1e-6));

res = sfmodel_fit(useData(dat)) 

