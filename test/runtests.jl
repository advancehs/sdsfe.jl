using Test
using sdsfe
using Optim:LineSearches

# include("..\\src\\sdsfe.jl")


using XLSX
using MAT
using CSV
using DataFrames



# dat=DataFrame(XLSX.readtable("C:\\Users\\10197\\yuv_nox_mu\\spxtsfayuv.xlsx", "Sheet1"  #filename
#            ));

dat=DataFrame(XLSX.readtable("C:\\Users\\10197\\spatialsfa_matlab\\spxtsfay.xlsx", "Sheet1" ));
# dat=DataFrame(XLSX.readtable("C:\\Users\\10197\\yuven_nox_mu\\spxtsfayuv.xlsx", "Sheet1" ));

dat[!, :_cons] .= 1.0;

# file = matopen("C:\\Users\\10197\\yuv_nox_mu\\wm.mat");
file = matopen("C:\\Users\\10197\\spatialsfa_matlab\\wm.mat");
# file = matopen("C:\\Users\\10197\\yuven_nox_mu\\wm.mat");


W = read(file, "wm"); # note that this does NOT introduce a variable ``varname`` into scope
Wx = Array{Matrix}(undef, 1, 1)
Wx[1]= W;

# sfmodel_spec(sfpanel(SSF_OA2019),sftype(prod), sfdist(half), wy(Wx), wx(Wx),wu(Wx),wv(Wx),
#                     @timevar(t), @idvar(id),
#                     @depvar(y),
#                     @frontier(noconstant,z1,qf),
#                     @frontierWx(z1,qf),
#                     @hscale(z2,qu),                # h(.) function
#                     # @envar(qf,qu),
#                     # @ivvar(z3,z4),
#                     # @μ(_cons),
#                     @σᵤ²(_cons),
#                     @σᵥ²(_cons), message = true);      

# sfmodel_spec(sfpanel(SSF_OAD2024),sftype(prod), sfdist(half), wy(Wx), wx(Wx),wu(Wx),wv(Wx),
#                     @timevar(t), @idvar(id),
#                     @depvar(y),
#                     @frontier(noconstant,z1,qf),
#                     @frontierWx(z1,qf),
#                     @hscale(z2,qu),                # h(.) function
#                     @envar(qf,qu),
#                     @ivvar(z3,z4),
#                     # @μ(_cons),
#                     @σᵤ²(_cons),
#                     @σᵥ²(_cons), message = true);             
                    
sfmodel_spec(sfpanel(SSF_KK2017),sftype(prod), sfdist(half), wy(Wx), #wx(Wx),
                    @timevar(t), @idvar(id),
                    @depvar(y),
                    @frontier(noconstant,z1,qf),
                    # @frontierWx(z1,qf),
                    @hscale(z2,qu),                # h(.) function
                    # @envar(qf,qu),
                    # @ivvar(z3,z4),
                    # @μ(_cons),
                    @σᵤ²(_cons),
                    @σᵥ²(_cons), message = true);   

sfmodel_opt(warmstart_solver(NelderMead()),   
                    warmstart_maxIT(1000),
                    # main_solver(Newton(linesearch=LineSearches.BackTracking())), 
                    main_solver(BFGS(linesearch=LineSearches.BackTracking())), 
                    main_maxIT(2000), 
                    tolerance(1e-6),autodiff_mode(forward),
                    cfindices(Dict(1=>3)));

res = sfmodel_fit(useData(dat)) 