
###############################################
####        Counterfactual analysis        ####
###############################################

#? --------------- Truncated Normal --------------


function  cfindext_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end





function  cfindext_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end








function  cfindext_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple,index::Dict,  rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))
end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end










function  cfindext_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple,index::Dict,  rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))
end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+ 1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma * (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms[ind] =Mgamma * ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end









function  cfindext_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end







function  cfindext_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1
ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    

end






function  jcfindext_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple,index::Dict,  rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end


function  cfindext_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = δ1

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);


@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];

@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   

end





function counterfactindex(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]
index = _dicOPT[:cfindices]

if Wy!=Nothing  # yuv
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
     dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms_, bc_  = cfindext_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, index,rowIDT )
else # yu
    jlms_, bc_  = cfindext_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, index,rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms_, bc_  = cfindext_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu,index, rowIDT )
else #y
    jlms_, bc_  = cfindext_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, index,rowIDT )  
end
end
jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms_, bc_  = cfindext_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu,index, rowIDT  )
else # u
    jlms_, bc_  = cfindext_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu,index, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms_, bc_  = cfindext_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu,index, rowIDT )
else # 
    jlms_, bc_  = cfindext_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu,index, rowIDT  )  
end
end
jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
end 

return jlms_df, bc_df  


end




function  cfindexdt_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
end





function  cfindexdt_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
end









function  cfindexdt_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
end










function  cfindexdt_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  
@views jlms[ind] = Mgamma*(hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end










function  cfindexdt_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end













function  cfindexdt_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end













function  cfindexdt_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end











function  cfindexdt_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);


@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc       
end

function IrhoWratiocf(gamma::Float64, rowIDT::Matrix{Any} )
    Wy = _dicM[:wy]
    T = size(rowIDT,1)
  if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

       @views N = rowIDT[1,2];
        wirho = (I(N)-gamma*Wy[1])\I(N)
        dire = tr(wirho)/N
        indire = (sum(wirho) - tr(wirho))/N
   else
       NN=0
       dire=0
       indire=0
       for ttt=i:T
       @views N = rowIDT[ttt,2];

        NN = NN +N
        wirho = (I(N)-gamma*Wy[ttt])\I(N)
        dire =dire+ tr(wirho)
        indire =indire+ (sum(wirho) - tr(wirho))
       end
        dire = dire/NN 
        indire = indire /NN
   end
   dire_ratio =dire / (dire + indire)
   indire_ratio =indire / (dire + indire)
   return dire_ratio,indire_ratio
end



function counterfactindex(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
   index = _dicOPT[:cfindices]

     if Wy!=Nothing  # yuv
        gammap = rho[pos.beggamma]
        gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
         dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                 jlms_, bc_ = cfindexdt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else # yu
                jlms_, bc_ = cfindexdt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms_, bc_ = cfindexdt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, index,rowIDT )
             else #y
                jlms_, bc_ = cfindexdt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index,rowIDT )  
             end
         end
         jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms_, bc_  = cfindexdt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu,index, rowIDT  )
             else # u
                jlms_, bc_ = cfindexdt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, index,rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms_, bc_  = cfindexdt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, index,rowIDT )
             else # 
                jlms_, bc_  = cfindexdt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, index,rowIDT  )  
             end
         end
         jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
     end 
     
     return jlms_df, bc_df
 
 end

 




 function  cfindexh_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
end





function  cfindexh_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
end








function  cfindexh_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end










function cfindexh_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end









function  cfindexh_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
end







function  cfindexh_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0
ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
end






function  cfindexh_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
end


function  cfindexh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);



@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
end





function counterfactindex(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]
  index = _dicOPT[:cfindices]

  gammap = rho[pos.beggamma]
  gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
   dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
    if Wy!=Nothing  # yuv
        if Wu!=Nothing 
            if Wv!=Nothing #yuv
                jlms_, bc_ = cfindexh_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, index, rowIDT )
            else # yu
                jlms_, bc_ = cfindexh_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, index, rowIDT  )
            end    
        else 
            if Wv!=Nothing #yv
                jlms_, bc_ = cfindexh_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, index, rowIDT )
            else #y
                jlms_, bc_ = cfindexh_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  
            end
        end
        jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
        bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
    else
        if Wu!=Nothing 
            if Wv!=Nothing #uv
                jlms_, bc_ = cfindexh_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, index, rowIDT  )
            else # u
                jlms_, bc_ = cfindexh_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, index, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms_, bc_ = cfindexh_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, index, rowIDT )
            else # 
                jlms_, bc_ = cfindexh_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, index, rowIDT  )  
            end
        end
        jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
        bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
    end 
    
    return jlms_df, bc_df  

  end







function  cfindexdh_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views invPi = (Pi)\I(N);

  @views Mgamma = (I(N)-gamma*Wy[1])\I(N)

  @floop begin
  @inbounds for ttt=1:T 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
    @views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # begin
end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views invPi = (Pi)\I(N);
  @views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
    @views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
        sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
end





function  cfindexdh_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc            
end









function  cfindexdh_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc       
end










function  cfindexdh_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms[ind] =Mgamma*( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ) )
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
end










function  cfindexdh_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
end













function  cfindexdh_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
end













function  cfindexdh_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(index[2], size(Q, 1))
Q[:, index[1]] = column_of_threes


phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
end











function  cfindexdh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0.0
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);




@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  ;
end # for ttt=1:T
end # begin
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
end




function counterfactindex(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
   index = _dicOPT[:cfindices]

   gammap = rho[pos.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms_, bc_ = cfindexdh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else # yu
                jlms_, bc_ = cfindexdh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms_, bc_ = cfindexdh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else #y
                jlms_, bc_ = cfindexdh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  
             end
         end
         jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms_, bc_ = cfindexdh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             else # u
                jlms_, bc_ = cfindexdh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms_, bc_ = cfindexdh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num,  pos, rho,  eigvalu, index, rowIDT )
             else # 
                jlms_, bc_ = cfindexdh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, index, rowIDT  )  
             end
         end
         jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
     end 
     
     return jlms_df, bc_df  
 
 end
 





#  function jlmsbc(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
#     PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
#    Wy = _dicM[:wy]
  
#    gammap = rho[pos.beggamma]
#    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
#     dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
     
#     β  = rho[1:pos.endx]
#     τ  = rho[pos.begq:pos.endq]
#     phi = rho[pos.begphi:pos.endphi]
    
#     nofiv=num.nofphi/num.nofeta
#     phi = reshape(phi,:,num.nofeta)
#     eps = EN-IV*phi
#     eta = rho[pos.begeta:pos.endeta]
    
#     δ2 = rho[pos.begw]  
#     γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
#     # δ1 = rho[pos.begz]
#     gammap = rho[pos.beggamma]
#     gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    
#     hi  = exp.(Q*τ)
#     σᵤ²= exp(δ2) 
#     σᵤ= exp(0.5*δ2) 
#     σᵥ² = exp(γ)  
#     σᵥ = exp(0.5*γ)    
#     μ   = 0.0
#     ϵ = PorC*(y - x * β )
#     T = size(rowIDT,1)
#     nobs = num.nofobs
#     sigs2 = zeros(eltype(y),T,1);
#     mus = zeros(eltype(y),T,1);
#     bc = zeros(eltype(y),nobs,1);
#     jlms = zeros(eltype(y),nobs,1);
    
#     if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
    
#         @views N = rowIDT[1,2];
#         @views invPi = 1.0 /σᵥ²*(I(N));
#         @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
        
#         @floop begin
#         @inbounds for ttt=1:T 
#             @views ind = rowIDT[ttt,1];
#             @views hi[ind]= hi[ind];
#             @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
#             @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
#             @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
#             @views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
#             @views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
#         end # for ttt=1:T
#         end # begin
    
#     elseif length(Wy)>1
#         @floop begin
#         @inbounds for ttt=1:T  
#             @views N = rowIDT[1,2];
#             @views invPi = 1.0 /σᵥ²*(I(N));
#             @views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)
            
#             @views ind = rowIDT[ttt,1];
#             @views hi[ind]= hi[ind];
#             @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
#             @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
#             @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
#             @views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
#             @views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
#         end # for ttt=1:T
#         end # begin
#     end  #    if length(Wy)==1 

#     @views bc_ = exp.(-bc);
#     @views jlms_ = exp.(-jlms);

#     jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
#     bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     
#      return jlms_df, bc_df  
#  end
 


 

#  function jlmsbc(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
#     PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
#    Wy = _dicM[:wy]
  
#    gammap = rho[pos.beggamma]
#    gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
#     dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
     
#     β  = rho[1:pos.endx]
#     τ  = rho[pos.begq:pos.endq]
#     phi = rho[pos.begphi:pos.endphi]
    
#     nofiv=num.nofphi/num.nofeta
#     phi = reshape(phi,:,num.nofeta)
#     eps = EN-IV*phi
#     eta = rho[pos.begeta:pos.endeta]
    
#     δ2 = rho[pos.begw]  
#     γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
#     δ1 = rho[pos.begz]
#     gammap = rho[pos.beggamma]
#     gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    
#     hi  = exp.(Q*τ)
#     σᵤ²= exp(δ2) 
#     σᵤ= exp(0.5*δ2) 
#     σᵥ² = exp(γ)  
#     σᵥ = exp(0.5*γ)    
#     μ   = δ1
#     ϵ = PorC*(y - x * β )
#     T = size(rowIDT,1)
#     nobs = num.nofobs
#     sigs2 = zeros(eltype(y),T,1);
#     mus = zeros(eltype(y),T,1);
#     bc = zeros(eltype(y),nobs,1);
#     jlms = zeros(eltype(y),nobs,1);
    
#     if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
    
#         @views N = rowIDT[1,2];
#         @views invPi = 1.0 /σᵥ²*(I(N));
#         @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
        
#         @floop begin
#         @inbounds for ttt=1:T 
#             @views ind = rowIDT[ttt,1];
#             @views hi[ind]= hi[ind];
#             @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
#             @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
#             @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
#             @views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
#             @views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
#         end # for ttt=1:T
#         end # begin
    
#     elseif length(Wy)>1
#         @floop begin
#         @inbounds for ttt=1:T  
#             @views N = rowIDT[1,2];
#             @views invPi = 1.0 /σᵥ²*(I(N));
#             @views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)
            
#             @views ind = rowIDT[ttt,1];
#             @views hi[ind]= hi[ind];
#             @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
#             @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
#             @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
#             @views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
#             @views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
#         end # for ttt=1:T
#         end # begin
#     end  #    if length(Wy)==1 

#     @views bc_ = exp.(-bc);
#     @views jlms_ = exp.(-jlms);

#     jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
#     bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     
#      return jlms_df, bc_df  
#  end
 

function  cfindexkute(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})


 β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes        
phi = rho[pos.begphi:pos.endphi]

phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

# sigs2 = zeros(eltype(y),T,1);
# mus = zeros(eltype(y),T,1);
# bc = zeros(eltype(y),size(hi,1),1);
# jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    @views N = rowIDT[1,2];
    @views Wyt = kron(I(T), Wy[1])
    @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
    @views Mgammat = kron(I(T), Mgamma)

    @views invPi = 1.0 /σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 = @. 1.0  / (hi^2 *invPi + 1.0 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
    @views jlms1 =@.  hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
    @views jlms =Mgammat*jlms1
    @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views bc = Mgammat*bc1;  

elseif length(Wy)>1

    @views Wyt = BlockDiagonal([Wy...])
    @views Mgammat_ = Array{Matrix}(undef, T);
    @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

        @views Mgammat_[ttt] = (I(N)-gamma*Wy[ttt])\I(N);
    end # for ttt=1:T
    @views Mgammat = BlockDiagonal([Mgammat_...])

    @views invPi = 1.0/σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 =@. 1.0 / (hi^2 *invPi + 1 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
    @views jlms1 = @. hi * ( mus + normpdf(mus/sqrt(sigs2))* sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
    @views jlms =Mgammat*jlms1
    @views bc1 = @. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views bc = Mgammat*bc1;  
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end




function  cfindexkut(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
   column_of_threes = fill(collect(values(index))[1], size(Q, 1))
   Q[:, collect(keys(index))[1]] = column_of_threes
   δ2 = rho[pos.begw]  
   γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[pos.begz]
   gammap = rho[pos.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
   
   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)    
   μ   = δ1
   ϵ = PorC*(y - x * β )
   T = size(rowIDT,1)
   
   # sigs2 = zeros(eltype(y),T,1);
   # mus = zeros(eltype(y),T,1);
   # bc = zeros(eltype(y),size(hi,1),1);
   # jlms = zeros(eltype(y),size(hi,1),1);
   
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
   
       @views N = rowIDT[1,2];
       @views Wyt = kron(I(T), Wy[1])
       @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
       @views Mgammat = kron(I(T), Mgamma)
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
 
       @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
       @views jlms =Mgammat*jlms1
       @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
       @views bc = Mgammat*bc1;  

   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
       @views Mgammat_ = Array{Matrix}(undef, T);
       @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

           @views Mgammat_[ttt] = (I(N)-gamma*Wy[ttt])\I(N);
       end # for ttt=1:T
       @views Mgammat = BlockDiagonal([Mgammat_...])
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma.*Wyt*y  ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1 /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
       @views jlms =Mgammat*jlms1
       @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
       @views bc =Mgammat* bc1;  
   end  #    if length(Wy)==1 
   @views TE_bc = exp.(-bc);
   @views TE_jlms = exp.(-jlms);
   return TE_jlms, TE_bc     
   end
   
   




function counterfactindex(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]

    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

    dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
    jlms_, bc_ = cfindexkute(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  

    jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
    bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
    
     return jlms_df, bc_df
 
 end

 function counterfactindex(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]

    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

    dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
    jlms_, bc_ = cfindexkut(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  

    jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
    bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
    
     return jlms_df, bc_df
 
 end


 
function  cdindexkuhe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
   column_of_threes = fill(collect(values(index))[1], size(Q, 1))
   Q[:, collect(keys(index))[1]] = column_of_threes
   phi = rho[pos.begphi:pos.endphi]
   
   phi = reshape(phi,:,num.nofeta)
   eps = EN-IV*phi
   eta = rho[pos.begeta:pos.endeta]
   
   δ2 = rho[pos.begw]  
   γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
#    δ1 = rho[pos.begz]
   gammap = rho[pos.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
   
   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)    
   μ   = 0.0
   ϵ = PorC*(y - x * β )
   T = size(rowIDT,1)
   
   # sigs2 = zeros(eltype(y),T,1);
   # mus = zeros(eltype(y),T,1);
   # bc = zeros(eltype(y),size(hi,1),1);
   # jlms = zeros(eltype(y),size(hi,1),1);
   
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
   
       @views N = rowIDT[1,2];
       @views Wyt = kron(I(T), Wy[1])
       @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
       @views Mgammat = kron(I(T), Mgamma)
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
       @views jlms =Mgammat*jlms1
       @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
       @views bc = Mgammat*bc1;  
   
   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
       @views Mgammat_ = Array{Matrix}(undef, T);
       @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

           @views Mgammat_[ttt] = (I(N)-gamma*Wy[ttt])\I(N);
       end # for ttt=1:T
       @views Mgammat = BlockDiagonal([Mgammat_...])
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
       @views jlms =Mgammat*jlms1
       @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
       @views bc = Mgammat*bc1;  
   end  #    if length(Wy)==1 
   @views TE_bc = exp.(-bc);
   @views TE_jlms = exp.(-jlms);
   return TE_jlms, TE_bc     
   end
   
   
   
   
   function  cfindexkuh(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
      
      
        β  = rho[1:pos.endx]
        τ  = rho[pos.begq:pos.endq]
        column_of_threes = fill(collect(values(index))[1], size(Q, 1))
        Q[:, collect(keys(index))[1]] = column_of_threes
        δ2 = rho[pos.begw]  
        γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    #   δ1 = rho[pos.begz]
      gammap = rho[pos.beggamma]
      gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
      
      hi  = exp.(Q*τ)
      σᵤ²= exp(δ2) 
      σᵤ= exp(0.5*δ2) 
      σᵥ² = exp(γ)  
      σᵥ = exp(0.5*γ)    
      μ   = 0.0
      ϵ = PorC*(y - x * β )
      T = size(rowIDT,1)
      
      # sigs2 = zeros(eltype(y),T,1);
      # mus = zeros(eltype(y),T,1);
      # bc = zeros(eltype(y),size(hi,1),1);
      # jlms = zeros(eltype(y),size(hi,1),1);
      
      if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
      
          @views N = rowIDT[1,2];
          @views Wyt = kron(I(T), Wy[1])
          @views Mgamma = (I(N)-gamma*Wy[1])\I(N)
          @views Mgammat = kron(I(T), Mgamma)
      
          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
          @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
          @views jlms =Mgammat*jlms1
          @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
          @views bc = Mgammat*bc1;  
      
      elseif length(Wy)>1
      
        @views Wyt = BlockDiagonal([Wy...])
          @views Mgammat_ = Array{Matrix}(undef, T);
          @inbounds for ttt=1:T  
            @views N = rowIDT[ttt,2];

              @views Mgammat_[ttt] = (I(N)-gamma*Wy[ttt])\I(N);
          end # for ttt=1:T
          @views Mgammat = BlockDiagonal([Mgammat_...])
      
          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
          @views jlms1 =@. hi * ( mus + normpdf(mus/sqrt(sigs2)) * sqrt(sigs2) / normcdf(mus/sqrt(sigs2)) )
          @views jlms =Mgammat*jlms1
          @views bc1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
          @views bc = Mgammat*bc1;  

        end  #    if length(Wy)==1 
      @views TE_bc = exp.(-bc);
      @views TE_jlms = exp.(-jlms);
      return TE_jlms, TE_bc     
      end
      
      
   
   
   
   
   function counterfactindex(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
       PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    
       Wy = _dicM[:wy]
       index = _dicOPT[:cfindices]

       gammap = rho[pos.beggamma]
       gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
           dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)
   
        jlms_, bc_ = cdindexkuhe(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  

           
        jlms_df = DataFrame(hcat(dire_ratio.*jlms_,indire_ratio.*jlms_), [:dire_jlms, :indire_jlms])
        bc_df = DataFrame(hcat(dire_ratio.*bc_,indire_ratio.*bc_), [:dire_bc, :indire_bc])
       
        return jlms_df, bc_df
    
    end



       
   function counterfactindex(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]

    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
        dire_ratio,indire_ratio = IrhoWratiocf(gamma, rowIDT)

    jlms_, bc_ = cfindexkuh(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  

        
     jlms_df = DataFrame(hcat(dire_ratio.*jlms_,indire_ratio.*jlms_), [:dire_jlms, :indire_jlms])
     bc_df = DataFrame(hcat(dire_ratio.*bc_,indire_ratio.*bc_), [:dire_bc, :indire_bc])
    
     return jlms_df, bc_df
 
 end



 
 function  cfindexkkte(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    column_of_threes = fill(collect(values(index))[1], size(Q, 1))
    Q[:, collect(keys(index))[1]] = column_of_threes
    phi = rho[pos.begphi:pos.endphi]
    
    phi = reshape(phi,:,num.nofeta)
    eps = EN-IV*phi
    eta = rho[pos.begeta:pos.endeta]
    
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    
    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)  
    σᵥ = exp(0.5*γ)    
    μ   = δ1
    ϵ = PorC*(y - x * β )
    ID = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),ID,1);
    mus = zeros(eltype(y),ID,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views bc[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
        @views jlms[ind] = hi[ind] .* ( mus[idid] + normpdf(mus[idid]/sqrt(sigs2[idid])) * sqrt(sigs2[idid]) / normcdf(mus[idid]/sqrt(sigs2[idid])) )  
    end # for idid=1:ID
    end # begin
   
    @views TE_bc = exp.(-bc);
    @views TE_jlms = exp.(-jlms);
    return TE_jlms, TE_bc     
    end
    
    
   
   
function  cfindexkkt(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
        PorC::Int64, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
       
       β  = rho[1:pos.endx]
       τ  = rho[pos.begq:pos.endq]
       column_of_threes = fill(collect(values(index))[1], size(Q, 1))
       Q[:, collect(keys(index))[1]] = column_of_threes
       δ2 = rho[pos.begw]  
       γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
       δ1 = rho[pos.begz]
       
       hi  = exp.(Q*τ)
       σᵤ²= exp(δ2) 
       σᵤ= exp(0.5*δ2) 
       σᵥ² = exp(γ)  
       σᵥ = exp(0.5*γ)    
       μ   = δ1
       ϵ = PorC*(y - x * β )
       ID = size(rowIDT,1)
       
       sigs2 = zeros(eltype(y),ID,1);
       mus = zeros(eltype(y),ID,1);
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       

       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for idid=1:ID 

           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views bc[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
           @views jlms[ind] = hi[ind] .* ( mus[idid] + normpdf(mus[idid]/sqrt(sigs2[idid])) * sqrt(sigs2[idid]) / normcdf(mus[idid]/sqrt(sigs2[idid])) ) 
       end # for idid=1:ID
       end # begin
      
       @views TE_bc = exp.(-bc);
       @views TE_jlms = exp.(-jlms);
       return TE_jlms, TE_bc     
    end    
      
      
    
function counterfactindex(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms_, bc_ = cfindexkkte(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, index, rowIDT )  

    return jlms_, bc_

end

function counterfactindex(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms_, bc_ = cfindexkkt(y, x, Q, PorC, pos, rho,  eigvalu, index, rowIDT )  

    return jlms_, bc_

end
   



 
function  cfindexkkhe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict, rowIDT::Matrix{Any})
   
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]

    column_of_threes = fill(collect(values(index))[1], size(Q, 1))
    Q[:, collect(keys(index))[1]] = column_of_threes
    phi = rho[pos.begphi:pos.endphi]
    
    phi = reshape(phi,:,num.nofeta)
    eps = EN-IV*phi
    eta = rho[pos.begeta:pos.endeta]
    
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    # δ1 = rho[pos.begz]
    
    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)  
    σᵥ = exp(0.5*γ)    
    μ   = 0.0
    ϵ = PorC*(y - x * β )
    ID = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),ID,1);
    mus = zeros(eltype(y),ID,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views bc[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
        @views jlms[ind] = hi[ind] .* ( mus[idid] + normpdf(mus[idid]/sqrt(sigs2[idid])) * sqrt(sigs2[idid]) / normcdf(mus[idid]/sqrt(sigs2[idid])) ) 
    end # for idid=1:ID
    end # begin
   
    @views TE_bc = exp.(-bc);
    @views TE_jlms = exp.(-jlms);
    return TE_jlms, TE_bc    
    end
    
    
   
   
   
function  cfindexkkh(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, index::Dict,rowIDT::Matrix{Any})


       β  = rho[1:pos.endx]
       τ  = rho[pos.begq:pos.endq]
       δ2 = rho[pos.begw]  
       γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    #    δ1 = rho[pos.begz]
        column_of_threes = fill(collect(values(index))[1], size(Q, 1))
        # println(column_of_threes)
        Q[:, collect(keys(index))[1]] = column_of_threes

       hi  = exp.(Q*τ)
       σᵤ²= exp(δ2) 
       σᵤ= exp(0.5*δ2) 
       σᵥ² = exp(γ)  
       σᵥ = exp(0.5*γ)    
       μ   = 0.0
       ϵ = PorC*(y - x * β )
       ID = size(rowIDT,1)
       
       sigs2 = zeros(eltype(y),ID,1);
       mus = zeros(eltype(y),ID,1);
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       
       @views invPi = 1.0 /σᵥ²  ;

       @floop begin
       @inbounds for idid=1:ID 
           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views bc[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
           @views jlms[ind] = hi[ind] .* ( mus[idid] + normpdf(mus[idid]/sqrt(sigs2[idid])) * sqrt(sigs2[idid]) / normcdf(mus[idid]/sqrt(sigs2[idid])) ) 
       end # for idid=1:ID
       end # begin
      
       @views TE_bc = exp.(-bc);
       @views TE_jlms = exp.(-jlms);



       return TE_jlms, TE_bc
       
    end
     
      
      
    
function counterfactindex(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms_, bc_ = cfindexkkhe(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu,index, rowIDT )  

    return jlms_, bc_

end

function counterfactindex(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms_, bc_ = cfindexkkh(y, x, Q, PorC, pos, rho,  eigvalu, index, rowIDT )  

    return jlms_, bc_

end