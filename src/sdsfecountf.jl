
###############################################
####        Counterfactual analysis        ####
###############################################

#? --------------- Truncated Normal --------------


function  cfindext_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, 
    index::Dict, rowIDT::Matrix{Any})
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct 

end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct 

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
end







function  cfindext_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]=  hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =1* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

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
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);



@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =1* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin

return jlms,jlms_direct,jlms_indirect
end






function counterfactindex(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]
index = _dicOPT[:cfindices]

if Wy!=Nothing  # yuv

if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms,jlms_direct,jlms_indirect  = cfindext_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, index,rowIDT )
else # yu
    jlms,jlms_direct,jlms_indirect  = cfindext_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, index,rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms,jlms_direct,jlms_indirect  = cfindext_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu,index, rowIDT )
else #y
    jlms,jlms_direct,jlms_indirect  = cfindext_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, index,rowIDT )  
end
end
else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms,jlms_direct,jlms_indirect  = cfindext_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu,index, rowIDT  )
else # u
    jlms,jlms_direct,jlms_indirect  = cfindext_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu,index, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms,jlms_direct,jlms_indirect  = cfindext_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu,index, rowIDT )
else # 
    jlms,jlms_direct,jlms_indirect  = cfindext_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu,index, rowIDT  )  
end
end

end 

return jlms,jlms_direct,jlms_indirect  


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

 
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

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
  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
     normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
  normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
  normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
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

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);



@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin

return jlms,jlms_direct,jlms_indirect
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

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = cfindexdt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else # yu
                jlms,jlms_direct,jlms_indirect = cfindexdt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = cfindexdt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, index,rowIDT )
             else #y
                jlms,jlms_direct,jlms_indirect = cfindexdt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index,rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect  = cfindexdt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu,index, rowIDT  )
             else # u
                jlms,jlms_direct,jlms_indirect = cfindexdt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, index,rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect  = cfindexdt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, index,rowIDT )
             else # 
                jlms,jlms_direct,jlms_indirect  = cfindexdt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, index,rowIDT  )  
             end
         end

     end 
     
     return jlms,jlms_direct,jlms_indirect
 
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct 

end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct 

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]=  hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =1* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

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
@views jlms[ind] =Mtau* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms,jlms_direct,jlms_indirect
end






function  cfindexh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple,index::Dict, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
column_of_threes = fill(collect(values(index))[1], size(Q, 1))
Q[:, collect(keys(index))[1]] = column_of_threes
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);



@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =1* (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ; 
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin

return jlms,jlms_direct,jlms_indirect
end








function counterfactindex(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]
  index = _dicOPT[:cfindices]


    if Wy!=Nothing  # yuv
        if Wu!=Nothing 
            if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = cfindexh_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, index, rowIDT )
            else # yu
                jlms,jlms_direct,jlms_indirect = cfindexh_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, index, rowIDT  )
            end    
        else 
            if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = cfindexh_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, index, rowIDT )
            else #y
                jlms,jlms_direct,jlms_indirect = cfindexh_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  
            end
        end

    else
        if Wu!=Nothing 
            if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect = cfindexh_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, index, rowIDT  )
            else # u
                jlms,jlms_direct,jlms_indirect = cfindexh_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, index, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect = cfindexh_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, index, rowIDT )
            else # 
                jlms,jlms_direct,jlms_indirect = cfindexh_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, index, rowIDT  )  
            end
        end

    end 
    
    return jlms,jlms_direct,jlms_indirect

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
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

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
  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
     normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
  normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
  normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct
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

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views invPi = 1.0 /σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mgamma*Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mgamma*Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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



jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (Mtau)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(Mtau))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms,jlms_direct,jlms_indirect
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


jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms,jlms_direct,jlms_indirect
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

jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);
hitau = zeros(eltype(hi),size(hi,1),1);



@floop begin
@inbounds for ttt=1:T 
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (1)*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_direct[ind] = Diagonal(diag(1))*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
    normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  
@views jlms_indirect = jlms - jlms_direct

end # for ttt=1:T
end # begin

return jlms,jlms_direct,jlms_indirect
end




function counterfactindex(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
   index = _dicOPT[:cfindices]


     if Wy!=Nothing  # yuv

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = cfindexdh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else # yu
                jlms,jlms_direct,jlms_indirect = cfindexdh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = cfindexdh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT )
             else #y
                jlms,jlms_direct,jlms_indirect = cfindexdh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect = cfindexdh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, index, rowIDT  )
             else # u
                jlms,jlms_direct,jlms_indirect = cfindexdh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, index, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect = cfindexdh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num,  pos, rho,  eigvalu, index, rowIDT )
             else # 
                jlms,jlms_direct,jlms_indirect = cfindexdh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, index, rowIDT  )  
             end
         end

     end 
     
     return jlms,jlms_direct,jlms_indirect
 
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

    @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms = Mgammat*jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;

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
    @views jlms1 = @. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms = Mgammat*jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
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

    @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms = Mgammat*jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;
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

    @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms =Mgammat* jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect
end





function counterfactindex(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]


    jlms,jlms_direct,jlms_indirect = cfindexkute(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

 
 end

 function counterfactindex(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]

    jlms,jlms_direct,jlms_indirect = cfindexkut(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  
    return jlms,jlms_direct,jlms_indirect
 
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

    @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms = Mgammat*jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;
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

    @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
    @views jlms = Mgammat*jlms1;  
    @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
    @views jlms_indirect = jlms - jlms_direct;
end  #    if length(Wy)==1 

return jlms,jlms_direct,jlms_indirect    
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

        @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
        @views jlms = Mgammat*jlms1;  
        @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
        @views jlms_indirect = jlms - jlms_direct;
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
  
        @views jlms1 =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
        @views jlms = Mgammat*jlms1;  
        @views jlms_direct = Diagonal(diag(Mgammat))*jlms1;  
        @views jlms_indirect = jlms - jlms_direct;
      end  #    if length(Wy)==1 

    return jlms,jlms_direct,jlms_indirect
    end
    
    
 
 
      
   
   
   
   
   function counterfactindex(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
       PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    
       Wy = _dicM[:wy]
       index = _dicOPT[:cfindices]

       jlms,jlms_direct,jlms_indirect = cdindexkuhe(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, index, rowIDT )  
       return jlms,jlms_direct,jlms_indirect

    
    end



       
   function counterfactindex(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    index = _dicOPT[:cfindices]

    jlms,jlms_direct,jlms_indirect = cfindexkuh(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

 
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
 
    
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);

    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
        @views jlms_direct = jlms
        @views jlms_indirect = jlms - jlms_direct;
    end # for idid=1:ID
    end # begin
   
    return jlms,jlms_direct,jlms_indirect
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
 
       jlms = zeros(eltype(y),size(hi,1),1);
       jlms_direct = zeros(eltype(y),size(hi,1),1);
       jlms_indirect = zeros(eltype(y),size(hi,1),1);

       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for idid=1:ID 

           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
           @views jlms_direct = jlms
           @views jlms_indirect = jlms - jlms_direct;
        end # for idid=1:ID
       end # begin
      

       return jlms,jlms_direct,jlms_indirect
    end    
      
      
      
    
function counterfactindex(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms,jlms_direct,jlms_indirect = cfindexkkte(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function counterfactindex(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms,jlms_direct,jlms_indirect  = cfindexkkt(y, x, Q, PorC, pos, rho,  eigvalu, index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

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
 
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);

    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
        @views jlms_direct = jlms
        @views jlms_indirect = jlms - jlms_direct;
    end # for idid=1:ID
    end # begin
   

    return jlms,jlms_direct,jlms_indirect
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
 
       jlms = zeros(eltype(y),size(hi,1),1);
       jlms_direct = zeros(eltype(y),size(hi,1),1);
       jlms_indirect = zeros(eltype(y),size(hi,1),1);


       @views invPi = 1.0 /σᵥ²  ;

       @floop begin
       @inbounds for idid=1:ID 
           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
           @views jlms_direct = jlms
           @views jlms_indirect = jlms - jlms_direct;
        end # for idid=1:ID
       end # begin

       return jlms,jlms_direct,jlms_indirect
    end    
      
      


      
      
    
function counterfactindex(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]

    jlms,jlms_direct,jlms_indirect = cfindexkkhe(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu,index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function counterfactindex(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    index = _dicOPT[:cfindices]
   
    jlms,jlms_direct,jlms_indirect = cfindexkkh(y, x, Q, PorC, pos, rho,  eigvalu, index, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end
# ============ Giannini 2025 (GI) 系列 ============
function counterfactindex(::Type{SSFGIEH},
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    qvar = dat[:, _dicM[:hscale]]

    envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])
    ivvar = dat[:, name_new_ivvar]

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))
    IV = convert(Array{Float64}, Matrix(ivvar))
    EN = convert(Array{Float64}, Matrix(envar))

    jlms,jlms_direct,jlms_indirect = cfindexgihe(y, x, Q, EN, IV, PorC, num, pos, rho, eigvalu, rowIDT)

    return jlms,jlms_direct,jlms_indirect
end

function counterfactindex(::Type{SSFGIH},
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    qvar = dat[:, _dicM[:hscale]]

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))

    jlms,jlms_direct,jlms_indirect = cfindexgih(y, x, Q, PorC, pos, rho, eigvalu, rowIDT)

    return jlms,jlms_direct,jlms_indirect
end

function counterfactindex(::Type{SSFGIET},
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    qvar = dat[:, _dicM[:hscale]]

    envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])
    ivvar = dat[:, name_new_ivvar]

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))
    IV = convert(Array{Float64}, Matrix(ivvar))
    EN = convert(Array{Float64}, Matrix(envar))

    jlms,jlms_direct,jlms_indirect = cfindexgite(y, x, Q, EN, IV, PorC, num, pos, rho, eigvalu, rowIDT)

    return jlms,jlms_direct,jlms_indirect
end

function counterfactindex(::Type{SSFGIT},
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}, dat::DataFrame)

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    qvar = dat[:, _dicM[:hscale]]

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))

    jlms,jlms_direct,jlms_indirect = cfindexgit(y, x, Q, PorC, pos, rho, eigvalu, rowIDT)

    return jlms,jlms_direct,jlms_indirect
end



function cfindexgihe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, EN::Matrix, IV::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    phi = rho[pos.begphi:pos.endphi]
    phi = reshape(phi, :, num.nofeta)
    eps = EN - IV*phi
    eta = rho[pos.begeta:pos.endeta]
    δ2 = rho[pos.begw]
    γ  = rho[pos.begv]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    Q_lag = _dicM[:qvar_lag]
    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    μ   = 0.0
    ϵ   = PorC*(y - x*β)

    ID     = size(rowIDT, 1)
    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    Wy = _dicM[:wy]
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y - PorC*(eps*eta)
    end

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²

    jlms1         = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_direct   = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_indirect = zeros(eltype(y), size(hi_cur,1), 1)

    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]
        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        @views jlms1[ind] = @. hi_cur[ind] * (mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2)) / normcdf(mus/sqrt(sigs2)))
    end

    # Apply spatial multiplier (I-λW)⁻¹ per Giannini (2025) Eq. 6
    N = ID
    @views Mgamma = (I(N) - gamma*Wy[1])\I(N)
    @views Mgammat = kron(Mgamma, I(T_fd))
    jlms          = Mgammat * jlms1
    jlms_direct   = Diagonal(diag(Mgammat)) * jlms1
    jlms_indirect = jlms - jlms_direct

    return jlms, jlms_direct, jlms_indirect
end


function cfindexgih(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,
    PorC::Int64, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]
    γ  = rho[pos.begv]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    Q_lag = _dicM[:qvar_lag]
    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    μ   = 0.0
    ϵ   = PorC*(y - x*β)

    ID     = size(rowIDT, 1)
    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    Wy = _dicM[:wy]
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y
    end

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²

    jlms1         = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_direct   = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_indirect = zeros(eltype(y), size(hi_cur,1), 1)

    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]
        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        @views jlms1[ind] = @. hi_cur[ind] * (mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2)) / normcdf(mus/sqrt(sigs2)))
    end

    # Apply spatial multiplier (I-λW)⁻¹ per Giannini (2025) Eq. 6
    N = ID
    @views Mgamma = (I(N) - gamma*Wy[1])\I(N)
    @views Mgammat = kron(Mgamma, I(T_fd))
    jlms          = Mgammat * jlms1
    jlms_direct   = Diagonal(diag(Mgammat)) * jlms1
    jlms_indirect = jlms - jlms_direct

    return jlms, jlms_direct, jlms_indirect
end


function cfindexgite(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, EN::Matrix, IV::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    phi = rho[pos.begphi:pos.endphi]
    phi = reshape(phi, :, num.nofeta)
    eps = EN - IV*phi
    eta = rho[pos.begeta:pos.endeta]
    δ2 = rho[pos.begw]
    γ  = rho[pos.begv]
    δ1 = rho[pos.begz]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    Q_lag = _dicM[:qvar_lag]
    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    μ   = δ1
    ϵ   = PorC*(y - x*β)

    ID     = size(rowIDT, 1)
    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    Wy = _dicM[:wy]
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y - PorC*(eps*eta)
    end

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²

    jlms1         = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_direct   = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_indirect = zeros(eltype(y), size(hi_cur,1), 1)

    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]
        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        @views jlms1[ind] = @. hi_cur[ind] * (mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2)) / normcdf(mus/sqrt(sigs2)))
    end

    # Apply spatial multiplier (I-λW)⁻¹ per Giannini (2025) Eq. 6
    N = ID
    @views Mgamma = (I(N) - gamma*Wy[1])\I(N)
    @views Mgammat = kron(Mgamma, I(T_fd))
    jlms          = Mgammat * jlms1
    jlms_direct   = Diagonal(diag(Mgammat)) * jlms1
    jlms_indirect = jlms - jlms_direct

    return jlms, jlms_direct, jlms_indirect
end


function cfindexgit(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,
    PorC::Int64, pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]
    γ  = rho[pos.begv]
    δ1 = rho[pos.begz]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    Q_lag = _dicM[:qvar_lag]
    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵥ² = exp(γ)
    μ   = δ1
    ϵ   = PorC*(y - x*β)

    ID     = size(rowIDT, 1)
    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    Wy = _dicM[:wy]
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y
    end

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²

    jlms1         = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_direct   = zeros(eltype(y), size(hi_cur,1), 1)
    jlms_indirect = zeros(eltype(y), size(hi_cur,1), 1)

    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]
        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        @views jlms1[ind] = @. hi_cur[ind] * (mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2)) / normcdf(mus/sqrt(sigs2)))
    end

    # Apply spatial multiplier (I-λW)⁻¹ per Giannini (2025) Eq. 6
    N = ID
    @views Mgamma = (I(N) - gamma*Wy[1])\I(N)
    @views Mgammat = kron(Mgamma, I(T_fd))
    jlms          = Mgammat * jlms1
    jlms_direct   = Diagonal(diag(Mgammat)) * jlms1
    jlms_indirect = jlms - jlms_direct

    return jlms, jlms_direct, jlms_indirect
end
# ============ Wang & Ho 2010 (WH) 系列 ============


 
function  cfindexwhte(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
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
 
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);


    @views invPi = 1.0 /σᵥ²  ;
    @floop begin
    @inbounds for iidd=1:ID 
         @views T = rowIDT[iidd,2];
         @views onecol = ones(T, 1);
         @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
         @views ind = rowIDT[iidd,1];
         @views his = IMT*(hi[ind]);
         @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
         @views sigs2[iidd] = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
         @views mus[iidd] = (μ/σᵤ² - (ϵ[ind]'*his*invPi))*sigs2[iidd] ;
         @views jlms[ind] = @. hi[ind] *( mus[iidd] + sqrt(sigs2[iidd])* normpdf(
                                mus[iidd]/sqrt(sigs2[iidd]))/normcdf(mus[iidd]/sqrt(sigs2[iidd]))   ) ;  
         @views jlms_direct[ind] = jlms[ind]
         @views jlms_indirect[ind] = jlms[ind] - jlms_direct[ind];

     end # for iidd=1:ID
    end # begin

    return jlms,jlms_direct,jlms_indirect
    end
    
    
   
function  cfindexwht(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})


       β  = rho[1:pos.endx]
       τ  = rho[pos.begq:pos.endq]
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
 
       jlms = zeros(eltype(y),size(hi,1),1);
       jlms_direct = zeros(eltype(y),size(hi,1),1);
       jlms_indirect = zeros(eltype(y),size(hi,1),1);


       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for iidd=1:ID 
            @views T = rowIDT[iidd,2];
            @views onecol = ones(T, 1);
            @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
            @views ind = rowIDT[iidd,1];
            @views his = IMT*(hi[ind]);
            @views ϵ[ind] = ϵ[ind] ;
            @views sigs2[iidd] = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
            @views mus[iidd] = (μ/σᵤ² - (ϵ[ind]'*his*invPi))*sigs2[iidd] ;
            @views jlms[ind] = @. hi[ind] *( mus[iidd] + sqrt(sigs2[iidd])* normpdf(
                                   mus[iidd]/sqrt(sigs2[iidd]))/normcdf(mus[iidd]/sqrt(sigs2[iidd]))   ) ;  
            @views jlms_direct[ind] = jlms[ind]
            @views jlms_indirect[ind] = jlms[ind] - jlms_direct[ind];
   
        end # for iidd=1:ID
       end # begin

       return jlms,jlms_direct,jlms_indirect
    end    
      
      
      
      
    
function counterfactindex(::Type{SSFWHET}, 
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any},dat::DataFrame)
    
    yvar = dat[:, _cfdicM[:depvar]]   
    xvar = dat[:, _cfdicM[:frontier]] 
    qvar = dat[:, _cfdicM[:hscale]]  

    envar = dat[:, _dicM[:envar]]   
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
    ivvar = dat[:, name_new_ivvar]   

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))
    IV  = convert(Array{Float64}, Matrix(ivvar))
    EN  = convert(Array{Float64}, Matrix(envar))


    jlms,jlms_direct,jlms_indirect = cfindexwhte(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function counterfactindex(::Type{SSFWHT}, 
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any},dat::DataFrame)

    yvar = dat[:, _cfdicM[:depvar]]   
    xvar = dat[:, _cfdicM[:frontier]] 
    qvar = dat[:, _cfdicM[:hscale]]  


    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))



    jlms,jlms_direct,jlms_indirect = cfindexwht(y, x, Q, PorC, pos, rho,  eigvalu,  rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end












 
function  cfindexwhhe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple,  rowIDT::Matrix{Any})
   
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]

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
 
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);


    @views invPi = 1.0 /σᵥ²  ;
    @floop begin
    @inbounds for iidd=1:ID 
         @views T = rowIDT[iidd,2];
         @views onecol = ones(T, 1);
         @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
         @views ind = rowIDT[iidd,1];
         @views his = IMT*(hi[ind]);
         @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
         @views sigs2[iidd] = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
         @views mus[iidd] = (μ/σᵤ² - (ϵ[ind]'*his*invPi))*sigs2[iidd] ;
         @views jlms[ind] = @. hi[ind] *( mus[iidd] + sqrt(sigs2[iidd])* normpdf.(
                                mus[iidd]/sqrt.(sigs2[iidd]))./normcdf.(mus[iidd]/sqrt.(sigs2[iidd]))   ) ;  
         @views jlms_direct[ind] = jlms[ind]
         @views jlms_indirect[ind] = jlms[ind] - jlms_direct[ind];

     end # for iidd=1:ID
    end # begin
   

    return jlms,jlms_direct,jlms_indirect
    end
    
    
   
   
   
function  cfindexwhh(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})


       β  = rho[1:pos.endx]
       τ  = rho[pos.begq:pos.endq]
       δ2 = rho[pos.begw]  
       γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    #    δ1 = rho[pos.begz]

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
 
       jlms = zeros(eltype(y),size(hi,1),1);
       jlms_direct = zeros(eltype(y),size(hi,1),1);
       jlms_indirect = zeros(eltype(y),size(hi,1),1);


       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for iidd=1:ID 
            # @views T = rowIDT[iidd,2];
            # @views onecol = ones(T, 1);
            # @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
            @views ind = rowIDT[iidd,1];
            @views his = sf_demean(hi[ind]);

            @views ϵs = ϵ[ind]   ;
            @views sigs2 = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
            @views mus = (μ/σᵤ² - (ϵs'*his*invPi))*sigs2 ;
            @views jlms[ind] = @. hi[ind] *( mus + sqrt(sigs2)* normpdf(
                                                    mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) ;  
            @views jlms_direct[ind] = jlms[ind]
            @views jlms_indirect[ind] = jlms[ind] - jlms_direct[ind];

        end # for iidd=1:ID
       end # begin

       return jlms,jlms_direct,jlms_indirect
    end    
      
      


      
      
    
function counterfactindex(::Type{SSFWHEH}, 
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any},dat::DataFrame)
    
    yvar = dat[:, _cfdicM[:depvar]]   
    xvar = dat[:, _cfdicM[:frontier]] 
    qvar = dat[:, _cfdicM[:hscale]]  
 
    envar = dat[:, _dicM[:envar]]   
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
    ivvar = dat[:, name_new_ivvar]   

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))
    IV  = convert(Array{Float64}, Matrix(ivvar))
    EN  = convert(Array{Float64}, Matrix(envar))


    jlms,jlms_direct,jlms_indirect = cfindexwhhe(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function counterfactindex(::Type{SSFWHH}, 
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any} , dat::DataFrame)

    yvar = dat[:, _cfdicM[:depvar]]   
    xvar = dat[:, _cfdicM[:frontier]] 
    qvar = dat[:, _cfdicM[:hscale]]  

    y  = convert(Array{Float64}, Matrix(yvar))
    x  = convert(Array{Float64}, Matrix(xvar))
    Q  = convert(Array{Float64}, Matrix(qvar))



    jlms,jlms_direct,jlms_indirect = cfindexwhh(y, x, Q, PorC, pos, rho,  eigvalu,  rowIDT )  

    return jlms,jlms_direct,jlms_indirect

