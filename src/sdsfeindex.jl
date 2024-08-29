
### get barely inefficiency  code 0 i.e. hsale * ut
### get  inefficiency  code 1 i.e. (I-wtau)^-1 * hsale * ut   if has
### get  inefficiency  code 2 i.e. (I-wrho)^-1 * (I-wtau)^-1 * hsale * ut,decompostion at (I-wrho)^-1 * (I-wtau)^-1 
#########################################
####        JLMS and BC index        ####
#########################################

#? --------------- Truncated Normal --------------


function  jlmsbct_yuv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
# @views jlms[ind] = ( hi[ind] .* ( mus[ttt] + normpdf(mus[ttt]/sqrt(sigs2[ttt])) * 
#         sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) ))

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] =   (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms   
end





function  jlmsbct_yu0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
 
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0/σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  
 

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
 
return jlms 
end








function  jlmsbct_yv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbct_y0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+ 1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms    
end









function  jlmsbct_uv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms 
end







function  jlmsbct_u0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms   

end






function  jlmsbct_v0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms 
end


function  jlmsbct_0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin

return jlms  

end





function jlmsbc0(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms  = jlmsbct_yuv0( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else # yu
    jlms  = jlmsbct_yu0( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms  = jlmsbct_yv0(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else #y
    jlms  = jlmsbct_y0(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
end
end

else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms  = jlmsbct_uv0(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
else # u
    jlms  = jlmsbct_u0(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms  = jlmsbct_v0(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
else # 
    jlms  = jlmsbct_0( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
end
end

end 

return jlms  


end




function  jlmsbcdt_yuv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms      
end





function  jlmsbcdt_yu0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0 /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms   
end









function  jlmsbcdt_yv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms 
end










function  jlmsbcdt_y0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms   
end










function  jlmsbcdt_uv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdt_u0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms     
end













function  jlmsbcdt_v0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms  
end











function  jlmsbcdt_0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin

return jlms 
end



function jlmsbc0(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]

     if Wy!=Nothing  # yuv

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms = jlmsbcdt_yuv0( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms = jlmsbcdt_yu0( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms = jlmsbcdt_yv0(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms = jlmsbcdt_y0(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms  = jlmsbcdt_uv0(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms = jlmsbcdt_u0(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms  = jlmsbcdt_v0(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
             else # 
                jlms  = jlmsbcdt_0( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end

     end 
     
     return jlms
 
 end

 




 function  jlmsbch_yuv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.( Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
# println(ind)
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] .= (hi[ind] .* ( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


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

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .* ( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms  
end





function  jlmsbch_yu0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end








function  jlmsbch_yv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms 
end










function  jlmsbch_y0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end









function  jlmsbch_uv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms      
end







function  jlmsbch_u0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end






function  jlmsbch_v0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2 ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms
end


function  jlmsbch_0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin

return jlms
end





function jlmsbc0(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]
    if Wy!=Nothing  # yuv
        if Wu!=Nothing 
            if Wv!=Nothing #yuv
                jlms = jlmsbch_yuv0( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else # yu
                jlms = jlmsbch_yu0( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
            end    
        else 
            if Wv!=Nothing #yv
                jlms = jlmsbch_yv0(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else #y
                jlms = jlmsbch_y0(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
            end
        end

    else
        if Wu!=Nothing 
            if Wv!=Nothing #uv
                jlms = jlmsbch_uv0(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
            else # u
                jlms = jlmsbch_u0(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms = jlmsbch_v0(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
            else # 
                jlms = jlmsbch_0( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
            end
        end

    end 
    
    return jlms

  end







function  jlmsbcdh_yuv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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


  @floop begin
  @inbounds for ttt=1:T 
  @views ind = rowIDT[ttt,1];
  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

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

  @views hitau[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
  @views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 

return jlms
end





function  jlmsbcdh_yu0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms       
end









function  jlmsbcdh_yv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms      
end










function  jlmsbcdh_y0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views invPi = 1.0 /σᵥ²*(I(N));

@views ind = rowIDT[ttt,1];
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbcdh_uv0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms     
end













function  jlmsbcdh_u0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdh_v0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms  
end











function  jlmsbcdh_0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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
@views hitau[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hitau[ind]'*invPi*hitau[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hitau[ind])*sigs2[ttt] ;
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin

return jlms
end




function jlmsbc0(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms = jlmsbcdh_yuv0( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms = jlmsbcdh_yu0( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms = jlmsbcdh_yv0(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms = jlmsbcdh_y0(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms = jlmsbcdh_uv0(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms = jlmsbcdh_u0(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms = jlmsbcdh_v0(y, x, Q, w, v, z, EN, IV, Wv,PorC, num,  pos, rho,  eigvalu, rowIDT )
             else # 
                jlms = jlmsbcdh_0( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end

     end 
     
     return jlms
 
 end
 



function  jlmsbckute0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
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


    @views invPi = 1.0 /σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 = @. 1.0  / (hi^2 *invPi + 1.0 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

    @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

elseif length(Wy)>1

    @views Wyt = BlockDiagonal([Wy...])
    @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

    end # for ttt=1:T

    @views invPi = 1.0/σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 =@. 1.0 / (hi^2 *invPi + 1 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

    @views jlms = @. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

end  #    if length(Wy)==1 

return jlms
end




function  jlmsbckut0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
  
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
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
 

       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 


   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
       @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

       end # for ttt=1:T
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma.*Wyt*y  ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1 /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
  
   end  #    if length(Wy)==1 

   return jlms
   end
   
   




function jlmsbc0(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckute0(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms
 
 end

 function jlmsbc0(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckut0(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  
    
    return jlms
 
 end


 
function  jlmsbckuhe0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
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

       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

   
   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
       @inbounds for ttt=1:T  
        @views N = rowIDT[ttt,2];

       end # for ttt=1:T
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

   end  #    if length(Wy)==1 

   return jlms
   end
   
   
   
   
   function  jlmsbckuh0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
      
      
       β  = rho[1:pos.endx]
      τ  = rho[pos.begq:pos.endq]
     
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

      
          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

          @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

      
      elseif length(Wy)>1
      
        @views Wyt = BlockDiagonal([Wy...])
          @inbounds for ttt=1:T  
            @views N = rowIDT[ttt,2];

          end # for ttt=1:T
      
          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

          @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 


        end  #    if length(Wy)==1 

      return jlms   
      end
      
      
   
   
   
   
   function jlmsbc0(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
       PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    
        Wy = _dicM[:wy]   
        jlms = jlmsbckuhe0(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

       
        return jlms
    
    end



       
   function jlmsbc0(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckuh0(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  

     return jlms
 
 end



 
function  jlmsbckkte0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
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
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
    end # for idid=1:ID
    end # begin
   
    return jlms
    end
    
    
   
   
   
function  jlmsbckkt0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       

       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for idid=1:ID 

           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
       end # for idid=1:ID
       end # begin

       return jlms
    end    
      
      
    
function jlmsbc0(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkte0(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms

end

function jlmsbc0(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkt0(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms

end
   
   

 
function  jlmsbckkhe0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
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
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
    end # for idid=1:ID
    end # begin

    return jlms 
    end
    
    
   
   
   
function  jlmsbckkh0(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       

       @views invPi = 1.0 /σᵥ²  ;

       @floop begin
       @inbounds for idid=1:ID 
           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
       end # for idid=1:ID
       end # begin

       return jlms  
    end    
      
      
    
function jlmsbc0(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkhe0(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms

end

function jlmsbc0(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkh0(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms

end




### get barely inefficiency  code 0 i.e. hsale * ut
### get  inefficiency  code 1 i.e. (I-wtau)^-1 * hsale * ut   if has
### get  inefficiency  code 2 i.e. (I-wrho)^-1 * (I-wtau)^-1 * hsale * ut,decompostion at (I-wrho)^-1 * (I-wtau)^-1 
#########################################
####        JLMS and BC index        ####
#########################################

#? --------------- Truncated Normal --------------


function  jlmsbct_yuv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  


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

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms 
end





function  jlmsbct_yu1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] =   (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms 
end








function  jlmsbct_yv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] =   (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms 
end










function  jlmsbct_y1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0 /(hi[ind]'*invPi*hi[ind]+ 1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] =  (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   )) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end









function  jlmsbct_uv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end







function  jlmsbct_u1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms

end






function  jlmsbct_v1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms
end


function  jlmsbct_1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin

return jlms

end





function jlmsbc1(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms = jlmsbct_yuv1( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else # yu
    jlms  = jlmsbct_yu1( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms  = jlmsbct_yv1(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else #y
    jlms  = jlmsbct_y1(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
end
end

else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms = jlmsbct_uv1(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
else # u
    jlms  = jlmsbct_u1(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms  = jlmsbct_v1(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
else # 
    jlms = jlmsbct_1( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
end
end

end 

return jlms


end




function  jlmsbcdt_yuv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

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

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end





function  jlmsbcdt_yu1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0 /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end









function  jlmsbcdt_yv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbcdt_y1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) );  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbcdt_uv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdt_u1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdt_v1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms
end











function  jlmsbcdt_1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
return jlms 
end





function jlmsbc1(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]

     if Wy!=Nothing  # yuv

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms = jlmsbcdt_yuv1( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms = jlmsbcdt_yu1( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms = jlmsbcdt_yv1(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms = jlmsbcdt_y1(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms  = jlmsbcdt_uv1(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms = jlmsbcdt_u1(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms  = jlmsbcdt_v1(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
             else # 
                jlms  = jlmsbcdt_1( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end

     end 
     
     return jlms
 
 end

 




 function  jlmsbch_yuv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

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

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end





function  jlmsbch_yu1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end








function  jlmsbch_yv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbch_y1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
    @views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views jlms[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end









function  jlmsbch_uv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms      
end







function  jlmsbch_u1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end






function  jlmsbch_v1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms
end


function  jlmsbch_1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  

end # for ttt=1:T
end # begin

return jlms
end





function jlmsbc1(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]
    if Wy!=Nothing  # yuv
        if Wu!=Nothing 
            if Wv!=Nothing #yuv
                jlms = jlmsbch_yuv1( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else # yu
                jlms = jlmsbch_yu1( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
            end    
        else 
            if Wv!=Nothing #yv
                jlms = jlmsbch_yv1(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else #y
                jlms = jlmsbch_y1(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
            end
        end

    else
        if Wu!=Nothing 
            if Wv!=Nothing #uv
                jlms = jlmsbch_uv1(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
            else # u
                jlms = jlmsbch_u1(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms = jlmsbch_v1(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
            else # 
                jlms = jlmsbch_1( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
            end
        end

    end 
    
    return jlms

  end







function  jlmsbcdh_yuv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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


  @floop begin
  @inbounds for ttt=1:T 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

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

  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 

return jlms
end





function  jlmsbcdh_yu1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1.0 /σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta);
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms   
end









function  jlmsbcdh_yv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbcdh_y1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = Mgamma*(hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
    @views N = rowIDT[ttt,2];
@views invPi = 1.0 /σᵥ²*(I(N));

@views ind = rowIDT[ttt,1];
@views hi[ind]= hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1.0  /(hi[ind]'*invPi*hi[ind]+1.0 /σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views jlms[ind] = (hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))  ) ) ;  

end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 

return jlms
end










function  jlmsbcdh_uv1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdh_u1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 

return jlms
end













function  jlmsbcdh_v1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 

return jlms
end











function  jlmsbcdh_1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
@views jlms[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
end # for ttt=1:T
end # begin

return jlms
end




function jlmsbc1(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms = jlmsbcdh_yuv1( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms = jlmsbcdh_yu1( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms = jlmsbcdh_yv1(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms = jlmsbcdh_y1(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms = jlmsbcdh_uv1(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms = jlmsbcdh_u1(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms = jlmsbcdh_v1(y, x, Q, w, v, z, EN, IV, Wv,PorC, num,  pos, rho,  eigvalu, rowIDT )
             else # 
                jlms = jlmsbcdh_1( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end

     end 
     
     return jlms_df, bc_df  
 
 end
 

 

function  jlmsbckute1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
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


    @views invPi = 1.0 /σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 = @. 1.0  / (hi^2 *invPi + 1.0 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

    @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

elseif length(Wy)>1

    @views Wyt = BlockDiagonal([Wy...])


    @views invPi = 1.0/σᵥ²;
    @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
    @views sigs2 =@. 1.0 / (hi^2 *invPi + 1 /σᵤ²) ;
    @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

    @views jlms = @. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
end




function  jlmsbckut1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
  
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
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 


   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma.*Wyt*y  ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1 /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
   end  #    if length(Wy)==1 

   return jlms
   end
   
   




function jlmsbc1(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckute1(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

     return jlms
 
 end

 function jlmsbc1(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckut1(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  

     return jlms
 
 end


 
function  jlmsbckuhe1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
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

       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

   
   elseif length(Wy)>1
   
       @views Wyt = BlockDiagonal([Wy...])
   
       @views invPi = 1.0 /σᵥ²;
       @views ϵ  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
       @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
       @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

       @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
   end  #    if length(Wy)==1 

   return jlms
   end
   
   
   
   
   function  jlmsbckuh1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
      
      
       β  = rho[1:pos.endx]
      τ  = rho[pos.begq:pos.endq]
     
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

      
          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;

          @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 
      
      elseif length(Wy)>1
      
        @views Wyt = BlockDiagonal([Wy...])

          @views invPi = 1.0 /σᵥ²;
          @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
          @views sigs2 =@. 1.0  / (hi^2 *invPi + 1.0  /σᵤ²) ;
          @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
          @views jlms =@. hi *( mus + sqrt(sigs2) * normpdf(mus/sqrt(sigs2))/normcdf(mus/sqrt(sigs2))   ) 

        end  #    if length(Wy)==1 

      return jlms
      end
      
      
   
   
   
   
   function jlmsbc1(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
       PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    
       Wy = _dicM[:wy]
        jlms = jlmsbckuhe1(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

        return jlms
    
    end



       
   function jlmsbc1(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms = jlmsbckuh1(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  

     return jlms
 
 end



 
function  jlmsbckkte1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
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
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
    end # for idid=1:ID
    end # begin
   

    return jlms
    end
    
    
   
   
   
function  jlmsbckkt1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       

       @views invPi = 1.0 /σᵥ²  ;
       @floop begin
       @inbounds for idid=1:ID 

           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
       end # for idid=1:ID
       end # begin

       return jlms
    end    
      
      
    
function jlmsbc1(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkte1(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms

end

function jlmsbc1(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkt1(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms

end
   
   

 
function  jlmsbckkhe1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
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
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    
    @views invPi = 1.0 /σᵥ²  ;
    
    @floop begin
    @inbounds for idid=1:ID 

        @views ind = rowIDT[idid,1];
        @views ϵ[ind] = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
        @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
        @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
        @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
    end # for idid=1:ID
    end # begin
   
    return jlms  
    end
    
    
   
   
   
function  jlmsbckkh1(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
       jlms = zeros(eltype(y),size(hi,1),1);
       

       @views invPi = 1.0 /σᵥ²  ;

       @floop begin
       @inbounds for idid=1:ID 
           @views ind = rowIDT[idid,1];
           @views ϵ[ind] = ϵ[ind]  ;
           @views sigs2[idid] = 1.0 /(hi[ind]'*hi[ind]*invPi  +1.0/σᵤ²);
           @views mus[idid] = (μ/σᵤ² - ϵ[ind]'*hi[ind] *invPi )*sigs2[idid] ;
           @views jlms[ind] = hi[ind] .*( mus[idid] + sqrt(sigs2[idid])* normpdf(mus[idid]/sqrt(sigs2[idid]))./normcdf(mus[idid]/sqrt(sigs2[idid]))   ) ;  
       end # for idid=1:ID
       end # begin

       return jlms
    end    
      
      
    
function jlmsbc1(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms_, bc_ = jlmsbckkhe1(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms_, bc_

end

function jlmsbc1(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms = jlmsbckkh1(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms

end







### get barely inefficiency  code 0 i.e. hsale * ut
### get  inefficiency  code 1 i.e. (I-wtau)^-1 * hsale * ut   if has
### get  inefficiency  code 2 i.e. (I-wrho)^-1 * (I-wtau)^-1 * hsale * ut,decompostion at (I-wrho)^-1 * (I-wtau)^-1 
#########################################
####        JLMS and BC index        ####
#########################################

#? --------------- Truncated Normal --------------


function  jlmsbct_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
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
    hitau = zeros(eltype(hi),size(hi,1),1);
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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
    





function  jlmsbct_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    
    taup = rho[pos.begtau]
    tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   =δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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
    







function  jlmsbct_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    
    rhomyp = rho[pos.begrho]
    rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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









function  jlmsbct_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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
    








function  jlmsbct_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
   δ2 = rho[pos.begw]  
   γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[pos.begz]
   
   taup = rho[pos.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
   
   rhomyp = rho[pos.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));
   
   hi  = exp.(Q*τ)
   hitau = zeros(eltype(hi),size(hi,1),1);
   
   σᵤ²= exp(δ2) 
   σᵥ² = exp(γ)  
   μ   = δ1
   
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   
   sigs2 = zeros(eltype(y),T,1);
   mus = zeros(eltype(y),T,1);
   bc = zeros(eltype(y),size(hi,1),1);
   jlms = zeros(eltype(y),size(hi,1),1);
   jlms_direct = zeros(eltype(y),size(hi,1),1);
   jlms_indirect = zeros(eltype(y),size(hi,1),1);
   
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
   






function  jlmsbct_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    
    
    taup = rho[pos.begtau]
    tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
    
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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
    






function  jlmsbct_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    
    
    rhomyp = rho[pos.begrho]
    rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
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
    


function  jlmsbct_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    β  = rho[1:pos.endx]
    τ  = rho[pos.begq:pos.endq]
    δ2 = rho[pos.begw]  
    γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[pos.begz]
    
    
    hi  = exp.(Q*τ)
    hitau = zeros(eltype(hi),size(hi,1),1);
    
    σᵤ²= exp(δ2) 
    σᵥ² = exp(γ)  
    μ   = δ1
    
    ϵ = PorC*(y - x * β)
    T = size(rowIDT,1)
    
    sigs2 = zeros(eltype(y),T,1);
    mus = zeros(eltype(y),T,1);
    bc = zeros(eltype(y),size(hi,1),1);
    jlms = zeros(eltype(y),size(hi,1),1);
    jlms_direct = zeros(eltype(y),size(hi,1),1);
    jlms_indirect = zeros(eltype(y),size(hi,1),1);
    
    
    
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





function jlmsbc(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
if Wu!=Nothing 
    taup = rho[pos.begtau]
    tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
if Wv!=Nothing #yuv
    jlms,jlms_direct,jlms_indirect  = jlmsbct_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else # yu
    jlms,jlms_direct,jlms_indirect  = jlmsbct_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms,jlms_direct,jlms_indirect  = jlmsbct_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else #y
    jlms,jlms_direct,jlms_indirect  = jlmsbct_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
end
end
jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
else
if Wu!=Nothing 
    taup = rho[pos.begtau]
    tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
if Wv!=Nothing #uv
    jlms,jlms_direct,jlms_indirect  = jlmsbct_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
else # u
    jlms,jlms_direct,jlms_indirect  = jlmsbct_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms,jlms_direct,jlms_indirect = jlmsbct_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
else # 
    jlms,jlms_direct,jlms_indirect = jlmsbct_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
end
end
end 

return jlms,jlms_direct,jlms_indirect  


end




function  jlmsbcdt_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
 
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
 





function  jlmsbcdt_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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
 








function  jlmsbcdt_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
 σᵤ²= exp(δ2) 
 σᵤ= exp(0.5*δ2) 
 σᵥ² = exp(γ)  
 σᵥ = exp(0.5*γ)    
 μ   =δ1
 ϵ = PorC*(y - x * β )
 T = size(rowIDT,1)
 
 sigs2 = zeros(eltype(y),T,1);
 mus = zeros(eltype(y),T,1);
 bc = zeros(eltype(y),size(hi,1),1);
 jlms = zeros(eltype(y),size(hi,1),1);
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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










function  jlmsbcdt_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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









function  jlmsbcdt_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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
 












function  jlmsbcdt_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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
 
 












function  jlmsbcdt_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
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
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
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











function  jlmsbcdt_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 β  = rho[1:pos.endx]
 τ  = rho[pos.begq:pos.endq]
 phi = rho[pos.begphi:pos.endphi]
 
 nofiv=num.nofphi/num.nofeta
 phi = reshape(phi,:,num.nofeta)
 eps = EN-IV*phi
 eta = rho[pos.begeta:pos.endeta]
 
 δ2 = rho[pos.begw]  
 γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
 δ1 = rho[pos.begz]
 
 hi  = exp.(Q*τ)
 hitau = zeros(eltype(hi),size(hi,1),1);
 
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
 jlms_direct = zeros(eltype(y),size(hi,1),1);
 jlms_indirect = zeros(eltype(y),size(hi,1),1);
 
 
 
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
 




function jlmsbc(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]

     if Wy!=Nothing  # yuv
        gammap = rho[pos.beggamma]
        gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect  = jlmsbcdt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect  = jlmsbcdt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
             else # 
                jlms,jlms_direct,jlms_indirect = jlmsbcdt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end

     end 
     
     return jlms,jlms_direct,jlms_indirect
 
 end

 




 function  jlmsbch_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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





function  jlmsbch_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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








function  jlmsbch_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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










function  jlmsbch_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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









function  jlmsbch_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
   Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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







function  jlmsbch_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0
ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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






function  jlmsbch_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);

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





function  jlmsbch_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0.0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);
jlms_direct = zeros(eltype(y),size(hi,1),1);
jlms_indirect = zeros(eltype(y),size(hi,1),1);



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





function jlmsbc(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]



    if Wy!=Nothing  # yuv
        gammap = rho[pos.beggamma]
        gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
        if Wu!=Nothing 
            taup = rho[pos.begtau]
            tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
            if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = jlmsbch_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else # yu
                jlms,jlms_direct,jlms_indirect = jlmsbch_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
  
            end    
        else 
            if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = jlmsbch_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else #y
                jlms,jlms_direct,jlms_indirect = jlmsbch_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
            end
        end

    else
        if Wu!=Nothing 
            taup = rho[pos.begtau]
            tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));
            if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect = jlmsbch_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
            else # u
                jlms,jlms_direct,jlms_indirect = jlmsbch_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect = jlmsbch_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
            else # 
                jlms,jlms_direct,jlms_indirect = jlmsbch_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
            end
        end

    end 
    
    return jlms,jlms_direct,jlms_indirect

  end







function  jlmsbcdh_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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





function  jlmsbcdh_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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









function  jlmsbcdh_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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










function  jlmsbcdh_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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










function  jlmsbcdh_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix, Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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













function  jlmsbcdh_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wu::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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













function  jlmsbcdh_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
   Wv::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
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
hitau = zeros(eltype(hi),size(hi,1),1);

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











function  jlmsbcdh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
 PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
phi = rho[pos.begphi:pos.endphi]

nofiv=num.nofphi/num.nofeta
phi = reshape(phi,:,num.nofeta)
eps = EN-IV*phi
eta = rho[pos.begeta:pos.endeta]

δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

hi  = exp.(Q*τ)
hitau = zeros(eltype(hi),size(hi,1),1);

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




function jlmsbc(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]

     if Wy!=Nothing  # yuv
        gammap = rho[pos.beggamma]
        gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end

     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num,  pos, rho,  eigvalu, rowIDT )
             else # 
                jlms,jlms_direct,jlms_indirect = jlmsbcdh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
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
#     dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)
     
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
#     dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)
     
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
 

function  jlmsbckute(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
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




function  jlmsbckut(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
    PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
  
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
   
   




function jlmsbc(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]

    jlms,jlms_direct,jlms_indirect = jlmsbckute(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

     return jlms,jlms_direct,jlms_indirect
 
 end

 function jlmsbc(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]

    jlms,jlms_direct,jlms_indirect = jlmsbckut(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  
    
     return jlms,jlms_direct,jlms_indirect
 
 end


 
function  jlmsbckuhe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, Wy::Matrix,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
   
    β  = rho[1:pos.endx]
   τ  = rho[pos.begq:pos.endq]
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
   
   
   
   
   function  jlmsbckuh(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   Wy::Matrix,
       PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
      
      
       β  = rho[1:pos.endx]
      τ  = rho[pos.begq:pos.endq]
     
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
      
      
   
   
   
   
   function jlmsbc(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
       PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
    
       Wy = _dicM[:wy]
       jlms,jlms_direct,jlms_indirect = jlmsbckuhe(y, x, Q,  EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  

        return jlms,jlms_direct,jlms_indirect
    
    end



       
   function jlmsbc(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
    Wy = _dicM[:wy]
    jlms,jlms_direct,jlms_indirect = jlmsbckuh(y, x, Q,  Wy, PorC, pos, rho,  eigvalu, rowIDT )  

     return jlms,jlms_direct,jlms_indirect
 
 end



 
function  jlmsbckkte(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
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
    bc = zeros(eltype(y),size(hi,1),1);
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
    
    
   
   
   
function  jlmsbckkt(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
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
      
      
    
function jlmsbc(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms,jlms_direct,jlms_indirect = jlmsbckkte(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function jlmsbc(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms,jlms_direct,jlms_indirect = jlmsbckkt(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end
   
   

 
function  jlmsbckkhe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN::Matrix, IV::Matrix, 
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
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
    bc = zeros(eltype(y),size(hi,1),1);
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
    
    
   
   
   
function  jlmsbckkh(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,   
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
       bc = zeros(eltype(y),size(hi,1),1);
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
      
      
    
function jlmsbc(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms,jlms_direct,jlms_indirect = jlmsbckkhe(y, x, Q, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end

function jlmsbc(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN, IV,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

    jlms,jlms_direct,jlms_indirect = jlmsbckkh(y, x, Q, PorC, pos, rho,  eigvalu, rowIDT )  

    return jlms,jlms_direct,jlms_indirect

end