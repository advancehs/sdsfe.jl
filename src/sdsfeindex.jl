
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end





function  jlmsbct_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

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

@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end








function  jlmsbct_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
@views hi[ind]= 1*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1 ;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end










function  jlmsbct_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

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
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));


# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end









function  jlmsbct_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
     Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc  

end







function  jlmsbct_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


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
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc  

end






function  jlmsbct_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
return jlms, bc  

end


function  jlmsbct_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
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


@views N = rowIDT[1,2];
@views Mtau = 1 ;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin

return jlms, bc  

end





function jlmsbc(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
    PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms, bc = jlmsbct_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else # yu
    jlms, bc = jlmsbct_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms, bc = jlmsbct_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else #y
    jlms, bc = jlmsbct_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
end
end
else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms, bc = jlmsbct_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
else # u
    jlms, bc = jlmsbct_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms, bc = jlmsbct_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
else # 
    jlms, bc = jlmsbct_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
end
end
end 

return jlms, bc  


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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
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
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
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
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc    
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
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc    
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
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
return jlms, bc    
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
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = δ1
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);


@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

return jlms, bc    
end





function jlmsbc(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
      PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
     Wy = _dicM[:wy]
     Wu = _dicM[:wu]
     Wv = _dicM[:wv]
       
       if Wy!=Nothing  # yuv
           if Wu!=Nothing 
               if Wv!=Nothing #yuv
                   jlms, bc = jlmsbcdt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
               else # yu
                   jlms, bc = jlmsbcdt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
               end    
           else 
               if Wv!=Nothing #yv
                   jlms, bc = jlmsbcdt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
               else #y
                   jlms, bc = jlmsbcdt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
               end
           end
       else
           if Wu!=Nothing 
               if Wv!=Nothing #uv
                   jlms, bc = jlmsbcdt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
               else # u
                   jlms, bc = jlmsbcdt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
               end    
           else 
               if Wv!=Nothing #v
                   jlms, bc = jlmsbcdt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
               else # 
                   jlms, bc = jlmsbcdt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
               end
           end
       end 
       
       return jlms, bc  
   
   end

   




   function  jlmsbch_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

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
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end





function  jlmsbch_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);

@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end








function  jlmsbch_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

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
@views hi[ind]= 1*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1 ;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end










function  jlmsbch_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wy::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]
gammap = rho[pos.beggamma]
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));


# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc  

end









function  jlmsbch_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
     Wu::Matrix, Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

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
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc  

end







function  jlmsbch_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wu::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0
ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc  

end






function  jlmsbch_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      Wv::Matrix, PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wv[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
          normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
return jlms, bc  

end


function  jlmsbch_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z,
      PorC::Int64,  pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
β  = rho[1:pos.endx]
τ  = rho[pos.begq:pos.endq]
δ2 = rho[pos.begw]  
γ  = rho[pos.begv]  # May rho[po.begw : po.endw][1]
# δ1 = rho[pos.begz]


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵥ² = exp(γ)  
μ   = 0

ϵ = PorC*(y - x * β)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);


@views N = rowIDT[1,2];
@views Mtau = 1 ;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
              normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
end # begin

return jlms, bc  

end





function jlmsbc(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
    PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
  
    Wy = _dicM[:wy]
    Wu = _dicM[:wu]
    Wv = _dicM[:wv]
      
      if Wy!=Nothing  # yuv
          if Wu!=Nothing 
              if Wv!=Nothing #yuv
                  jlms, bc = jlmsbch_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
              else # yu
                  jlms, bc = jlmsbch_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
              end    
          else 
              if Wv!=Nothing #yv
                  jlms, bc = jlmsbch_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
              else #y
                  jlms, bc = jlmsbch_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
              end
          end
      else
          if Wu!=Nothing 
              if Wv!=Nothing #uv
                  jlms, bc = jlmsbch_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
              else # u
                  jlms, bc = jlmsbch_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
              end    
          else 
              if Wv!=Nothing #v
                  jlms, bc = jlmsbch_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
              else # 
                  jlms, bc = jlmsbch_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
              end
          end
      end 
      
      return jlms, bc  
  
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
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
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

taup = rho[pos.begtau]
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
return jlms, bc    
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
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

rhomyp = rho[pos.begrho]
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
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
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc    
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
tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wu)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wu)==1 
return jlms, bc    
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
rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

hi  = exp.(Q*τ)
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wv)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
@views Pi = σᵥ²*(Mrho*Mrho');
@views invPi = (Pi)\I(N);

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wv)==1 
return jlms, bc    
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
σᵤ²= exp(δ2) 
σᵤ= exp(0.5*δ2) 
σᵥ² = exp(γ)  
σᵥ = exp(0.5*γ)    
μ   = 0
ϵ = PorC*(y - x * β-eps*eta)
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);


@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

return jlms, bc    
end




function jlmsbc(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
      PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
   
     Wy = _dicM[:wy]
     Wu = _dicM[:wu]
     Wv = _dicM[:wv]
       
       if Wy!=Nothing  # yuv
           if Wu!=Nothing 
               if Wv!=Nothing #yuv
                   jlms, bc = jlmsbcdh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
               else # yu
                   jlms, bc = jlmsbcdh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
               end    
           else 
               if Wv!=Nothing #yv
                   jlms, bc = jlmsbcdh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
               else #y
                   jlms, bc = jlmsbcdh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
               end
           end
       else
           if Wu!=Nothing 
               if Wv!=Nothing #uv
                   jlms, bc = jlmsbcdh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
               else # u
                   jlms, bc = jlmsbcdh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
               end    
           else 
               if Wv!=Nothing #v
                   jlms, bc = jlmsbcdh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, num, pos, rho,  eigvalu, rowIDT )
               else # 
                   jlms, bc = jlmsbcdh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
               end
           end
       end 
       
       return jlms, bc  
   
   end
   

