
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= 1*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] =Mgamma* hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    

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

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   

end





function jlmsbc(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z::Matrix,en,iv,
  PorC::Int64, num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any}) 

Wy = _dicM[:wy]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
    gammap = rho[pos.beggamma]
    gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
     dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)
if Wu!=Nothing 
if Wv!=Nothing #yuv
    jlms_, bc_  = jlmsbct_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else # yu
    jlms_, bc_  = jlmsbct_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
end    
else 
if Wv!=Nothing #yv
    jlms_, bc_  = jlmsbct_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
else #y
    jlms_, bc_  = jlmsbct_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
end
end
jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
else
if Wu!=Nothing 
if Wv!=Nothing #uv
    jlms_, bc_  = jlmsbct_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
else # u
    jlms_, bc_  = jlmsbct_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
end    
else 
if Wv!=Nothing #v
    jlms_, bc_  = jlmsbct_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
else # 
    jlms_, bc_  = jlmsbct_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
end
end
jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
end 

return jlms_df, bc_df  


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
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
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
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
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
ϵ = PorC*(y - x * β )
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] =Mgamma* hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
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
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
ϵ = PorC*(y - x * β )
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
ϵ = PorC*(y - x * β )
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
ϵ = PorC*(y - x * β )
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

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc       
end

function IrhoWratio(gamma::Float64, rowIDT::Matrix{Any} )
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



function jlmsbc(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]

     if Wy!=Nothing  # yuv
        gammap = rho[pos.beggamma]
        gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
         dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)

         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                 jlms_, bc_ = jlmsbcdt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms_, bc_ = jlmsbcdt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms_, bc_ = jlmsbcdt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms_, bc_ = jlmsbcdt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end
         jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms_, bc_  = jlmsbcdt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms_, bc_ = jlmsbcdt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #v
                jlms_, bc_  = jlmsbcdt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
             else # 
                jlms_, bc_  = jlmsbcdt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end
         jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
     end 
     
     return jlms_df, bc_df
 
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin

elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= 1*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] =Mgamma* hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc     
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

# @floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
            normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  

end # for ttt=1:T
# end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind];
@views sigs2 = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2 ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* 
        normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  


end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
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

@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc  
end





function jlmsbc(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, en,iv,
  PorC::Int64,  num::NamedTuple, pos::NamedTuple, rho::Array{Float64, 1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]
  gammap = rho[pos.beggamma]
  gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
   dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)
    if Wy!=Nothing  # yuv
        if Wu!=Nothing 
            if Wv!=Nothing #yuv
                jlms_, bc_ = jlmsbch_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else # yu
                jlms_, bc_ = jlmsbch_yu( y, x, Q, w, v, z, Wy, Wu, PorC, pos, rho,  eigvalu, rowIDT  )
            end    
        else 
            if Wv!=Nothing #yv
                jlms_, bc_ = jlmsbch_yv(y, x, Q, w, v, z, Wy, Wv, PorC, pos, rho,  eigvalu, rowIDT )
            else #y
                jlms_, bc_ = jlmsbch_y(y, x, Q, w, v, z, Wy, PorC, pos, rho,  eigvalu, rowIDT )  
            end
        end
        jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
        bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
    else
        if Wu!=Nothing 
            if Wv!=Nothing #uv
                jlms_, bc_ = jlmsbch_uv(y, x, Q, w, v, z, Wu, Wv, PorC, pos, rho,  eigvalu, rowIDT  )
            else # u
                jlms_, bc_ = jlmsbch_u(y, x, Q, w, v, z, Wu, PorC, pos, rho,  eigvalu, rowIDT  ) 
            end    
        else 
            if Wv!=Nothing #v
                jlms_, bc_ = jlmsbch_v(y, x, Q, w, v, z, Wv,PorC, pos, rho,  eigvalu, rowIDT )
            else # 
                jlms_, bc_ = jlmsbch_( y, x, Q, w, v, z, PorC, pos, rho,  eigvalu, rowIDT  )  
            end
        end
        jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
        bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
    end 
    
    return jlms_df, bc_df  

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
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views bc[ind] = Mgamma * hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
  @views jlms[ind] = Mgamma * hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
  end # begin
end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views invPi = (Pi)\I(N);
  @views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views bc[ind] = Mgamma * hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
  @views jlms[ind] = Mgamma * hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
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
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[1])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma* hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = (I(N)-tau*Wu[ttt])\I(N);
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc            
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
ϵ = PorC*(y - x * β )
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
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
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
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] =Mgamma* hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc       
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
ϵ = PorC*(y - x * β )
T = size(rowIDT,1)

sigs2 = zeros(eltype(y),T,1);
mus = zeros(eltype(y),T,1);
bc = zeros(eltype(y),size(hi,1),1);
jlms = zeros(eltype(y),size(hi,1),1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[1])\I(N)

@floop begin
@inbounds for ttt=1:T 
@views ind = rowIDT[ttt,1];
@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = Mgamma*hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = Mgamma*hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
elseif length(Wy)>1
@floop begin
@inbounds for ttt=1:T  
@views N = rowIDT[1,2];
@views Mtau = 1;
@views invPi = 1/σᵥ²*(I(N));
@views Mgamma = (I(N)-gamma*Wy[ttt])\I(N)

@views hi[ind]= Mtau*hi[ind];
@views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
@views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
@views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
@views bc[ind] = hi[ind] .*( mus[ttt] + sqrt(sigs2[ttt])* normpdf(mus[ttt]/sqrt(sigs2[ttt]))./normcdf(mus[ttt]/sqrt(sigs2[ttt]))   ) ;  
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  
end # for ttt=1:T
end # begin
end  #    if length(Wy)==1 
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc        
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
ϵ = PorC*(y - x * β )
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc    
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
ϵ = PorC*(y - x * β )
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
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc   
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
ϵ = PorC*(y - x * β )
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
@views jlms[ind] = hi[ind] .* ( mus[ttt] + normcdf(mus[ttt]/sqrt(sigs2[ttt])) * sqrt(sigs2[ttt]) / normcdf(mus[ttt]/sqrt(sigs2[ttt])) )  ;
end # for ttt=1:T
end # begin
@views TE_bc = exp.(-bc);
@views TE_jlms = exp.(-jlms);
return TE_jlms, TE_bc      
end




function jlmsbc(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
   gammap = rho[pos.beggamma]
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
    dire_ratio,indire_ratio = IrhoWratio(gamma, rowIDT)
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                jlms_, bc_ = jlmsbcdh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                jlms_, bc_ = jlmsbcdh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                jlms_, bc_ = jlmsbcdh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                jlms_, bc_ = jlmsbcdh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end
         jlms_df = DataFrame(hcat(dire_ratio*jlms_,indire_ratio*jlms_), [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(hcat(dire_ratio*bc_,indire_ratio*bc_), [:dire_bc, :indire_bc])
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                jlms_, bc_ = jlmsbcdh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                jlms_, bc_ = jlmsbcdh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                jlms_, bc_ = jlmsbcdh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, num, pos, rho,  eigvalu, rowIDT )
             else # 
                jlms_, bc_ = jlmsbcdh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end
         jlms_df = DataFrame(jlms_, [:dire_jlms, :indire_jlms])
         bc_df = DataFrame(bc_, [:dire_bc, :indire_bc])
     end 
     
     return jlms_df, bc_df  
 
 end
 

