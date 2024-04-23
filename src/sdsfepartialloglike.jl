
#########################################
####        JLMS and BC index        ####
#########################################

#? --------------- Truncated Normal --------------



function  prtlloglikedh_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);

  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end




function  prtlloglikedh_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
  
  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind]  ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end







function  prtlloglikedh_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,  Wv::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = 1 ;
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);


  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = 1;
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end









function  prtlloglikedh_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = 1;
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);

#   @views Mgamma = (I(N)-gamma*Wy[1])\I(N)


  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = 1;
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind]  ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end








function  prtlloglikedh_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);

  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end










function  prtlloglikedh_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

 @views N = rowIDT[1,2];
 @views Mtau = (I(N)-tau*Wu[1])\I(N);
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);


 @floop begin
 @inbounds for ttt=1:T 
   @views ind = rowIDT[ttt,1];
   @views hi[ind]= Mtau*hi[ind];
   @views ϵ[ind] = ϵ[ind];
   @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
   @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
   @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
   @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
 end # begin
 end # for ttt=1:T
elseif length(Wy)>1
 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind]   ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end









function  prtlloglikedh_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

 @views N = rowIDT[1,2];
 @views Mtau = 1 ;
 @views Mrho = (I(N)-rhomy*Wv[1])\I(N);
 @views Pi = σᵥ²*(Mrho*Mrho');
 @views lndetPi = log(det(Pi));
 @views invPi = (Pi)\I(N);

 @floop begin
 @inbounds for ttt=1:T 
   @views ind = rowIDT[ttt,1];
   @views hi[ind]= Mtau*hi[ind];
   @views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
   @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
   @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
   @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
   @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
 end # begin
 end # for ttt=1:T
elseif length(Wy)>1
 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = 1;
 @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
 @views Pi = σᵥ²*(Mrho*Mrho');
 @views lndetPi = log(det(Pi));
 @views invPi = (Pi)\I(N);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end










function  prtlloglikedh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);


 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = 1;
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind]   ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end







function prtlloglike(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
     
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                 jlms, bc = prtlloglikedt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                 jlms, bc = prtlloglikedt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                 jlms, bc = prtlloglikedt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                 jlms, bc = prtlloglikedt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                 jlms, bc = prtlloglikedt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                 jlms, bc = prtlloglikedt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                 jlms, bc = prtlloglikedt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, pos, rho,  eigvalu, rowIDT )
             else # 
                 jlms, bc = prtlloglikedt_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end
     end 
     
     return jlms, bc  
 
 end







function  prtlloglikedh_yuv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);

#   @views Mgamma = (I(N)-gamma*Wy[1])\I(N)


  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end




function  prtlloglikedh_yu(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
  
  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind]  ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end







function  prtlloglikedh_yv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    Wy::Matrix,  Wv::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = 1 ;
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);

#   @views Mgamma = (I(N)-gamma*Wy[1])\I(N)


  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = 1;
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end









function  prtlloglikedh_y(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
  @views Mtau = 1;
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);

#   @views Mgamma = (I(N)-gamma*Wy[1])\I(N)


  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind]  ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
  @views Mtau = 1;
  @views invPi = 1/σᵥ²*I(N);
  @views lndetPi = N*log(σᵥ²);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind]  ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end








function  prtlloglikedh_uv(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[1])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);

  @floop begin
  @inbounds for ttt=1:T 
    @views ind = rowIDT[ttt,1];
    @views hi[ind]= Mtau*hi[ind];
    @views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
    @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
    @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
    @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
  end # begin
  end # for ttt=1:T
elseif length(Wy)>1
  @floop begin
  @inbounds for ttt=1:T  
  @views N = rowIDT[1,2];
  @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
  @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
  @views Pi = σᵥ²*(Mrho*Mrho');
  @views lndetPi = log(det(Pi));
  @views invPi = (Pi)\I(N);
 
  @views ind = rowIDT[ttt,1];
  @views hi[ind]= Mtau*hi[ind];
  @views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
  @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
  @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
  @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
  @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
  end # for ttt=1:T
  end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end










function  prtlloglikedh_u(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

 @views N = rowIDT[1,2];
 @views Mtau = (I(N)-tau*Wu[1])\I(N);
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);


 @floop begin
 @inbounds for ttt=1:T 
   @views ind = rowIDT[ttt,1];
   @views hi[ind]= Mtau*hi[ind];
   @views ϵ[ind] = ϵ[ind];
   @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
   @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
   @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
   @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
 end # begin
 end # for ttt=1:T
elseif length(Wy)>1
 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind]   ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end









function  prtlloglikedh_v(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);

if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

 @views N = rowIDT[1,2];
 @views Mtau = 1 ;
 @views Mrho = (I(N)-rhomy*Wv[1])\I(N);
 @views Pi = σᵥ²*(Mrho*Mrho');
 @views lndetPi = log(det(Pi));
 @views invPi = (Pi)\I(N);

 @floop begin
 @inbounds for ttt=1:T 
   @views ind = rowIDT[ttt,1];
   @views hi[ind]= Mtau*hi[ind];
   @views ϵ[ind] = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
   @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
   @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
   @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
   @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi;
 end # begin
 end # for ttt=1:T
elseif length(Wy)>1
 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = 1;
 @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
 @views Pi = σᵥ²*(Mrho*Mrho');
 @views lndetPi = log(det(Pi));
 @views invPi = (Pi)\I(N);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
end  #    if length(Wy)==1 
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end










function  prtlloglikedh_(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
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
es2 = zeros(eltype(y),T,1);
KK = zeros(eltype(y),T,1);


 @floop begin
 @inbounds for ttt=1:T  
 @views N = rowIDT[1,2];
 @views Mtau = 1;
 @views invPi = 1/σᵥ²*I(N);
 @views lndetPi = N*log(σᵥ²);

 @views ind = rowIDT[ttt,1];
 @views hi[ind]= Mtau*hi[ind];
 @views ϵ[ind] = ϵ[ind]   ;
 @views sigs2[ttt] = 1 /(hi[ind]'*invPi*hi[ind]+1/σᵤ²);
 @views mus[ttt] = (μ/σᵤ² - ϵ[ind]'*invPi*hi[ind])*sigs2[ttt] ;
 @views es2[ttt] = -0.5*ϵ[ind]'*invPi*ϵ[ind] ;
 @views KK[ttt] = -0.5*N*log(2 * π)-0.5*lndetPi; 
 end # for ttt=1:T
 end # begin
@views liky = @. KK + es2 + 0.5*(((mus^2)/sigs2) - (μ^2 /σᵤ²) ) + 
               0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) - 
               0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ ))
return liky    
end




function prtlloglike(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w::Matrix, v::Matrix, z, EN::Matrix, IV::Matrix,
    PorC::Int64,  num::NamedTuple,  pos::NamedTuple, rho::Array{Float64,1}, eigvalu::NamedTuple, rowIDT::Matrix{Any})
 
   Wy = _dicM[:wy]
   Wu = _dicM[:wu]
   Wv = _dicM[:wv]
     
     if Wy!=Nothing  # yuv
         if Wu!=Nothing 
             if Wv!=Nothing #yuv
                liky = prtlloglikedh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else # yu
                liky = prtlloglikedh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  )
             end    
         else 
             if Wv!=Nothing #yv
                liky = prtlloglikedh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, pos, rho,  eigvalu, rowIDT )
             else #y
                liky = prtlloglikedh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, pos, rho,  eigvalu, rowIDT )  
             end
         end
     else
         if Wu!=Nothing 
             if Wv!=Nothing #uv
                liky = prtlloglikedh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, pos, rho,  eigvalu, rowIDT  )
             else # u
                liky = prtlloglikedh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, pos, rho,  eigvalu, rowIDT  ) 
             end    
         else 
             if Wv!=Nothing #v
                liky = prtlloglikedh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, num, pos, rho,  eigvalu, rowIDT )
             else # 
                liky = prtlloglikedh_( y, x, Q, w, v, z, EN, IV, PorC, num, pos, rho,  eigvalu, rowIDT  )  
             end
         end
     end 
     
     return liky 
 
 end
 

