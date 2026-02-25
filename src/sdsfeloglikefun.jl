# y x         σᵤ²  σᵥ² -- expo      
# y x         σᵤ²  σᵥ² -- half
# y x μ       σᵤ²  σᵥ² -- trun     
# y x μ  h    σᵤ²  σᵥ² -- scal  
# y x    h    σᵤ²  σᵥ² -- TFE_WH2010, half  
# y x μ  h    σᵤ²  σᵥ² -- TFE_WH2010, truncated  
# y x μ  g    σᵤ²  σᵥ² -- decay  
# y x         σᵤ²  σᵥ² -- panel half (2014 JoE)
# y x    σₐ²  σᵤ²  σᵥ² -- TRE  
# ------------------------------------------
# y x z  q    w    v   -- generic varname
#   β δ1 τ    δ2   γ   -- coeff 


function simple_check(xs)
    any(x -> isnan(x) || !isfinite(x) ,xs)
end

function ssdoah_yuv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   

    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function ssdoah_yv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   # taup = rho[po.begtau]
   # tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   

    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoah_yu( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )   

β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   # rhomyp = rho[po.begrho]
   # rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt]) \I(N);
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoah_y( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = 0.0 # δ1
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
                
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end






function ssdoah_uv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = 0.0 # δ1
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views Mrho =   (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wu)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wu)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoah_u( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind];
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wu)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wu)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoah_v( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]


   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mrho =   (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wv)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wv)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoah_( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0.0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try

    lik = zero(eltype(y));



        @floop begin
        @inbounds for ttt=1:T  
            @views N = rowIDT[ttt,2];
            @views invPi = I(N)/σᵥ²;
            @views lndetPi = N*log(σᵥ²);

                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin


    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end


function ssdoadh_yuv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)



   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = Mtau*hi[ind];
                    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                    @views es2 = -0.5*ϵs'*invPi*ϵs ;
                    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function ssdoadh_yv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β )
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = hi[ind];
                    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                    @views es2 = -0.5*ϵs'*invPi*ϵs ;
                    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1
        
    lik = zero(eltype(y));
 @floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadh_yu( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )   
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0

   ϵ =   PorC*(y - x*β  )
   T = size(rowIDT,1)

   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadh_y( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta)  ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end






function ssdoadh_uv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
            if simple_check(temp)
                lik += -1e9
            else
                lik += temp
            end # simple_check(temp)
    end # for ttt=1:T
        end # begin

elseif length(Wu)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wu)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoadh_u( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wu)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1.0 /σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wu)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadh_v( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)

   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wv)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wv)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadh_( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0.0
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)



    lik = zero(eltype(y));

        @floop begin
    @inbounds  for ttt=1:T  
        @views N = rowIDT[ttt,2];
        @views invPi = 1.0/σᵥ²*I(N);
        @views lndetPi = N*log(σᵥ²);
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end




function LL_T(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,en,iv,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 
 Wy = _dicM[:wy]
 Wu = _dicM[:wu]
 Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
   if Wu!=Nothing 
       if Wv!=Nothing #yuv
           llt = ssdoah_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
       else # yu
           llt = ssdoah_yu( y, x, Q, w, v, z, Wy, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
       end    
   else 
       if Wv!=Nothing #yv
           llt = ssdoah_yv(y, x, Q, w, v, z, Wy, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
       else #y
           llt = ssdoah_y(y, x, Q, w, v, z, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  
       end
   end
else
   if Wu!=Nothing 
       if Wv!=Nothing #uv
           llt = ssdoah_uv(y, x, Q, w, v, z, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT  )
       else # u
           llt = ssdoah_u(y, x, Q, w, v, z, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
       end    
   else 
       if Wv!=Nothing #v
           llt = ssdoah_v(y, x, Q, w, v, z, Wv,PorC, num, po, rho,  eigvalu, rowIDT )
       else # 
           llt = ssdoah_( y, x, Q, w, v, z, PorC, num, po, rho,  eigvalu, rowIDT  )  
       end
   end
end 
return llt
end



function ssdoat_yuv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)

   try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   

    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function ssdoat_yv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   # taup = rho[po.begtau]
   # tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   =  δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   

    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoat_yu( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )   

β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   # rhomyp = rho[po.begrho]
   # rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt]) \I(N);
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoat_y( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1.0 +exp(gammap))+eigvalu.rymax*exp(gammap)/(1.0 +exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wy)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
                
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wy)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end






function ssdoat_uv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views Mrho =   (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wu)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wu)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoat_u( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1.0 +exp(taup))+eigvalu.rumax*exp(taup)/(1.0 +exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind];
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wu)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wu)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoat_v( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1.0 +exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mrho =   (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin
elseif length(Wv)>1
@floop begin

    lik = zero(eltype(y));
@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        lik += -1e9
    else
        lik += temp
    end # simple_check(temp)
    end # for ttt=1:T
    end # begin
    
end #  if length(Wv)==1

    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoat_( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz] 

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try

    lik = zero(eltype(y));


        @floop begin
        @inbounds for ttt=1:T  
            @views N = rowIDT[ttt,2];

            @views invPi = I(N)/σᵥ²;
            @views lndetPi = N*log(σᵥ²);
                @views ind = rowIDT[ttt,1];
                @views his = hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                @views es2 = -0.5*ϵs'*invPi*ϵs ;
                @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;


                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                if simple_check(temp)
                    lik += -1e9
                else
                    lik += temp
                end # simple_check(temp)
            end # for ttt=1:T 
        end # begin


    return -lik

catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function ssdoadt_yuv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β)
   T = size(rowIDT,1)



   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = Mtau*hi[ind];
                    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                    @views es2 = -0.5*ϵs'*invPi*ϵs ;
                    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1
# println("aaa",likx,"aaasss",lik) 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function ssdoadt_yv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β   )
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = hi[ind];
                    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
                    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
                    @views es2 = -0.5*ϵs'*invPi*ϵs ;
                    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1
        
    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadt_yu( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )   
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  )
   T = size(rowIDT,1)

   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind] - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta)  ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadt_y( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  )
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wy)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind]  - PorC*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wy)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end






function ssdoadt_uv( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wu::Matrix, Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β )
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
            if simple_check(temp)
                lik += -1e9
            else
                lik += temp
            end # simple_check(temp)
    end # for ttt=1:T
        end # begin

elseif length(Wu)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]- PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wu)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoadt_u( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wu::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1.0 +exp(taup))+(eigvalu.rumax)*exp(taup)/(1.0 +exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β   )
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));

    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
        @views N = rowIDT[ttt,2];
        @views Mtau = (I(N)-tau*Wu[1])\I(N);
        @views invPi = 1.0/σᵥ²*I(N);
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wu)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1.0/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wu)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end








function ssdoadt_v( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    Wv::Matrix,PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1.0 +exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1.0 +exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  )
   T = size(rowIDT,1)

   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

elseif length(Wv)>1

    lik = zero(eltype(y));
@floop begin

@inbounds for ttt=1:T
    @views N = rowIDT[ttt,2];

    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = hi[ind];
    @views ϵs  = ϵ[ind] - PorC* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            lik += -1e9
        else
            lik += temp
        end # simple_check(temp)
    end # for ttt=1:T
end # begin

end # length(Wv)==1 
    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end





function ssdoadt_( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    
    β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%
   # @inbounds  for ii=1:num.nofeta
   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
   #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try
@floop begin
@inbounds for iitt =1:num.nofobs
         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
            if simple_check(tempx)
                likx += -1e9
            else
                likx += tempx
            end # simple_check(tempx)

    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   δ1 = rho[po.begz]

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  )
   T = size(rowIDT,1)



    lik = zero(eltype(y));

        @floop begin
    @inbounds  for ttt=1:T  
        @views N = rowIDT[ttt,2];
        @views invPi = 1.0/σᵥ²*I(N);
        @views lndetPi = N*log(σᵥ²);
            @views ind = rowIDT[ttt,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*invPi*his + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
            @views es2 = -0.5*ϵs'*invPi*ϵs ;
            @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:T
        end # begin

    return -lik-likx
catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
    return 1e100
end
end



function LL_T(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,en,iv,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 
 Wy = _dicM[:wy]
 Wu = _dicM[:wu]
 Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
   if Wu!=Nothing 
       if Wv!=Nothing #yuv
           llt = ssdoat_yuv( y, x, Q, w, v, z, Wy, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
       else # yu
           llt = ssdoat_yu( y, x, Q, w, v, z, Wy, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
       end    
   else 
       if Wv!=Nothing #yv
           llt = ssdoat_yv(y, x, Q, w, v, z, Wy, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
       else #y
           llt = ssdoat_y(y, x, Q, w, v, z, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  
       end
   end
else
   if Wu!=Nothing 
       if Wv!=Nothing #uv
           llt = ssdoat_uv(y, x, Q, w, v, z, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT  )
       else # u
           llt = ssdoat_u(y, x, Q, w, v, z, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
       end    
   else 
       if Wv!=Nothing #v
           llt = ssdoat_v(y, x, Q, w, v, z, Wv,PorC, num, po, rho,  eigvalu, rowIDT )
       else # 
           llt = ssdoat_( y, x, Q, w, v, z, PorC, num, po, rho,  eigvalu, rowIDT  )  
       end
   end
end 
return llt
end





function LL_T(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
    EN::Matrix,IV::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
    if Wu!=Nothing 
        if Wv!=Nothing #yuv
            llt = ssdoadt_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
        else # yu
            llt = ssdoadt_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
        end    
    else 
        if Wv!=Nothing #yv
            llt = ssdoadt_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
        else #y
            llt = ssdoadt_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  
        end
    end
else
    if Wu!=Nothing 
        if Wv!=Nothing #uv
            llt = ssdoadt_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT  )
        else # u
            llt = ssdoadt_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
        end    
    else 
        if Wv!=Nothing #v
            llt = ssdoadt_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, po, rho,  eigvalu, rowIDT )
        else # 
            llt = ssdoadt_( y, x, Q, w, v, z, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT  )  
        end
    end
end 
return llt
end




function LL_T(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
    EN::Matrix,IV::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

  Wy = _dicM[:wy]
  Wu = _dicM[:wu]
  Wv = _dicM[:wv]

if Wy!=Nothing  # yuv
    if Wu!=Nothing 
        if Wv!=Nothing #yuv
            llt = ssdoadh_yuv( y, x, Q, w, v, z, EN, IV, Wy, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
        else # yu
            llt = ssdoadh_yu( y, x, Q, w, v, z, EN, IV, Wy, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
        end    
    else 
        if Wv!=Nothing #yv
            llt = ssdoadh_yv(y, x, Q, w, v, z, EN, IV, Wy, Wv, PorC, num, po, rho,  eigvalu, rowIDT )
        else #y
            llt = ssdoadh_y(y, x, Q, w, v, z, EN, IV, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  
        end
    end
else
    if Wu!=Nothing 
        if Wv!=Nothing #uv
            llt = ssdoadh_uv(y, x, Q, w, v, z, EN, IV, Wu, Wv, PorC, num, po, rho,  eigvalu, rowIDT  )
        else # u
            llt = ssdoadh_u(y, x, Q, w, v, z, EN, IV, Wu, PorC, num, po, rho,  eigvalu, rowIDT  )
        end    
    else 
        if Wv!=Nothing #v
            llt = ssdoadh_v(y, x, Q, w, v, z, EN, IV, Wv,PorC, num, po, rho,  eigvalu, rowIDT )
        else # 
            llt = ssdoadh_( y, x, Q, w, v, z, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT  )  
        end
    end
end 
return llt
end



# function LL_T(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
#     EN::Matrix,IV::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

#   Wy = _dicM[:wy]

#   β  = rho[1:po.endx]
#   τ  = rho[po.begq:po.endq]
#   phi = rho[po.begphi:po.endphi]
# ## calculate lkx
#   nofiv = num.nofphi/num.nofeta
#   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

#   # %%%%%%%%%%%%%%
#   # @inbounds  for ii=1:num.nofeta
#   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
#   #  end
#    @views phi = reshape(phi, :, num.nofeta)
#    @views eps = EN- IV*phi

#    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
#    @views logdetll = log(det(LL))
#    @views invll = LL\I(num.nofeta)
#    likx = zero(eltype(y));


# try
# @floop begin
# @inbounds for iitt =1:num.nofobs
#         tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
#            if simple_check(tempx)
#                likx += -1e9
#            else
#                likx += tempx
#            end # simple_check(tempx)

#    end # iitt =1:num.nofobs
# end # @floop begin
# ## calculate lky

#   eta = rho[po.begeta:po.endeta]
#   δ2 = rho[po.begw]  
#   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
# #   δ1 = rho[po.begz]
#   gammap = rho[po.beggamma]
#   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

#   hi  = exp.(Q*τ)
#   σᵤ²= exp(δ2) 
#   σᵤ= exp(0.5*δ2) 
#   σᵥ² = exp(γ)            # todo: 重新换一下字母 
#   σᵥ = exp(0.5*γ)  
#   μ   = 0.0
#   ϵ = PorC*(y - x*β)
#   T = size(rowIDT,1)


#   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

#    lik = zero(eltype(y));
#    @views N = rowIDT[1,2];
#    lndetIrhoW_sum = zero(eltype(y));

#    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
#    @views lndetIrhoW_sum =N*lndetIrhoW
   
#    @floop begin
#     @inbounds for ttt=1:T
#         @views ind = rowIDT[ttt,1];     ###rowIDT 

#         @inbounds for (ind_indx, ii) in enumerate(ind)

#         @views KK = -0.5*log(2 * π)-0.5*log(σᵥ²);
#         @views his = hi[ii];
#         @views wyy = PorC*gamma*Wy[1]'*y[ind]
#         @views ϵs  = ϵ[ii]-wyy[ind_indx] - PorC*(eps[ii,:]'*eta)  ;
#         @views sigs2 = 1 / (his'*(1/σᵥ²)*his + 1 /σᵤ²) ;
#         @views mus = (μ/σᵤ² - ϵs'*(1/σᵥ²)*his)*sigs2 ;
#         @views es2 = -0.5*ϵs'*ϵs ;
#         @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
#                         0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
#                         0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
#                 if simple_check(temp)
#                     lik += -1e9
#                 else
#                     lik += temp
#                 end # simple_check(temp)      
#             end # for ii=1:num.nofobs
#         end # for ttt=1:T
#     end # begin


# elseif length(Wy)>1

# lik = zero(eltype(y));
# @views N = rowIDT[1,2];
# lndetIrhoW_sum = zero(eltype(y));

# @floop begin
# @inbounds for ttt=1:T
#    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
#    @views lndetIrhoW_sum +=lndetIrhoW
#    @views ind = rowIDT[ttt,1];     ###rowIDT    

# #    10×3 DataFrame
# #    Row │ obs    id     t     
# #        │ Int64  Int64  Int64 
# #   ─────┼─────────────────────
# #      1 │     1      1      1
# #      2 │     2      2      1
# #      3 │     3      3      1
# #      4 │     4      1      2
# #      5 │     5      2      2
# #      6 │     6      1      3
# #      7 │     7      1      4
# #      8 │     8      2      4
# #      9 │     9      3      4
# #     10 │    10      4      4

# #    4×2 Matrix{Any}:
# #  [1, 2, 3]      3
# #  [4, 5]         2
# #  [6]            1
# #  [7, 8, 9, 10]  4



# @inbounds for (ind_indx, ii) in enumerate(ind)
#     @views KK = -0.5*log(2 * π)-0.5*log(σᵥ²);
#    @views his = hi[ii];
#    @views wyy = PorC*gamma*Wy[1]'*y[ind]
#    @views ϵs  = ϵ[ii]-wyy[ind_indx] - PorC*(eps[ii,:]'*eta)  ;
#    @views sigs2 = 1 / (his'*(1/σᵥ²)*his + 1 /σᵤ²) ;
#    @views mus = (μ/σᵤ² - ϵs'*(1/σᵥ²)*his)*sigs2 ;
#    @views es2 = -0.5*ϵs'*ϵs ;
#    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
#                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
#                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
#            if simple_check(temp)
#                lik += -1e9
#            else
#                lik += temp
#            end # simple_check(temp)      
#        end # for ii=1:num.nofobs
# end # begin
# end # for ttt=1:T

# end # length(Wy)==1 
#    return -lik-lndetIrhoW_sum-likx
# catch e
# 处理异常的代码
# println("操作失败，发生错误：$e")
#    return 1e100
# end

# end


# function LL_T(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
#     EN::Matrix,IV::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

#   Wy = _dicM[:wy]

#   β  = rho[1:po.endx]
#   τ  = rho[po.begq:po.endq]
#   phi = rho[po.begphi:po.endphi]
# ## calculate lkx
#   nofiv = num.nofphi/num.nofeta
#   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

#   # %%%%%%%%%%%%%%
#   # @inbounds  for ii=1:num.nofeta
#   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
#   #  end
#    @views phi = reshape(phi, :, num.nofeta)
#    @views eps = EN- IV*phi

#    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
#    @views logdetll = log(det(LL))
#    @views invll = LL\I(num.nofeta)
#    likx = zero(eltype(y));


#    try
#     @floop begin
#     @inbounds for iitt =1:num.nofobs
#              tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
#                 if simple_check(tempx)
#                     likx += -1e9
#                 else
#                     likx += tempx
#                 end # simple_check(tempx)
    
#         end # iitt =1:num.nofobs
#     end # @floop begin
#     ## calculate lky
    
#        eta = rho[po.begeta:po.endeta]
#        δ2 = rho[po.begw]  
#        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
#        # δ1 = rho[po.begz]
#        gammap = rho[po.beggamma]
#        gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));
    
#        hi  = exp.(Q*τ)
#        σᵤ²= exp(δ2) 
#        σᵤ= exp(0.5*δ2) 
#        σᵥ² = exp(γ)            # todo: 重新换一下字母 
#        σᵥ = exp(0.5*γ)  
#        μ   = 0.0
#        ϵ = PorC*(y - x*β)
#        T = size(rowIDT,1)
    
    
#        if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
    
#         lik = zero(eltype(y));
#         @views N = rowIDT[1,2];
#         @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
#         @views invPi = 1/σᵥ²*I(N);
#         @views lndetPi = N*log(σᵥ²);
#             @floop begin
#         @inbounds  for ttt=1:T  
#                 @views ind = rowIDT[ttt,1];
#                 @views his = hi[ind];
#                 @views ϵs  = ϵ[ind]-PorC*gamma*Wy[1]*y[ind]  - PorC*(eps[ind,:]*eta) ;
#                 @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
#                 @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
#                 @views es2 = -0.5*ϵs'*invPi*ϵs ;
#                 @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
    
#                 @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
#                                 0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
#                                 0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
#                         if simple_check(temp)
#                             lik += -1e9
#                         else
#                             lik += temp
#                         end # simple_check(temp)
#                     end # for ttt=1:T
#             end # begin
    
#     elseif length(Wy)>1
    
#         lik = zero(eltype(y));
#         @views N = rowIDT[1,2];
#     @floop begin
    
#     @inbounds for ttt=1:T
    
#         @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
#         @views invPi = 1/σᵥ²*I(N);
#         @views lndetPi = N*log(σᵥ²);
        
#         @views ind = rowIDT[ttt,1];
#         @views his = hi[ind];
#         @views ϵs  = ϵ[ind]-PorC*gamma*Wy[ttt]*y[ind] - PorC*(eps[ind,:]*eta)  ;
#         @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
#         @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
#         @views es2 = -0.5*ϵs'*invPi*ϵs ;
#         @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                    
#         @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
#                         0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
#                         0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
            
#             if simple_check(temp)
#                 lik += -1e9
#             else
#                 lik += temp
#             end # simple_check(temp)
#         end # for ttt=1:T
#     end # begin
    
#     end # length(Wy)==1 
#         return -lik-likx
#     catch e
#     # 处理异常的代码
#     println("操作失败，发生错误：$e")
#         return 1e100
#     end
#     end
    
    

# function LL_T(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
#     EN::Matrix,IV::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

#   Wy = _dicM[:wy]

#   β  = rho[1:po.endx]
#   τ  = rho[po.begq:po.endq]
#   phi = rho[po.begphi:po.endphi]
# ## calculate lkx
#   nofiv = num.nofphi/num.nofeta
#   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

#   # %%%%%%%%%%%%%%
#   # @inbounds  for ii=1:num.nofeta
#   #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
#   #  end
#    @views phi = reshape(phi, :, num.nofeta)
#    @views eps = EN- IV*phi

#    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
#    @views logdetll = log(det(LL))
#    @views invll = LL\I(num.nofeta)
#    likx = zero(eltype(y));


#    try
#     @floop begin
#     @inbounds for iitt =1:num.nofobs
#              tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
#                 if simple_check(tempx)
#                     likx += -1e9
#                 else
#                     likx += tempx
#                 end # simple_check(tempx)
    
#         end # iitt =1:num.nofobs
#     end # @floop begin
#     ## calculate lky
    
#        eta = rho[po.begeta:po.endeta]
#        δ2 = rho[po.begw]  
#        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
#        # δ1 = rho[po.begz]
#        gammap = rho[po.beggamma]
#        gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));
    
#        hi  = exp.(Q*τ)
#        σᵤ²= exp(δ2) 
#        σᵤ= exp(0.5*δ2) 
#        σᵥ² = exp(γ)            # todo: 重新换一下字母 
#        σᵥ = exp(0.5*γ)  
#        μ   = 0.0
#        ϵ = PorC*(y - x*β)
#        T = size(rowIDT,1)
    

#     if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

#         lik = zero(eltype(y));
#         @views N = rowIDT[1,2];
#         Wyt = kron(I(T), Wy[1])

#         @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));  
#         @views lndetIrhoWt = lndetIrhoW*T
#         @views invPi = 1/σᵥ²;
#         @views lndetPi = log(σᵥ²);
   
#         @views his = hi;
#         @views ϵs  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
#         @views sigs2 = 1 ./ (his.^2 .*invPi .+ 1 /σᵤ²) ;
#         @views mus = (μ/σᵤ² .- ϵs .* his .* invPi) .* sigs2 ;
#         @views es2 = -0.5 .* ϵs.^2 .*invPi;
#         @views KK = -0.5*log(2 * π)-0.5*lndetPi;
#         @views temp_1 = KK .+  es2 .+ 0.5 .* (((mus .^ 2) ./ sigs2) .- (μ^2 / σᵤ²) ) .+
#                         0.5 .* log.(sigs2) .+ log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                         0.5 * log(σᵤ²) .- log(normcdf(μ / σᵤ))
#                 # print(size(temp_1))

#         # 检查 lik 是否为 NaN, 非实数, 或 Inf
#         @views temp_1 = map(x -> x ≠ x ? -1e9 : isinf(x) ? -1e9 : x, temp_1)
#         # 计算总和
#         @views lik = sum(temp_1)

#     elseif length(Wy)>1
#         lik = zero(eltype(y));
#         @views N = rowIDT[1,2];
#         Wyt = kron(I(T), Wy[1])

#         @floop begin
#         @inbounds for ttt=1:T
#             @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));    
#             @views lndetIrhoWt += lndetIrhoW
#         end # for ttt=1:T
#         end # begin
        
#         @views invPi = 1/σᵥ²;
#         @views lndetPi = log(σᵥ²);
#         @views his = hi;
#         @views ϵs  = ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
#         @views sigs2 = 1 ./ (his.^2 .*invPi .+ 1 /σᵤ²) ;
#         @views mus = (μ/σᵤ² .- ϵs .* his .* invPi) .* sigs2 ;
#         @views es2 = -0.5 .* ϵs.^2 .*invPi;
#         @views KK = -0.5*log(2 * π)-0.5*lndetPi;
#         @views temp_1 = KK .+  es2 .+ 0.5 .* (((mus .^ 2) ./ sigs2) .- (μ^2 / σᵤ²) ) .+
#                         0.5 .* log.(sigs2) .+ log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                         0.5 * log(σᵤ²) .- log(normcdf(μ / σᵤ))
#                 # print(size(temp_1))

#         # 检查 lik 是否为 NaN, 非实数, 或 Inf
#         @views temp_1 = map(x -> x ≠ x ? -1e9 : isinf(x) ? -1e9 : x, temp_1)
#         # 计算总和
#         @views lik = sum(temp_1)

#     end # if length(Wy)==1 
#         return -lik-likx-lndetIrhoWt
#     catch e
#     # 处理异常的代码
#     println("操作失败，发生错误：$e")
#         return 1e100
#     end
#     end
    
    

    
function ssdkuhe( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]
## calculate lkx
    nofiv = num.nofphi/num.nofeta
    eps = zeros(eltype(EN),num.nofobs,num.nofeta);

    # %%%%%%%%%%%%%%
    # @inbounds  for ii=1:num.nofeta
    #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
    #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


   try
    @floop begin
    @inbounds for iitt =1:num.nofobs
                tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                if simple_check(tempx)
                    likx += -1e99
                else
                    likx += tempx
                end # simple_check(tempx)
    
        end # iitt =1:num.nofobs
    end # @floop begin
    ## calculate lky
    
        eta = rho[po.begeta:po.endeta]
        δ2 = rho[po.begw]  
        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
        # δ1 = rho[po.begz]
        gammap = rho[po.beggamma]
        gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));
    
        hi  = exp.(Q*τ)
        σᵤ²= exp(δ2) 
        σᵤ= exp(0.5*δ2) 
        σᵥ² = exp(γ)            # todo: 重新换一下字母 
        σᵥ = exp(0.5*γ)  
        μ   = 0.0
        ϵ = PorC*(y - x*β)
        T = size(rowIDT,1)
    

    if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

        lik = zero(eltype(y));
        @views N = rowIDT[1,2];
        Wyt = kron(I(T), Wy[1])

        @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));  
        @views lndetIrhoWt = lndetIrhoW*T
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
    
        @views ϵ  =  ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
        @views sigs2 = @. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                # print(size(temp_1))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)


    elseif length(Wy)>1
        lik = zero(eltype(y));
        Wyt = kron(I(T), Wy[1])

        @floop begin
        @inbounds for ttt=1:T
            @views N = rowIDT[ttt,2];
            @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));    
            @views lndetIrhoWt += lndetIrhoW
        end # for ttt=1:T
        end # begin
        
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
        @views ϵ  =  ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
        @views sigs2 = @. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                # print(size(temp_1))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)

    end # if length(Wy)==1 
        return -lik-likx-lndetIrhoWt
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end


function ssdkuh( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
   try
    ## calculate lky
        δ2 = rho[po.begw]  
        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
        # δ1 = rho[po.begz]
        gammap = rho[po.beggamma]
        gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));
    
        hi  = exp.(Q*τ)
        σᵤ²= exp(δ2) 
        σᵤ= exp(0.5*δ2) 
        σᵥ² = exp(γ)            # todo: 重新换一下字母 
        σᵥ = exp(0.5*γ)  
        μ   = 0.0
        ϵ = PorC*(y - x*β)
        T = size(rowIDT,1)
    

    if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

        lik = zero(eltype(y));
        @views N = rowIDT[1,2];
        Wyt = kron(I(T), Wy[1])

        @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));  
        @views lndetIrhoWt = lndetIrhoW*T
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
    
        @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
        @views sigs2 = @. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@.  -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                # print(size(temp_1))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)
        @views lll =  lik+lndetIrhoWt


    elseif length(Wy)>1
        lik = zero(eltype(y));
        Wyt = kron(I(T), Wy[1])

        @floop begin
        @inbounds for ttt=1:T
            @views N = rowIDT[ttt,2];

            @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));    
            @views lndetIrhoWt += lndetIrhoW
        end # for ttt=1:T
        end # begin
        
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
        @views ϵ  = ϵ-PorC*gamma*Wyt*y   ;
        @views sigs2 =@. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus = @. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 = @. -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                # print(size(temp_1))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)
        @views lll =  lik+lndetIrhoWt

    end # if length(Wy)==1 
        return -lll
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end

end


function LL_T(::Type{SSFKUEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    Wy = _dicM[:wy]

    llt = ssdkuhe(y, x, Q, EN, IV, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFKUH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    Wy = _dicM[:wy]

    llt = ssdkuh(y, x, Q,  Wy, PorC, num, po, rho,  eigvalu, rowIDT ) 

    return llt
end



function ssdkute( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]
## calculate lkx
    nofiv = num.nofphi/num.nofeta
    eps = zeros(eltype(EN),num.nofobs,num.nofeta);

    # %%%%%%%%%%%%%%
    # @inbounds  for ii=1:num.nofeta
    #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
    #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));


   try
    @floop begin
        @inbounds for iitt =1:num.nofobs
                 tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                    if simple_check(tempx)
                        likx += -1e99
                    else
                        likx += tempx
                    end # simple_check(tempx)
        
            end # iitt =1:num.nofobs
        end # @floop begin
        ## calculate lky
    
        eta = rho[po.begeta:po.endeta]
        δ2 = rho[po.begw]  
        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
        δ1 = rho[po.begz]
        gammap = rho[po.beggamma]
        gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));
    
        hi  = exp.(Q*τ)
        σᵤ²= exp(δ2) 
        σᵤ= exp(0.5*δ2) 
        σᵥ² = exp(γ)            # todo: 重新换一下字母 
        σᵥ = exp(0.5*γ)  
        μ   = δ1
        ϵ = PorC*(y - x*β)
        T = size(rowIDT,1)
    

    if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

        lik = zero(eltype(y));
        @views N = rowIDT[1,2];
        Wyt = kron(I(T), Wy[1])

        @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));  
        @views lndetIrhoWt = lndetIrhoW*T
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
    
        @views ϵ  =  ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
        @views sigs2 = @.  1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
 
        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)
        @views lll =  lik+likx+lndetIrhoWt

    elseif length(Wy)>1
        lik = zero(eltype(y));
        Wyt = kron(I(T), Wy[1])

        @floop begin
        @inbounds for ttt=1:T
            @views N = rowIDT[ttt,2];

            @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));    
            @views lndetIrhoWt += lndetIrhoW
        end # for ttt=1:T
        end # begin
        
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
        @views ϵ  =  ϵ-PorC*gamma*Wyt*y  - PorC*(eps*eta) ;
        @views sigs2 = @.  1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)
        @views lll =  lik+likx+lndetIrhoWt

    end # if length(Wy)==1 

        return -lll
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end


function ssdkut( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
   try
    ## calculate lky
        δ2 = rho[po.begw]  
        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
        δ1 = rho[po.begz]
        gammap = rho[po.beggamma]
        gamma  = (eigvalu.rymin)/(1.0 +exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1.0 +exp(gammap));
    
        hi  = exp.(Q*τ)
        σᵤ²= exp(δ2) 
        σᵤ= exp(0.5*δ2) 
        σᵥ² = exp(γ)            # todo: 重新换一下字母 
        σᵥ = exp(0.5*γ)  
        μ   = δ1
        ϵ = PorC*(y - x*β)
        T = size(rowIDT,1)
    

    if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

        lik = zero(eltype(y));
        @views N = rowIDT[1,2];
        Wyt = kron(I(T), Wy[1])

        @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));  
        @views lndetIrhoWt = lndetIrhoW*T
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);

        @views ϵ  =  ϵ-PorC*gamma*Wyt*y   ;
        @views sigs2 =@. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                # print(size(temp_1))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)
        @views lll = lik+lndetIrhoWt


    elseif length(Wy)>1
        lik = zero(eltype(y));
        Wyt = kron(I(T), Wy[1])

        @floop begin
        @inbounds for ttt=1:T
            @views N = rowIDT[ttt,2];

            @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));    
            @views lndetIrhoWt += lndetIrhoW
        end # for ttt=1:T
        end # begin
        
        @views invPi = 1.0/σᵥ²;
        @views lndetPi = log(σᵥ²);
        @views ϵ  =  ϵ-PorC*gamma*Wyt*y   ;
        @views sigs2 =@. 1.0 / (hi^2 *invPi + 1.0 /σᵤ²) ;
        @views mus =@. (μ/σᵤ² - ϵ * hi * invPi) * sigs2 ;
        @views es2 =@. -0.5 * ϵ ^2 *invPi;
        @views KK = -0.5*log(2 * π)-0.5*lndetPi;
        @views temp_1 =@. KK +  es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                        0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))

        # 检查 lik 是否为 NaN, 非实数, 或 Inf
        @views temp_1 = map(x -> x ≠ x ? -1e99  : isinf(x) ? -1e99  : x, temp_1)
        # 计算总和
        @views lik = sum(temp_1)

        @views lll = lik+lndetIrhoWt

    end # if length(Wy)==1 

        return -lll
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end

end


function LL_T(::Type{SSFKUET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    Wy = _dicM[:wy]
    llt = ssdkute(y, x, Q, EN, IV, Wy, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFKUT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    Wy = _dicM[:wy]

    llt = ssdkut(y, x, Q,  Wy, PorC, num, po, rho,  eigvalu, rowIDT )  
    return llt
end







    
function ssdkkhe( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]
    ## calculate lkx
    nofiv = num.nofphi/num.nofeta
    eps = zeros(eltype(EN),num.nofobs,num.nofeta);

    # %%%%%%%%%%%%%%
    # @inbounds  for ii=1:num.nofeta
    #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
    #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

   try
    @floop begin
    @inbounds for iitt =1:num.nofobs
                tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                if simple_check(tempx)
                    likx += -1e9
                else
                    likx += tempx
                end # simple_check(tempx)
    
        end # iitt =1:num.nofobs
    end # @floop begin
    ## calculate lky
    
    eta = rho[po.begeta:po.endeta]
    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
    # δ1 = rho[po.begz]


    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)            # todo: 重新换一下字母 
    σᵥ = exp(0.5*γ)  
    μ   = 0.0
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)


    lik = zero(eltype(y));
    @views invPi = 1.0/σᵥ²;
    @floop begin
    @inbounds  for iidd=1:ID  
        @views N = rowIDT[iidd,2];
        @views lndetPi = N*log(σᵥ²);
            @views ind = rowIDT[iidd,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*his*invPi + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*his*invPi)*sigs2 ;
            @views es2 = -0.5*ϵs'*ϵs*invPi ;
            @views KK = 0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:ID
        end # begin

    return -lik-likx
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end
    
    
    


    
function ssdkkh( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  
     PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]

    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
    # δ1 = rho[po.begz]


    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)            # todo: 重新换一下字母 
    σᵥ = exp(0.5*γ)  
    μ   = 0.0
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)

    lik = zero(eltype(y));
    @views invPi = 1.0/σᵥ²;

    try
    @floop begin
    @inbounds  for iidd=1:ID  
        @views N = rowIDT[iidd,2];
        @views lndetPi = N*log(σᵥ²);

            @views ind = rowIDT[iidd,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]   ;
            @views sigs2 = 1.0 / (his'*his*invPi + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*his*invPi)*sigs2 ;
            @views es2 = -0.5*ϵs'*ϵs*invPi ;
            @views KK = 0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:ID
        end # begin

    return -lik
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end
    
    


function LL_T(::Type{SSFKKEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdkkhe(y, x, Q, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFKKH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdkkh(y, x, Q, PorC, num, po, rho,  eigvalu, rowIDT ) 

    return llt
end



    
function ssdkkte( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]
    ## calculate lkx
    nofiv = num.nofphi/num.nofeta
    eps = zeros(eltype(EN),num.nofobs,num.nofeta);

    # %%%%%%%%%%%%%%
    # @inbounds  for ii=1:num.nofeta
    #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
    #  end
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

   try
    @floop begin
    @inbounds for iitt =1:num.nofobs
                tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                if simple_check(tempx)
                    likx += -1e9
                else
                    likx += tempx
                end # simple_check(tempx)
    
        end # iitt =1:num.nofobs
    end # @floop begin
    ## calculate lky
    
    eta = rho[po.begeta:po.endeta]
    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[po.begz]


    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)            # todo: 重新换一下字母 
    σᵥ = exp(0.5*γ)  
    μ   = δ1
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)


    lik = zero(eltype(y));
    @views invPi = 1.0/σᵥ²;
    @floop begin
    @inbounds  for iidd=1:ID  
        @views N = rowIDT[iidd,2];
        # println("T IS ",N)
        @views lndetPi = N*log(σᵥ²);
            @views ind = rowIDT[iidd,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]  - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / (his'*his*invPi + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*his*invPi)*sigs2 ;
            @views es2 = -0.5*ϵs'*ϵs*invPi ;
            @views KK = 0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:ID
        end # begin

    return -lik-likx
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end
    
    
    


    
function ssdkkt( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  
     PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]

    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[po.begz]


    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)            # todo: 重新换一下字母 
    σᵥ = exp(0.5*γ)  
    μ   = δ1
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)

    lik = zero(eltype(y));
    @views invPi = 1.0/σᵥ²;

    try
    @floop begin
    @inbounds  for iidd=1:ID  
        @views N = rowIDT[iidd,2];
        @views lndetPi = N*log(σᵥ²);
            @views ind = rowIDT[iidd,1];
            @views his = hi[ind];
            @views ϵs  = ϵ[ind]   ;
            @views sigs2 = 1.0 / (his'*his*invPi + 1.0 /σᵤ²) ;
            @views mus = (μ/σᵤ² - ϵs'*his*invPi)*sigs2 ;
            @views es2 = -0.5*ϵs'*ϵs*invPi ;
            @views KK = 0.5*N*log(2 * π)-0.5*lndetPi;

            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e9
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:ID
        end # begin

    return -lik
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end
    

function LL_T(::Type{SSFKKET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdkkte(y, x, Q, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFKKT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdkkt(y, x, Q, PorC, num, po, rho,  eigvalu, rowIDT )  
    return llt
end






# ============ Giannini 2025 (GI) 系列 ============
function ssdgih(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, Q_lag::Matrix,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,
    eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]
    gammap = rho[po.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵤ  = exp(0.5*δ2)
    σᵥ² = exp(γ)
    μ   = 0.0
    ϵ   = PorC*(y - x*β)

    ID   = size(rowIDT, 1)
    T_fd = rowIDT[1, 2]
    T_orig = T_fd + 1

  try
    # spatial filtering: kron(W, I(T_fd)) for data sorted by [id, time]
    lndetIrhoWt = 0.0
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y
        lndetIrhoW = log(det(I(N) - gamma*Wy[1]))
        lndetIrhoWt = lndetIrhoW * T_fd
    end

    # Toeplitz inverse: A^{-1}_{jk} = min(j,k)*(T-max(j,k))/T
    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²
    lndetA = log(T_orig)

    lik = zero(eltype(y))
    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        T_i_fd = rowIDT[iidd, 2]

        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]

        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        es2   = -0.5 * ϵs'*invPi*ϵs
        KK    = -0.5*T_i_fd*log(2π) - 0.5*T_i_fd*log(σᵥ²) - 0.5*lndetA

        temp = KK + es2 + 0.5*((mus^2)/sigs2 - μ^2/σᵤ²) +
               0.5*log(sigs2) + log(normcdf(mus/sqrt(sigs2))) -
               0.5*log(σᵤ²) - log(normcdf(μ/σᵤ))

        if simple_check(temp)
            lik += -1e99
        else
            lik += temp
        end
    end

    return -(lik + lndetIrhoWt)
  catch e
    return 1e100
  end
end

# __SSDGIHE_START__

function ssdgihe(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, Q_lag::Matrix,
    EN, IV, Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,
    eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β   = rho[1:po.endx]
    τ   = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]

    # endogeneity part (likx)
    nofiv = num.nofphi / num.nofeta
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN - IV*phi

    @views LL = cov(eps)
    @views logdetll = log(det(LL))
    @views invll = LL \ I(num.nofeta)
    likx = zero(eltype(y))

  try
    @floop begin
    @inbounds for iitt = 1:num.nofobs
        tempx = -0.5*tr(invll*eps[iitt,:]'*eps[iitt,:])
        if simple_check(tempx)
            likx += -1e9
        else
            likx += tempx
        end
    end
    end

    ID = size(rowIDT, 1)
    @floop begin
    @inbounds for iidd = 1:ID
        @views T_i_fd = rowIDT[iidd, 2]
        tempx2 = T_i_fd * (-0.5*num.nofeta*log(2π) - 0.5*logdetll)
        if simple_check(tempx2)
            likx += -1e99
        else
            likx += tempx2
        end
    end
    end

    # efficiency part
    eta = rho[po.begeta:po.endeta]
    δ2  = rho[po.begw]
    γ   = rho[po.begv]
    gammap = rho[po.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵤ  = exp(0.5*δ2)
    σᵥ² = exp(γ)
    μ   = 0.0
    ϵ   = PorC*(y - x*β)

    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    # spatial filtering
    lndetIrhoWt = 0.0
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y - PorC*(eps*eta)
        lndetIrhoW = log(det(I(N) - gamma*Wy[1]))
        lndetIrhoWt = lndetIrhoW * T_fd
    end

    # __SSDGIHE_PART2__

    # Toeplitz inverse
    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²
    lndetA = log(T_orig)

    lik = zero(eltype(y))
    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        T_i_fd = rowIDT[iidd, 2]

        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]

        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        es2   = -0.5 * ϵs'*invPi*ϵs
        KK    = -0.5*T_i_fd*log(2π) - 0.5*T_i_fd*log(σᵥ²) - 0.5*lndetA

        temp = KK + es2 + 0.5*((mus^2)/sigs2 - μ^2/σᵤ²) +
               0.5*log(sigs2) + log(normcdf(mus/sqrt(sigs2))) -
               0.5*log(σᵤ²) - log(normcdf(μ/σᵤ))

        if simple_check(temp)
            lik += -1e99
        else
            lik += temp
        end
    end

    return -(lik + lndetIrhoWt) - likx
  catch e
    return 1e100
  end
end


# __SSDGIT_START__

function ssdgit(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, Q_lag::Matrix,
    Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,
    eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]
    δ1 = rho[po.begz]
    gammap = rho[po.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵤ  = exp(0.5*δ2)
    σᵥ² = exp(γ)
    μ   = δ1
    ϵ   = PorC*(y - x*β)

    ID   = size(rowIDT, 1)
    T_fd = rowIDT[1, 2]
    T_orig = T_fd + 1

  try
    lndetIrhoWt = 0.0
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y
        lndetIrhoW = log(det(I(N) - gamma*Wy[1]))
        lndetIrhoWt = lndetIrhoW * T_fd
    end

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²
    lndetA = log(T_orig)

    lik = zero(eltype(y))
    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        T_i_fd = rowIDT[iidd, 2]

        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]

        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        es2   = -0.5 * ϵs'*invPi*ϵs
        KK    = -0.5*T_i_fd*log(2π) - 0.5*T_i_fd*log(σᵥ²) - 0.5*lndetA

        temp = KK + es2 + 0.5*((mus^2)/sigs2 - μ^2/σᵤ²) +
               0.5*log(sigs2) + log(normcdf(mus/sqrt(sigs2))) -
               0.5*log(σᵤ²) - log(normcdf(μ/σᵤ))

        if simple_check(temp)
            lik += -1e99
        else
            lik += temp
        end
    end

    return -(lik + lndetIrhoWt)
  catch e
    return 1e100
  end
end


# __SSDGITE_START__

function ssdgite(y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, Q_lag::Matrix,
    EN, IV, Wy::Matrix, PorC::Int64, num::NamedTuple, po::NamedTuple, rho,
    eigvalu::NamedTuple, rowIDT::Matrix{Any})

    β   = rho[1:po.endx]
    τ   = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]

    nofiv = num.nofphi / num.nofeta
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN - IV*phi

    @views LL = cov(eps)
    @views logdetll = log(det(LL))
    @views invll = LL \ I(num.nofeta)
    likx = zero(eltype(y))

  try
    @floop begin
    @inbounds for iitt = 1:num.nofobs
        tempx = -0.5*tr(invll*eps[iitt,:]'*eps[iitt,:])
        if simple_check(tempx)
            likx += -1e9
        else
            likx += tempx
        end
    end
    end

    ID = size(rowIDT, 1)
    @floop begin
    @inbounds for iidd = 1:ID
        @views T_i_fd = rowIDT[iidd, 2]
        tempx2 = T_i_fd * (-0.5*num.nofeta*log(2π) - 0.5*logdetll)
        if simple_check(tempx2)
            likx += -1e99
        else
            likx += tempx2
        end
    end
    end

    eta = rho[po.begeta:po.endeta]
    δ2  = rho[po.begw]
    γ   = rho[po.begv]
    δ1  = rho[po.begz]
    gammap = rho[po.beggamma]
    gamma  = eigvalu.rymin/(1.0+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1.0+exp(gammap))

    hi_cur = exp.(Q*τ)
    hi_lag = exp.(Q_lag*τ)
    dhi    = hi_cur - hi_lag

    σᵤ² = exp(δ2)
    σᵤ  = exp(0.5*δ2)
    σᵥ² = exp(γ)
    μ   = δ1
    ϵ   = PorC*(y - x*β)

    T_fd   = rowIDT[1, 2]
    T_orig = T_fd + 1

    lndetIrhoWt = 0.0
    if length(Wy) == 1
        N = ID
        Wyt = kron(Wy[1], I(T_fd))
        ϵ = ϵ - PorC*gamma*Wyt*y - PorC*(eps*eta)
        lndetIrhoW = log(det(I(N) - gamma*Wy[1]))
        lndetIrhoWt = lndetIrhoW * T_fd
    end

    # __SSDGITE_PART2__

    Ainv = zeros(T_fd, T_fd)
    for j = 1:T_fd, k = 1:T_fd
        Ainv[j,k] = min(j,k) * (T_orig - max(j,k)) / T_orig
    end
    invPi = Ainv / σᵥ²
    lndetA = log(T_orig)

    lik = zero(eltype(y))
    @inbounds for iidd = 1:ID
        ind = rowIDT[iidd, 1]
        T_i_fd = rowIDT[iidd, 2]

        @views dhis = dhi[ind]
        @views ϵs   = ϵ[ind]

        sigs2 = 1.0 / (dhis'*invPi*dhis + 1.0/σᵤ²)
        mus   = (μ/σᵤ² - ϵs'*invPi*dhis) * sigs2
        es2   = -0.5 * ϵs'*invPi*ϵs
        KK    = -0.5*T_i_fd*log(2π) - 0.5*T_i_fd*log(σᵥ²) - 0.5*lndetA

        temp = KK + es2 + 0.5*((mus^2)/sigs2 - μ^2/σᵤ²) +
               0.5*log(sigs2) + log(normcdf(mus/sqrt(sigs2))) -
               0.5*log(σᵤ²) - log(normcdf(μ/σᵤ))

        if simple_check(temp)
            lik += -1e99
        else
            lik += temp
        end
    end

    return -(lik + lndetIrhoWt) - likx
  catch e
    return 1e100
  end
end


#* --- LL_T dispatches for Giannini 2025 ---

function LL_T(::Type{SSFGIH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN, IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho, eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing)
    Wy = _dicM[:wy]
    Q_lag = _dicM[:qvar_lag]
    llt = ssdgih(y, x, Q, Q_lag, Wy, PorC, num, po, rho, eigvalu, rowIDT)
    return llt
end

function LL_T(::Type{SSFGIEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN, IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho, eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing)
    Wy = _dicM[:wy]
    Q_lag = _dicM[:qvar_lag]
    llt = ssdgihe(y, x, Q, Q_lag, EN, IV, Wy, PorC, num, po, rho, eigvalu, rowIDT)
    return llt
end

function LL_T(::Type{SSFGIT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN, IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho, eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing)
    Wy = _dicM[:wy]
    Q_lag = _dicM[:qvar_lag]
    llt = ssdgit(y, x, Q, Q_lag, Wy, PorC, num, po, rho, eigvalu, rowIDT)
    return llt
end

function LL_T(::Type{SSFGIET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN, IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho, eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing)
    Wy = _dicM[:wy]
    Q_lag = _dicM[:qvar_lag]
    llt = ssdgite(y, x, Q, Q_lag, EN, IV, Wy, PorC, num, po, rho, eigvalu, rowIDT)
    return llt
end

    


function LL_T(::Type{SSFWHEH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdwhhe(y, x, Q, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFWHH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdwhh(y, x, Q, PorC, num, po, rho,  eigvalu, rowIDT ) 

    return llt
end



function ssdwhhe( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    phi = rho[po.begphi:po.endphi]
    nofiv = num.nofphi/num.nofeta
    eps = zeros(eltype(EN),num.nofobs,num.nofeta);
    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi
    @views LL = cov(eps);
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));
   try
    @floop begin
    @inbounds for iitt =1:num.nofobs
                tempx=-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                if simple_check(tempx)
                    likx += -1e9
                else
                    likx += tempx
                end
        end
    end
    ID = size(rowIDT,1)
    @floop begin
        @inbounds for iidd=1:ID
            @views T = rowIDT[iidd,2];
                    tempx2= (T-1) * (-0.5*num.nofeta*log(2*π)-0.5*logdetll);
                    if simple_check(tempx2)
                        likx += -1e99
                    else
                        likx += tempx2
                    end
                 end
        end
    eta = rho[po.begeta:po.endeta]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]
    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2)
    σᵤ= exp(0.5*δ2)
    σᵥ² = exp(γ)
    σᵥ = exp(0.5*γ)
    μ   = 0.0
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)
    @floop begin
        lik = zero(eltype(y));
    @inbounds  for iidd=1:ID
        @views T = rowIDT[iidd,2];
        @views onecol = ones(T, 1);
        @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
        @views invPi = 1/σᵥ²;
        @views lndetPi = log(σᵥ²);
            @views ind = rowIDT[iidd,1];
            @views his = IMT*(hi[ind]);
            @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
            @views sigs2 = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
            @views mus = (μ/σᵤ² - (ϵs'*his*invPi))*sigs2 ;
            @views es2 = -0.5*(ϵs'*ϵs*invPi );
            @views KK = -0.5*(T-1)*log(2 * π)-0.5*(T-1)*lndetPi;
            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e99
                    else
                        lik += temp
                    end
                end
        end
    return -lik-likx
    catch e
        return 1e100
    end
end


function ssdwhh( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,
     PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]
    δ2 = rho[po.begw]
    γ  = rho[po.begv]
    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2)
    σᵤ= exp(0.5*δ2)
    σᵥ² = exp(γ)
    σᵥ = exp(0.5*γ)
    μ   = 0.0
    ϵ = PorC*(y - x*β)
    ID = size(rowIDT,1)
    try
    @floop begin
        lik = zero(eltype(y));
    @inbounds  for iidd=1:ID
        @views T = rowIDT[iidd,2];
        @views onecol = ones(T, 1);
        @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
        @views invPi = 1/σᵥ²;
        @views lndetPi = log(σᵥ²);
        @views ind = rowIDT[iidd,1];
        @views his = IMT*(hi[ind]);
        @views ϵs  = ϵ[ind]   ;
        @views sigs2 = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
        @views mus = (μ/σᵤ² - (ϵs'*his*invPi))*sigs2 ;
        @views es2 = -0.5*(ϵs'*ϵs*invPi );
        @views KK = -0.5*(T-1)*log(2 * π)-0.5*(T-1)*lndetPi;
        @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + (normlogcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - (normlogcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e99
                    else
                        lik += temp
                    end
                end
        end
    return -lik
    catch e
        return 1e100
    end
end



function ssdwhte( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  EN,IV,
        PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
        β  = rho[1:po.endx]
        τ  = rho[po.begq:po.endq]
        phi = rho[po.begphi:po.endphi]
        ## calculate lkx
        nofiv = num.nofphi/num.nofeta
        eps = zeros(eltype(EN),num.nofobs,num.nofeta);
    
        # %%%%%%%%%%%%%%
        # @inbounds  for ii=1:num.nofeta
        #      eps[:,ii]=EN[:,ii]-IV*phi[((Int(ii)-1)*Int(nofiv)+1):(Int(ii)*Int(nofiv))];
        #  end
        @views phi = reshape(phi, :, num.nofeta)
        @views eps = EN- IV*phi
    
        # @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
        @views LL = cov(eps);
        @views logdetll = log(det(LL))
        @views invll = LL\I(num.nofeta)
        likx = zero(eltype(y));
    
       try
        @floop begin
        @inbounds for iitt =1:num.nofobs
                    tempx=-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
                    if simple_check(tempx)
                        likx += -1e9
                    else
                        likx += tempx
                    end # simple_check(tempx)
        
            end # iitt =1:num.nofobs
        end # @floop begin
    
        ID = size(rowIDT,1)
        @floop begin
            @inbounds for iidd=1:ID 
                @views T = rowIDT[iidd,2];
                        tempx2= (T-1) * (-0.5*num.nofeta*log(2*π)-0.5*logdetll);
                        if simple_check(tempx2)
                            likx += -1e99
                        else
                            likx += tempx2
                        end # simple_check(tempx2)
            
                     end # iidd=1:ID 
            end # @floop begin
    
        ## calculate lky
        
        eta = rho[po.begeta:po.endeta]
        δ2 = rho[po.begw]  
        γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
        δ1 = rho[po.begz]
    
    
        hi  = exp.(Q*τ)
        σᵤ²= exp(δ2) 
        σᵤ= exp(0.5*δ2) 
        σᵥ² = exp(γ)            # todo: 重新换一下字母 
        σᵥ = exp(0.5*γ)  
        μ   = δ1
        ϵ = PorC*(y - x*β)
        ID = size(rowIDT,1)
    
        @floop begin  
            lik = zero(eltype(y));
        @inbounds  for iidd=1:ID  
            @views T = rowIDT[iidd,2];
            @views onecol = ones(T, 1);
            @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
            @views invPi = 1/σᵥ²;
            @views lndetPi = log(σᵥ²);
    
                @views ind = rowIDT[iidd,1];
                @views his = IMT*(hi[ind]);
                @views ϵs  = ϵ[ind] - PorC*(eps[ind,:]*eta) ;
                @views sigs2 = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
                @views mus = (μ/σᵤ² - (ϵs'*his*invPi))*sigs2 ;
                @views es2 = -0.5*(ϵs'*ϵs*invPi );
                @views KK = -0.5*(T-1)*log(2 * π)-0.5*(T-1)*lndetPi;
    
                @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                                0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                                0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
                        if simple_check(temp)
                            lik += -1e99
                        else
                            lik += temp
                        end # simple_check(temp)
                    end # for ttt=1:ID
            end # begin
    
        return -lik-likx
        catch e
        # 处理异常的代码
        # println("操作失败，发生错误：$e")
            return 1e100
        end
    end
        
        
        
    
    
        
function ssdwht( y::Union{Vector,Matrix}, x::Matrix, Q::Matrix,  
        PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any} )
    β  = rho[1:po.endx]
    τ  = rho[po.begq:po.endq]

    δ2 = rho[po.begw]  
    γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
    δ1 = rho[po.begz]


    hi  = exp.(Q*τ)
    σᵤ²= exp(δ2) 
    σᵤ= exp(0.5*δ2) 
    σᵥ² = exp(γ)            # todo: 重新换一下字母 
    σᵥ = exp(0.5*γ)  
    μ   = δ1
    ϵ = PorC*(y - x*β)


    ID = size(rowIDT,1)

    try
    @floop begin  
        lik = zero(eltype(y));
    @inbounds  for iidd=1:ID  
        @views T = rowIDT[iidd,2];
        @views onecol = ones(T, 1);
        @views IMT = (I(T)-onecol*pinv(onecol'*onecol)*onecol');
        @views invPi = 1/σᵥ²;
        @views lndetPi = log(σᵥ²);

            @views ind = rowIDT[iidd,1];
            @views his = IMT*(hi[ind]);
            @views ϵs  = ϵ[ind]   ;
            @views sigs2 = 1.0 / ((his'*his*invPi) + 1/σᵤ²) ;
            @views mus = (μ/σᵤ² - (ϵs'*his*invPi))*sigs2 ;
            @views es2 = -0.5*(ϵs'*ϵs*invPi );
            @views KK = -0.5*(T-1)*log(2 * π)-0.5*(T-1)*lndetPi;


            @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                            0.5 * log(sigs2) + (normlogcdf(mus / sqrt(sigs2))) -
                            0.5 * log(σᵤ²) - (normlogcdf(μ / σᵤ))
                    if simple_check(temp)
                        lik += -1e99
                    else
                        lik += temp
                    end # simple_check(temp)
                end # for ttt=1:ID
        end # begin

    return -lik
    catch e
    # 处理异常的代码
    # println("操作失败，发生错误：$e")
        return 1e100
    end
end
        

function LL_T(::Type{SSFWHET}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdwhte(y, x, Q, EN, IV, PorC, num, po, rho,  eigvalu, rowIDT )  

    return llt
end
    
function LL_T(::Type{SSFWHT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN,IV, 
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

    llt = ssdwht(y, x, Q, PorC, num, po, rho,  eigvalu, rowIDT )  
    return llt
end




