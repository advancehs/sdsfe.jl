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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0 # δ1

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
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   # taup = rho[po.begtau]
   # tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   

    @views Mtau = 1;
    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   # rhomyp = rho[po.begrho]
   # rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1]) \I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt]) \I(N);
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = 0 # δ1
   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
                
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  
   μ   = 0 # δ1
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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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

   μ   = 0 # δ1

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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)  
   σᵥ = exp(0.5*γ)  

   μ   = 0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try
   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mtau = 1;
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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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

   μ   = 0 # δ1

   ϵ = PorC*(y - x * β)
   T = size(rowIDT,1)
   # print(T)
try

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];

    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
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
                    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β - eps  *eta)
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = Mtau*hi[ind];
                    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0

   ϵ =   PorC*(y - x*β - eps  *eta)
   T = size(rowIDT,1)

   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
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
    @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind];
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
   T = size(rowIDT,1)

   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   μ   = 0
   ϵ = PorC*(y - x*β- eps  *eta)
   T = size(rowIDT,1)



    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   # taup = rho[po.begtau]
   # tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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

    @views Mtau = 1;
    @views Mrho = (I(N)-rhomy*Wv[1]) \I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)

    @views lndetPi = log(detPi);
        @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   # rhomyp = rho[po.begrho]
   # rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = (I(N)-tau*Wu[ttt]) \I(N);
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

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
    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));   
    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
                
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));


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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
        
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

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

    @views Mtau = 1;
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
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@inbounds for ttt=1:T

    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);
            
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views detPi = det(Pi)
    @views lndetPi = log(detPi);
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
        
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
            
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                        0.5 * log(σᵤ²) - log(normcdf(μ / sqrt(σᵤ²)))
        
    if simple_check(temp)
        # print("a")
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
println("操作失败，发生错误：$e")
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
    @views N = rowIDT[1,2];

    @views Mtau = 1;
    @views invPi = I(N)/σᵥ²;
    @views lndetPi = N*log(σᵥ²);

        @floop begin
        @inbounds for ttt=1:T  
                @views ind = rowIDT[ttt,1];
                @views his = Mtau*hi[ind];
                @views ϵs  = ϵ[ind] ;
                @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β - eps  *eta)
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
                    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β   - eps  *eta)
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
                    @views ind = rowIDT[ttt,1];
                    @views his = Mtau*hi[ind];
                    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
                    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  - eps  *eta)
   T = size(rowIDT,1)

   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind];
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   gamma  = (eigvalu.rymin)/(1+exp(gammap))+(eigvalu.rymax)*exp(gammap)/(1+exp(gammap));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  - eps  *eta)
   T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[1]*y[ind] ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views lndetIrhoW = log(det(I(N)-gamma*Wy[ttt]));      
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]-PorC.*gamma*Wy[ttt]*y[ind] ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = lndetIrhoW-0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  - eps  *eta)
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
    @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind]- PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   tau  = (eigvalu.rumin)/(1+exp(taup))+(eigvalu.rumax)*exp(taup)/(1+exp(taup));


   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β   - eps  *eta)
   T = size(rowIDT,1)

   if length(Wu)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind];
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = (I(N)-tau*Wu[ttt])\I(N);
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind];
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   rhomy  = (eigvalu.rvmin)/(1+exp(rhomyp))+(eigvalu.rvmax)*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  
   μ   = δ1
   ϵ = PorC*(y - x*β  - eps  *eta)
   T = size(rowIDT,1)

   if length(Wv)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi =  (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
    @views N = rowIDT[1,2];
@floop begin

@inbounds for ttt=1:T

    @views Mtau = 1;
    @views Mrho =  (I(N)-rhomy*Wv[ttt])\I(N);   
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[ttt])')*(I(N)-rhomy*Wv[ttt])/σᵥ²;
    
    @views ind = rowIDT[ttt,1];
    @views his = Mtau*hi[ind];
    @views ϵs  = ϵ[ind] - PorC.* Mrho*(eps[ind,:]*eta) ;
    @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
    @views mus = (μ/σᵤ² - ϵs'*invPi*his)*sigs2 ;
    @views es2 = -0.5*ϵs'*invPi*ϵs ;
    @views KK = -0.5*N*log(2 * π)-0.5*lndetPi;
                
    @views temp = KK + es2 + 0.5 * (((mus ^ 2) / sigs2) - (μ^2 / σᵤ²) ) +
                    0.5 * log(sigs2) + log(normcdf(mus / sqrt(sigs2))) -
                    0.5 * log(σᵤ²) - log(normcdf(μ / σᵤ))
        
        if simple_check(temp)
            # print("a")
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
println("操作失败，发生错误：$e")
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
   ϵ = PorC*(y - x*β  - eps  *eta)
   T = size(rowIDT,1)



    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views Mtau = 1;
    @views invPi = 1/σᵥ²*I(N);
    @views lndetPi = N*log(σᵥ²);
        @floop begin
    @inbounds  for ttt=1:T  
            @views ind = rowIDT[ttt,1];
            @views his = Mtau*hi[ind];
            @views ϵs  = ϵ[ind] ;
            @views sigs2 = 1 / (his'*invPi*his + 1 /σᵤ²) ;
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
println("操作失败，发生错误：$e")
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






