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

function simple_check2(xs)
    any(x -> isless(x,0.0)  ,sigs2)
end
function LL_T(::Type{SSFOAH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z,en,iv,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

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

try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    @views Pi = σᵥ²*(Mrho*Mrho');
    
    @views lndetPi = log(det(Pi));
    
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
                    # print("a")
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
    @views lndetPi = log(det(Pi));
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
#     # 计算 lik
# # if simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
# #     return 1e9
# # else # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#     # print(sigs2)
#    lik = KK + es2 + 0.5 * ((mus .^ 2) ./ sigs2 .- μ^2 / σᵤ²) +
#                  0.5 * log.(sigs2) + log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                  0.5 * log(σᵤ²) .- log(normcdf(μ / sqrt(σᵤ²)))
#    lik = -lik

#    # 计算 ls
#    ls = sum(lik)
    
#    if any(isnan.(ls)) || any(!isreal.(ls)) || any(isinf.(ls))
#           idx1 = findall(isinf.(lik))
#           idx2 = findall(isnan.(lik))
#           idx = [idx1; idx2]
#           a = copy(lik)  # 使用 copy() 函数创建 lik 的副本
#           a[idx] .= 1e9
#           ls = sum(a)
#    end
# # end # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#    return ls

    end
end



function LL_T(::Type{SSFOAT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, en, iv,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

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
try
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1]));   
    @views Mtau_before = I(N)-tau*Wu[1]
    @views Mtau_before = broadcast(x -> isnan(x) || isinf(x) ? rand(1000000:2000000) : x, Mtau_before)
    @views Mtau = (Mtau_before)\I(N);
    
    @views Mrho_before = I(N)-rhomy*Wv[1]
    @views Mrho_before = broadcast(x -> isnan(x) || isinf(x) ?  1e-7 + rand() * (2e-7 - 1e-7) : x, Mrho_before)
    @views Mrho =  (Mrho_before)\I(N);


    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    
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
                    # print("a")
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
    @views lndetPi = log(det(Pi));
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

#     # 计算 lik
# # if simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
# #     return 1e9
# # else # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#     # print(sigs2)
#    lik = KK + es2 + 0.5 * ((mus .^ 2) ./ sigs2 .- μ^2 / σᵤ²) +
#                  0.5 * log.(sigs2) + log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                  0.5 * log(σᵤ²) .- log(normcdf(μ / sqrt(σᵤ²)))
#    lik = -lik

#    # 计算 ls
#    ls = sum(lik)
    
#    if any(isnan.(ls)) || any(!isreal.(ls)) || any(isinf.(ls))
#           idx1 = findall(isinf.(lik))
#           idx2 = findall(isnan.(lik))
#           idx = [idx1; idx2]
#           a = copy(lik)  # 使用 copy() 函数创建 lik 的副本
#           a[idx] .= 1e9
#           ls = sum(a)
#    end
# # end # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#    return ls
end
end






function LL_T(::Type{SSFOADH}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, 
    EN::Matrix,IV::Matrix, Wy::Matrix, Wu::Matrix, Wv::Matrix,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

   β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%

    @views phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));

try 
@floop begin

   @inbounds  for iitt =1:num.nofobs
        @views tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
        if simple_check(tempx)
            print("tempx")
            @views likx += -1e9
        else
            @views likx += tempx
        end # simple_check(temp)
    end # iitt =1:num.nofobs
end # @floop begin
## calculate lky

   eta = rho[po.begeta:po.endeta]
   δ2 = rho[po.begw]  
   γ  = rho[po.begv]  # May rho[po.begw : po.endw][1]
   # δ1 = rho[po.begz]
   gammap = rho[po.beggamma]
   gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));

   taup = rho[po.begtau]
   tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));

   rhomyp = rho[po.begrho]
   rhomy  = eigvalu.rvmin/(1+exp(rhomyp))+eigvalu.rvmax*exp(rhomyp)/(1+exp(rhomyp));

   hi  = exp.(Q*τ)
   σᵤ²= exp(δ2) 
   σᵤ= exp(0.5*δ2) 
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  

   μ   = 0

   ϵ = PorC*(y - x*β  )
   T = size(rowIDT,1)
   # print(T)

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
                    # print("a")
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

#     # 计算 lik
# # if simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
# #     return 1e9
# # else # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#     # print(sigs2)
#    lik = KK + es2 + 0.5 * ((mus .^ 2) ./ sigs2 .- μ^2 / σᵤ²) +
#                  0.5 * log.(sigs2) + log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                  0.5 * log(σᵤ²) .- log(normcdf(μ / sqrt(σᵤ²)))
#    lik = -lik

#    # 计算 ls
#    ls = sum(lik)
    
#    if any(isnan.(ls)) || any(!isreal.(ls)) || any(isinf.(ls))
#           idx1 = findall(isinf.(lik))
#           idx2 = findall(isnan.(lik))
#           idx = [idx1; idx2]
#           a = copy(lik)  # 使用 copy() 函数创建 lik 的副本
#           a[idx] .= 1e9
#           ls = sum(a)
#    end
# # end # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#    return ls
    # print(lik)
    # print(likx)
end
end










function LL_T(::Type{SSFOADT}, y::Union{Vector,Matrix}, x::Matrix, Q::Matrix, w, v, z, EN::Matrix, IV::Matrix,
    Wy::Matrix, Wu::Matrix, Wv::Matrix,
    PorC::Int64, num::NamedTuple, po::NamedTuple, rho,  eigvalu::NamedTuple, rowIDT::Matrix{Any}, ::Nothing) 

   β  = rho[1:po.endx]
   τ  = rho[po.begq:po.endq]
   phi = rho[po.begphi:po.endphi]
## calculate lkx
   nofiv = num.nofphi/num.nofeta
   eps = zeros(eltype(EN),num.nofobs,num.nofeta);

   # %%%%%%%%%%%%%%

    phi = reshape(phi, :, num.nofeta)
    @views eps = EN- IV*phi

    @views LL = ((eps .- mean(eps, dims=1))' * (eps .- mean(eps, dims=1)) )/ num.nofobs
    @views logdetll = log(det(LL))
    @views invll = LL\I(num.nofeta)
    likx = zero(eltype(y));
    try
    @floop begin
    @inbounds for iitt =1:num.nofobs
        @views  tempx=-0.5*num.nofeta*log(2*π)-0.5*logdetll-0.5*tr(invll*eps[iitt,:]'*eps[iitt,:]);
        if simple_check(tempx)
            print("tempx")
            likx += -1e9
        else
            likx += tempx
        end # simple_check(temp)
    end # iitt =1:num.nofobs
    end # @floop begin

## calculate lky

   eta = rho[po.begeta:po.endeta]
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
   σᵥ² = exp(γ)            # todo: 重新换一下字母 
   σᵥ = exp(0.5*γ)  

   μ   = δ1

   ϵ = PorC*(y - x*β )
   T = size(rowIDT,1)

   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

    lik = zero(eltype(y));
    @views N = rowIDT[1,2];
    @views lndetIrhoW = log(det(I(N)-gamma*Wy[1])); 
        
    @views Mtau = (I(N)-tau*Wu[1])\I(N);
    @views Mrho =  (I(N)-rhomy*Wv[1])\I(N);
    
    @views Pi = σᵥ²*(Mrho*Mrho');
    @views lndetPi = log(det(Pi));
    @views invPi = (I(N)-rhomy*(Wv[1])')*(I(N)-rhomy*Wv[1])/σᵥ²;
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
                    # print("a")
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
    
#     # 计算 lik
# # if simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
# #     return 1e9
# # else # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#     # print(sigs2)
#    lik = KK + es2 + 0.5 * ((mus .^ 2) ./ sigs2 .- μ^2 / σᵤ²) +
#                  0.5 * log.(sigs2) + log.(normcdf.(mus ./ sqrt.(sigs2))) .-
#                  0.5 * log(σᵤ²) .- log(normcdf(μ / sqrt(σᵤ²)))
#    lik = -lik

#    # 计算 ls
#    ls = sum(lik)
    
#    if any(isnan.(ls)) || any(!isreal.(ls)) || any(isinf.(ls))
#           idx1 = findall(isinf.(lik))
#           idx2 = findall(isnan.(lik))
#           idx = [idx1; idx2]
#           a = copy(lik)  # 使用 copy() 函数创建 lik 的副本
#           a[idx] .= 1e9
#           ls = sum(a)
#    end
# # end # simple_check2( normcdf(μ / sqrt(σᵤ²)))  || simple_check2(sigs2)
#    return ls
end
end













  