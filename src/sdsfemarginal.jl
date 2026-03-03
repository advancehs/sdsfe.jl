#########################################################
#                                                       #
#  marginal effects of exogenous determinants on E(u)   #
#                                                       #
#########################################################

"""
    _safe_cholesky_L(A) -> LowerTriangular

安全的 Cholesky 分解：先尝试标准 cholesky，若矩阵不正定则通过
特征值修正（将负/零特征值截断为 ε）得到最近正定矩阵后再分解。
用于 MC 模拟中从方差-协方差矩阵生成随机扰动。
"""
function _safe_cholesky_L(A::Union{Symmetric{Float64, Matrix{Float64}}, Matrix{Float64}})
    As = A isa Symmetric ? A : Symmetric(A)
    C = cholesky(As; check=false)
    if issuccess(C)
        return C.L
    end
    # 特征值修正：最近正定矩阵
    F = eigen(As)
    ε = max(1e-10, eps(Float64) * maximum(abs.(F.values)))
    D_clamped = max.(F.values, ε)
    A_pd = Symmetric(F.vectors * Diagonal(D_clamped) * F.vectors')
    return cholesky(A_pd).L
end


#? ----------- truncated normal, marginal effect function -------


function marg_ssfoadt(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     
     taup = coef[pos.begtau]
     tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));
     Wu = _dicM[:wu]
    if Wu!=Nothing 
         if length(Wu)==1
              Mtau = (I(N)-tau*Wu[1])\I(N);
              hs = Mtau[iii]*h;
         else
              Mtau = (I(N)-tau*Wu[ttt])\I(N);
              hs = Mtau[iii]*h;
         end
    else
        hs = h;
    end
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFOADT}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfoadt(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  




function marg_ssfoadh(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0
     
     taup = coef[pos.begtau]
     tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));
     Wu = _dicM[:wu]
    if Wu!=Nothing 
         if length(Wu)==1
              Mtau = (I(N)-tau*Wu[1])\I(N);
              hs = Mtau[iii]*h;
         else
              Mtau = (I(N)-tau*Wu[ttt])\I(N);
              hs = Mtau[iii]*h;
         end
    else
        hs = h;
    end
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFOADH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     # mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfoadh(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:] ) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  




function marg_ssfoah(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0  # mu, a scalar
     
     taup = coef[pos.begtau]
     tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));
     Wu = _dicM[:wu]
     if Wu!=Nothing 

          if length(Wu)==1
               Mtau = (I(N)-tau*Wu[1])\I(N);
               hs = Mtau[iii]*h;
          else
               Mtau = (I(N)-tau*Wu[ttt])\I(N);
               hs = Mtau[iii]*h;
          end
     else
          hs=h
     end

     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFOAH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z ,eigvalu::NamedTuple ,rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    

      T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfoah(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  












function marg_ssfoat(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     
     taup = coef[pos.begtau]
     tau  = eigvalu.rumin/(1+exp(taup))+eigvalu.rumax*exp(taup)/(1+exp(taup));
     Wu = _dicM[:wu]
     if Wu!=Nothing 

          if length(Wu)==1
               Mtau = (I(N)-tau*Wu[1])\I(N);
               hs = Mtau[iii]*h;
          else
               Mtau = (I(N)-tau*Wu[ttt])\I(N);
               hs = Mtau[iii]*h;
          end
     else
          hs=h
     end

     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFOAT}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfoat(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  




# 对beta求导
function IrhoW(gamma::Float64, rowIDT::Matrix{Any} )

     Wy = _dicM[:wy]
     T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

        N = size(Wy[1], 1)   # 用空间权重矩阵维度，兼容 GI 个体索引 rowIDT
		 wirho = (I(N)-gamma*Wy[1])\I(N)
		 dire = tr(wirho)/N
		 indire = (sum(wirho) - tr(wirho))/N

	else
		NN=0.0
		dire=0.0
		indire=0.0
		for ttt=1:T
        @views N = rowIDT[ttt,2];

		 NN = NN +N
		 wirho = (I(N)-gamma*Wy[ttt])\I(N)
		 dire =dire+ tr(wirho)
		 indire =indire+ (sum(wirho) - tr(wirho))


        end
         dire = dire/NN 
		 indire = indire /NN
    end

    return dire,indire
end

# 对theta求导
function IrhoWW(gamma::Float64, rowIDT::Matrix{Any} )

	
     Wy = _dicM[:wy]
     Wx = _dicM[:wx]

     T = size(rowIDT,1)

	
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
        N = size(Wy[1], 1);

		 wirho = (I(N)-gamma*Wy[1])\I(N)*Wx[1]
		 dire = tr(wirho)/N
		 indire = (sum(wirho) - tr(wirho))/N

	else
		NN=0.0
		dire=0.0
		indire=0.0
		for ttt=1:T
			
        @views N = rowIDT[1,2];

		 NN = NN +N
		 wirho = (I(N)-gamma*Wy[ttt])\I(N)*Wx[ttt]
		 dire =dire+ tr(wirho)
		 indire =indire+ (sum(wirho) - tr(wirho))

        end
        dire = dire/NN 
		indire = indire /NN
    end	
	return dire, indire
end


# //对rho求导1
function IrhoWWIrhoW(gamma::Float64,dgamma::Float64, rowIDT::Matrix{Any} )
     Wy = _dicM[:wy]

     T = size(rowIDT,1)


   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
        N = size(Wy[1], 1);
         wirho = (I(N)-gamma*Wy[1])\I(N)*Wy[1]*(I(N)-gamma*Wy[1])*dgamma
         dire = tr(wirho)/N
         indire = (sum(wirho) - tr(wirho))/N
   else
        NN=0.0
        dire=0.0
        indire=0.0
        for ttt=1:T
        @views N = rowIDT[1,2];
    
         NN = NN +N
         wirho  = (I(N)-gamma*Wy[ttt])\I(N)*Wy[ttt]*(I(N)-gamma*Wy[ttt])*dgamma
         dire   = dire+ tr(wirho)
         indire = indire+ (sum(wirho) - tr(wirho))

        end
        dire = dire/NN 
		indire = indire /NN
   end	
	return dire, indire
end


# 对rho求导2
function IrhoWWIrhoWW(gamma::Float64,dgamma::Float64, rowIDT::Matrix{Any} )

     Wy = _dicM[:wy]
     Wx = _dicM[:wx]

     T = size(rowIDT,1)


   if length(Wy)==1

        N = size(Wy[1], 1);
		 wirho = (I(N)-gamma*Wy[1])\I(N)*Wy[1]*(I(N)-gamma*Wy[1])*Wx[1]*dgamma
		 dire = tr(wirho)/N
		 indire = (sum(wirho) - tr(wirho))/N
    else
		NN=0.0
		dire=0.0
		indire=0.0
		for ttt=1:T
            @views N = rowIDT[1,2];
    		 NN = NN +N
    		 wirho = (I(N)-gamma*Wy[ttt])\I(N)*Wy[ttt]*(I(N)-gamma*Wy[ttt])*Wx[ttt]*dgamma
    		 dire =dire+ tr(wirho)
    		 indire =indire+ (sum(wirho) - tr(wirho))
        end
        dire = dire/NN 
        indire = indire /NN
    end


	return dire, indire
end

# 对z求导
function IrhoWItauW(gamma::Float64,tau::Float64, jlms0, rowIDT::Matrix{Any} )

     Wy = _dicM[:wy]
     Wu = _dicM[:wu]

     T = size(rowIDT,1)
   if length(Wy)==1

     N_spatial = size(Wy[1], 1)
     N_rowIDT  = rowIDT[1,2]

     if N_rowIDT == N_spatial
       # 时间索引 rowIDT (KU/OA 等模型): rowIDT[t,1]=时间t的索引, rowIDT[t,2]=N
       NN=0.0; dire=0.0; total=0.0
       for ttt=1:T
            @views ind = rowIDT[ttt,1];
            @views jlms= jlms0[ind];
            N = N_spatial
            NN = NN +N
            if Wu!=Nothing
                 wirho = ((I(N)-gamma*Wy[1])\I(N))*((I(N)-tau*Wu[1])\I(N))*Diagonal(jlms)
            else
                 wirho = ((I(N)-gamma*Wy[1])\I(N))*Diagonal(jlms)
            end
            dire =dire+ tr(wirho)
            total =total+ (sum(wirho) )
       end
       dire = dire/NN
       total = total /NN
     else
       # 个体索引 rowIDT (GI 模型): rowIDT[i,1]=个体i的索引, rowIDT[i,2]=T_fd
       ID = T;  T_fd = N_rowIDT;  N = N_spatial
       Mgamma = (I(N) - gamma*Wy[1]) \ I(N)
       NN=0.0; dire=0.0; total=0.0
       for t in 1:T_fd
            jlms_t = [jlms0[rowIDT[i,1][t]] for i in 1:ID]
            if Wu!=Nothing
                 wirho = Mgamma*((I(N)-tau*Wu[1])\I(N))*Diagonal(vec(jlms_t))
            else
                 wirho = Mgamma*Diagonal(vec(jlms_t))
            end
            NN = NN + N
            dire = dire + tr(wirho)
            total = total + sum(wirho)
       end
       dire = dire/NN
       total = total/NN
     end

     else
		NN=0.0
		dire=0.0
		total=0.0
		for ttt=1:T
               @views ind = rowIDT[ttt,1];
               @views jlms= jlms0[ind];

            @views N = rowIDT[ttt,2];
    		 NN = NN +N
          if Wu!=Nothing 
               wirho = ((I(N)-gamma*Wy[ttt])\I(N))*((I(N)-tau*Wu[ttt])\I(N))*Diagonal(jlms)
    		else
               wirho = ((I(N)-gamma*Wy[ttt])\I(N))*Diagonal(jlms)
          end
               dire =dire+ tr(wirho)
    		 total =total+ (sum(wirho))
        end
        dire = dire/NN 
        total = total /NN
    end


	return dire, total
end

# 对z求导  gamma = 0
function IrhoWItauW2(tau::Float64, jlms0, rowIDT::Matrix{Any} )

     Wu = _dicM[:wu]

     T = size(rowIDT,1)
   if length(Wu)==1 
		 
     NN=0.0
     dire=0.0
     total=0.0
     for ttt=1:T
          @views ind = rowIDT[ttt,1];
          @views jlms= jlms0[ind];


       @views N = rowIDT[1,2];
          NN = NN +N
          if Wu!=Nothing 
               wirho = ((I(N)-tau*Wu[1])\I(N))*Diagonal(jlms)
          else
               wirho = Diagonal(jlms)
          end
          dire =dire+ tr(wirho)
          total =total+ (sum(wirho) )
     end
          dire = dire/NN 
          total = total /NN
     else
		NN=0.0
		dire=0.0
		total=0.0
		for ttt=1:T
               @views ind = rowIDT[ttt,1];
               @views jlms= jlms0[ind];

            @views N = rowIDT[ttt,2];
    		 NN = NN +N
          if Wu!=Nothing 
               wirho = ((I(N)-tau*Wu[ttt])\I(N))*Diagonal(jlms)
    		else
               wirho = Diagonal(jlms)
          end
               dire =dire+ tr(wirho)
    		 total =total+ (sum(wirho))
        end
        dire = dire/NN 
        total = total /NN
    end


	return dire, total
end



# function get_mareffx_single( b::Array{Float64, 1}, dire0::Float64, indire0::Float64, V::Matrix,gamma::Float64,dgamma::Float64,rowIDT::Matrix{Any})
# ## 这是delta method 的版本
#      if(length(b)==1)
#          dire = dire0 *b
#          indire = indire0*b
#          totale = dire + indire
#          dire1, indire1 = IrhoWWIrhoW(gamma,dgamma, rowIDT) 			
#          ddire = vcat(dire0 , dire1)
#          direse = sqrt(ddire'*V*ddire )
#      #     direse = sqrt(Complex(ddire'*V*ddire ))
#          inddire = vcat(indire0 , indire1)
#          indirese = sqrt(inddire'*V*inddire )
#          totalese = sqrt((ddire+inddire)'*V*(ddire+inddire))
 
#      else
#          dire1, indire1 = IrhoWW(gamma, rowIDT) 	
#          dire = dire0 *b[1] + dire1*b[2]
#          indire = indire0*b[1] + indire1*b[2]
#          totale = dire + indire
#          dire2, indire2 = IrhoWWIrhoW(gamma,dgamma, rowIDT) 	
#          dire3, indire3 = IrhoWWIrhoWW(gamma,dgamma, rowIDT) 	
#          ddire = vcat(dire0 , dire1 , (dire2*b[1]+dire3*b[2]))
#          direse = sqrt(ddire'*V*ddire )
#      #     direse = sqrt(Complex(ddire'*V*ddire ))
#          inddire = vcat(indire0 , indire1 , (indire2*b[1]+indire3*b[2]))
#          indirese =sqrt(inddire'*V*inddire) 
#          totalese = sqrt((ddire+inddire)'*V*(ddire+inddire))
         
#      end 
#      dire   = hcat(dire, direse)
#      indire = hcat(indire , indirese) 
#      totale = hcat(totale,totalese )
     
#      return dire,indire,totale
#  end
         

  
 
#  function get_mareffx(
      ## delta method  
#      pos::NamedTuple, coef::Array{Float64, 1},  var_cov_matrix::Matrix,
#      eigvalu::NamedTuple ,indices_list::Vector{Any}, rowIDT::Matrix{Any})
    
#      gammap = coef[pos.beggamma];
#      gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
#      dgamma =  exp(gammap)*((eigvalu.rymax-eigvalu.rymin)/(1+exp(gammap))^2);
    

#     totalemat = Array{Any}(undef,0,2)
#     diremat   = Array{Any}(undef,0,2)
#     indiremat = Array{Any}(undef,0,2)
    
#     dire0, indire0= IrhoW(gamma,  rowIDT)
    
#     for (symbol, indices) in indices_list
#         b = coef[(pos.begx-1).+indices]
#         V = var_cov_matrix[vcat((pos.begx-1).+indices,pos.beggamma),vcat((pos.begx-1).+indices,pos.beggamma)]
#         dire,indire,totale = get_mareffx_single(b, dire0,indire0,  V,gamma,dgamma,rowIDT)
# 		totalemat = vcat(totalemat , totale) 
# 		diremat   = vcat(diremat , dire) 
# 		indiremat = vcat(indiremat , indire) 
#     end
#     vars =  [string(pair[1]) for pair in indices_list]
#     indiremat = hcat(indiremat, (indiremat[:,1])./indiremat[:,2],(1.0.-(normcdf.(abs.((indiremat[:,1])./indiremat[:,2])))))
#     indiremat = hcat(vars,indiremat)
#     indiremat =  vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , indiremat)   
    
#     diremat   = hcat(diremat, (diremat[:,1])./diremat[:,2],(1.0.-(normcdf.(abs.((diremat[:,1])./diremat[:,2])))))
#     diremat   = hcat(vars,diremat)
#     diremat = vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , diremat)   

#     totalemat = hcat(totalemat, (totalemat[:,1])./totalemat[:,2],(1.0.-(normcdf.(abs.((totalemat[:,1])./totalemat[:,2])))))
#     totalemat = hcat(vars,totalemat)
#     totalemat =  vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , totalemat)   
#     return totalemat, diremat, indiremat
    
# end

function bmc(V::LowerTriangular{Float64, Matrix{Float64}})
     random_vector = randn(size(V, 1), 1)
     result = V * random_vector
     return result
 end

function get_mareffx_single(pos::NamedTuple, coef::Array{Float64, 1},indices, dire0::Float64, 
     indire0::Float64, L::LowerTriangular,gamma::Float64,dgamma::Float64,rowIDT::Matrix{Any})
     ## 这是mc method 的版本

     b = coef[(pos.begx-1).+indices]

     if(length(b)==1)
          diret = dire0 *b
          indiret = indire0*b
          total = diret + indiret
          
          diretmat_mc = Array{Any}(undef,0,1)
          indiretmat_mc   = Array{Any}(undef,0,1)
          totalmat_mc = Array{Any}(undef,0,1)

          ## get sd with mc method
          for _ in range(1, 500)
               coef_mc = coef + bmc(L)
               b_mc = coef_mc[(pos.begx-1).+indices]
               diret_mc = dire0 *b_mc[1] 
               indiret_mc = indire0*b_mc[1] 
               total_mc = diret_mc + indiret_mc

               diretmat_mc   = vcat(diretmat_mc , diret_mc) 
               indiretmat_mc = vcat(indiretmat_mc , indiret_mc)   
               totalmat_mc = vcat(totalmat_mc , total_mc) 
          end

          diret_sd = std(diretmat_mc)
          indiret_sd = std(indiretmat_mc)
          total_sd = std(totalmat_mc)
     
     else
          dire1, indire1 = IrhoWW(gamma, rowIDT) 	
          diret = dire0 *b[1] + dire1*b[2]
          indiret = indire0*b[1] + indire1*b[2]
          total = diret + indiret

          diretmat_mc = Array{Any}(undef,0,1)
          indiretmat_mc   = Array{Any}(undef,0,1)
          totalmat_mc = Array{Any}(undef,0,1)

          ## get sd with mc method
          for _ in range(1, 500)
               coef_mc = coef + bmc(L)
               b_mc = coef_mc[(pos.begx-1).+indices]
               diret_mc = dire0 *b_mc[1] + dire1*b_mc[2]
               indiret_mc = indire0*b_mc[1] + indire1*b_mc[2]
               total_mc = diret_mc + indiret_mc

               diretmat_mc   = vcat(diretmat_mc , diret_mc) 
               indiretmat_mc = vcat(indiretmat_mc , indiret_mc)   
               totalmat_mc = vcat(totalmat_mc , total_mc) 
          end

          diret_sd = std(diretmat_mc)
          indiret_sd = std(indiretmat_mc)
          total_sd = std(totalmat_mc)
          # println("aaaaaaaaa",total_sd)

     end 

     diret_   = hcat(diret, diret_sd)
     indiret_ = hcat(indiret , indiret_sd) 
     total_ = hcat(total,total_sd )

     return diret_,indiret_,total_
end  

function get_mareffx(
     ## mc method 
     pos::NamedTuple, coef::Array{Float64, 1},  var_cov_matrix::Symmetric{Float64, Matrix{Float64}},
     eigvalu::NamedTuple ,indices_list::Vector{Any}, rowIDT::Matrix{Any})
    
     gammap = coef[pos.beggamma];
     gamma  = eigvalu.rymin/(1+exp(gammap))+eigvalu.rymax*exp(gammap)/(1+exp(gammap));
     dgamma =  exp(gammap)*((eigvalu.rymax-eigvalu.rymin)/(1+exp(gammap))^2);

     L = _safe_cholesky_L(var_cov_matrix)

    totalemat = Array{Any}(undef,0,2)
    diremat   = Array{Any}(undef,0,2)
    indiremat = Array{Any}(undef,0,2)

    dire0, indire0= IrhoW(gamma,  rowIDT)
    
    for (_, indices) in indices_list
        dire,indire,totale = get_mareffx_single(pos,coef,indices, dire0,indire0, L, gamma,dgamma,rowIDT)
		totalemat = vcat(totalemat , totale) 
		diremat   = vcat(diremat , dire) 
		indiremat = vcat(indiremat , indire) 
    end
    vars =  [string(pair[1]) for pair in indices_list]
    indiremat = hcat(indiremat, (indiremat[:,1])./indiremat[:,2],(1.0.-(normcdf.(abs.((indiremat[:,1])./indiremat[:,2])))))
    indiremat = hcat(vars,indiremat)
    indiremat =  vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , indiremat)   
    
    diremat   = hcat(diremat, (diremat[:,1])./diremat[:,2],(1.0.-(normcdf.(abs.((diremat[:,1])./diremat[:,2])))))
    diremat   = hcat(vars,diremat)
    diremat = vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , diremat)   

    totalemat = hcat(totalemat, (totalemat[:,1])./totalemat[:,2],(1.0.-(normcdf.(abs.((totalemat[:,1])./totalemat[:,2])))))
    totalemat = hcat(vars,totalemat)
    totalemat =  vcat(["var" "Coef." "Std. Err." "z" "P>|z|"] , totalemat)   
    return totalemat, diremat, indiremat
    
end

function get_margeffu_single(pos::NamedTuple, coef::Array{Float64, 1}, indices,
     Q_mat::Matrix{Float64}, W_mat::Matrix{Float64}, Z_mat::Union{Matrix{Float64}, Nothing},
     obs_ci::Vector{Int}, A_diag::Vector{Float64}, A_colsum::Vector{Float64},
     L::LowerTriangular, modelid, target_k::Int)
     ## 修正版：使用完整梯度方法（2026-03-03）与 Henderson 45度图一致

     τ_coef = coef[pos.begq:pos.endq]
     δ_coef = coef[pos.begw:pos.endw]
     nq = length(τ_coef)
     nw = length(δ_coef)
     nobs = size(Q_mat, 1)

     # 检测模型类型
     # 半正态模型：所有以 H 结尾的模型（SSFKUH, SSFKUEH, SSFOAH, SSFOADH, SSFKKH, SSFKKEH, SSFWHH, SSFWHEH, SSFGIH, SSFGIEH）
     # 截断正态模型：所有以 T 结尾的模型（SSFKUT, SSFKUET, SSFOAT, SSFOADT, SSFKKT, SSFKKET, SSFWHT, SSFWHET, SSFGIT, SSFGIET）
     modelid_str = string(modelid)
     is_half = endswith(modelid_str, "H")

     raw_me = zeros(nobs)

     if is_half
         _Eu_half = let τ_coef=τ_coef, δ_coef=δ_coef, nq=nq, nw=nw
             x -> begin
                 q = x[1:nq]; w = x[nq+1:nq+nw]
                 h = exp(dot(q, τ_coef))
                 σᵤ = exp(0.5 * dot(w, δ_coef))
                 return h * σᵤ * sqrt(2.0/π)
             end
         end
         for i in 1:nobs
             qw = vcat(Q_mat[i,:], W_mat[i,:])
             grad = ForwardDiff.gradient(_Eu_half, qw)
             raw_me[i] = grad[target_k]
         end
     else
         μ_coef = coef[pos.begz:pos.endz]
         for i in 1:nobs
             z_i = Z_mat[i, :]
             μ_i = dot(z_i, μ_coef)
             _Eu_trun = let τ_coef=τ_coef, δ_coef=δ_coef, μ_i=μ_i, nq=nq, nw=nw
                 x -> begin
                     q = x[1:nq]; w = x[nq+1:nq+nw]
                     h = exp(dot(q, τ_coef))
                     σᵤ = exp(0.5 * dot(w, δ_coef))
                     Λ = μ_i / σᵤ
                     return h * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
                 end
             end
             qw = vcat(Q_mat[i,:], W_mat[i,:])
             grad = ForwardDiff.gradient(_Eu_trun, qw)
             raw_me[i] = grad[target_k]
         end
     end

     direct_vec = [A_diag[obs_ci[i]] * raw_me[i] for i in 1:nobs]
     total_vec = [A_colsum[obs_ci[i]] * raw_me[i] for i in 1:nobs]
     indirect_vec = total_vec .- direct_vec

     diret = mean(direct_vec)
     total = mean(total_vec)
     indiret = mean(indirect_vec)

     diretmat_mc = Float64[]
     indiretmat_mc = Float64[]
     totalmat_mc = Float64[]

     for _ in 1:500
         coef_mc = coef + bmc(L)
         τ_coef_mc = coef_mc[pos.begq:pos.endq]
         δ_coef_mc = coef_mc[pos.begw:pos.endw]
         raw_me_mc = zeros(nobs)

         if is_half
             _Eu_half_mc = let τ_coef_mc=τ_coef_mc, δ_coef_mc=δ_coef_mc, nq=nq, nw=nw
                 x -> begin
                     q = x[1:nq]; w = x[nq+1:nq+nw]
                     h = exp(dot(q, τ_coef_mc))
                     σᵤ = exp(0.5 * dot(w, δ_coef_mc))
                     return h * σᵤ * sqrt(2.0/π)
                 end
             end
             for i in 1:nobs
                 qw = vcat(Q_mat[i,:], W_mat[i,:])
                 grad_mc = ForwardDiff.gradient(_Eu_half_mc, qw)
                 raw_me_mc[i] = grad_mc[target_k]
             end
         else
             μ_coef_mc = coef_mc[pos.begz:pos.endz]
             for i in 1:nobs
                 z_i = Z_mat[i, :]
                 μ_i_mc = dot(z_i, μ_coef_mc)
                 _Eu_trun_mc = let τ_coef_mc=τ_coef_mc, δ_coef_mc=δ_coef_mc, μ_i_mc=μ_i_mc, nq=nq, nw=nw
                     x -> begin
                         q = x[1:nq]; w = x[nq+1:nq+nw]
                         h = exp(dot(q, τ_coef_mc))
                         σᵤ = exp(0.5 * dot(w, δ_coef_mc))
                         Λ = μ_i_mc / σᵤ
                         return h * σᵤ * (Λ + normpdf(Λ) / normcdf(Λ))
                     end
                 end
                 qw = vcat(Q_mat[i,:], W_mat[i,:])
                 grad_mc = ForwardDiff.gradient(_Eu_trun_mc, qw)
                 raw_me_mc[i] = grad_mc[target_k]
             end
         end

         direct_vec_mc = [A_diag[obs_ci[i]] * raw_me_mc[i] for i in 1:nobs]
         total_vec_mc = [A_colsum[obs_ci[i]] * raw_me_mc[i] for i in 1:nobs]
         indirect_vec_mc = total_vec_mc .- direct_vec_mc

         push!(diretmat_mc, mean(direct_vec_mc))
         push!(totalmat_mc, mean(total_vec_mc))
         push!(indiretmat_mc, mean(indirect_vec_mc))
     end

     diret_sd = std(diretmat_mc)
     indiret_sd = std(indiretmat_mc)
     total_sd = std(totalmat_mc)

     diret_ = hcat(diret, diret_sd)
     indiret_ = hcat(indiret, indiret_sd)
     total_ = hcat(total, total_sd)

     return diret_, indiret_, total_
end 

function get_margeffu(
     ## 修正版：使用完整梯度方法（2026-03-03）
     jlms0, pos::NamedTuple, coef::Array{Float64, 1},  var_cov_matrix::Symmetric{Float64, Matrix{Float64}},
     eigvalu::NamedTuple, indices_listz::Vector{Any}, rowIDT::Matrix{Any};
     qvar=nothing, wvar=nothing, zvar=nothing, modelid=nothing)

     L = _safe_cholesky_L(var_cov_matrix)

     # 获取数据矩阵
     if qvar === nothing || wvar === nothing
         error("需要传入 qvar 和 wvar 参数")
     end

     Q_mat = Float64.(Matrix(qvar))
     W_mat = Float64.(Matrix(wvar))
     Z_mat = (zvar === nothing || (isa(zvar, Tuple) && isempty(zvar))) ? nothing : Float64.(Matrix(zvar))

     # 获取模型 ID
     if modelid === nothing
         modelid = _dicM[:modelid]
     end

     # 面板结构
     T = size(rowIDT, 1)
     N_cities = rowIDT[1, 2]

     # 观测到城市的映射
     obs_ci = zeros(Int, size(Q_mat, 1))
     for ttt in 1:T
         ind = rowIDT[ttt, 1]
         for (local_i, global_i) in enumerate(ind)
             obs_ci[global_i] = local_i
         end
     end

     # 计算空间乘数
     Wu = _dicM[:wu]
     Wy = _dicM[:wy]

     if Wy !== Nothing
          gammap = coef[pos.beggamma]
          gamma = eigvalu.rymin/(1+exp(gammap)) + eigvalu.rymax*exp(gammap)/(1+exp(gammap))
          A = inv(I(N_cities) - gamma * Wy[1])
          A_diag = diag(A)
          A_colsum = vec(sum(A, dims=1))
     else
          A_diag = ones(N_cities)
          A_colsum = ones(N_cities)
     end

     # 对每个变量计算边际效应
     totalemat = Array{Any}(undef, 0, 2)
     diremat = Array{Any}(undef, 0, 2)
     indiremat = Array{Any}(undef, 0, 2)

     for (var_symbol, indices) in indices_listz
         target_k = indices[1]

         dire, indire, totale = get_margeffu_single(
             pos, coef, indices,
             Q_mat, W_mat, Z_mat,
             obs_ci, A_diag, A_colsum,
             L, modelid, target_k
         )

         totalemat = vcat(totalemat, totale)
         diremat = vcat(diremat, dire)
         indiremat = vcat(indiremat, indire)
     end

     # 格式化输出
     vars = [string(pair[1]) for pair in indices_listz]

     indiremat = hcat(indiremat, (indiremat[:,1])./indiremat[:,2], (1.0 .- normcdf.(abs.((indiremat[:,1])./indiremat[:,2]))))
     indiremat = hcat(vars, indiremat)
     indiremat = vcat(["var" "Coef." "Std. Err." "z" "P>|z|"], indiremat)

     diremat = hcat(diremat, (diremat[:,1])./diremat[:,2], (1.0 .- normcdf.(abs.((diremat[:,1])./diremat[:,2]))))
     diremat = hcat(vars, diremat)
     diremat = vcat(["var" "Coef." "Std. Err." "z" "P>|z|"], diremat)

     totalemat = hcat(totalemat, (totalemat[:,1])./totalemat[:,2], (1.0 .- normcdf.(abs.((totalemat[:,1])./totalemat[:,2]))))
     totalemat = hcat(vars, totalemat)
     totalemat = vcat(["var" "Coef." "Std. Err." "z" "P>|z|"], totalemat)

     return totalemat, diremat, indiremat
end


function nonConsDataFrame(D::DataFrame, M::Matrix)
     # Given a DataFrame containing the marginal effects 
     # of a set of exogenous determinants $(x1, x2, ..., xn)$
     # on E(u), it return the DataFrame where the marginal 
     # effect of constant $x$s are removed.
 
     # D: the marginal effect DataFrame; 
     # M: the matrix of (x1, .., xn) where the marginal 
     #    efect is calculated from.
 
    counter = 0      
    for w in collect(names(D),)
         counter += 1
         if length(unique(M[:, counter])) == 1 # is a constant
             select!(D, Not(Symbol(w)))
         end
    end 
    return D
 end
 
 
 
 function addDataFrame(Main::DataFrame, A::DataFrame)
       # Combine two DataFrame with unions of columns.
       # For same-name columns, the values are added together.
 
     for k in collect(names(A),) # deal with the wvar
               if k ∈ names(Main)
                    Main[:, Symbol(k)] = Main[:, Symbol(k)] + A[:, Symbol(k)]
               else 
                    insertcols!(Main, Symbol(k) => A[:, Symbol(k)])
               end
      end 
      return Main  
 end 
 






 function marg_ssfkuh(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0
     
     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


 
function get_marg(::Type{SSFKUH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     # mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkuh(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:] ) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  





function marg_ssfkut(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     

     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFKUT}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkut(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  



function marg_ssfkueh(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0
     
     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


 
function get_marg(::Type{SSFKUEH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     # mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkueh(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:] ) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  



function marg_ssfkuet(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     

     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFKUET}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkuet(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  







function marg_ssfkkh(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0
     
     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


 
function get_marg(::Type{SSFKKH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     # mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkkh(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:] ) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  





function marg_ssfkkt(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     

     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFKKT}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkkt(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  



function marg_ssfkkeh(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = 0.0
     
     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


 
function get_marg(::Type{SSFKKEH}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix, z, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     # mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkkeh(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                   ),
                              vcat(  Q[iii,:], W[iii,:] ) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               # mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     # mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     # mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     # margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  



function marg_ssfkket(ttt::Int, iii::Int, N::Int,# PorC::Int64, 
     pos::NamedTuple, eigvalu::NamedTuple,coef::Array{Float64, 1},
      Qmarg, Wmarg,Zmarg)

     h = exp(Qmarg'*coef[pos.begq : pos.endq])
     σᵤ = exp(0.5 * Wmarg'*coef[pos.begw : pos.endw]) # σᵤ
     μ = Zmarg'*coef[pos.begz : pos.endz]  # mu, a scalar
     

     hs = h;
    
     hsμ  = hs*μ
     hsσᵤ = hs*σᵤ 
      Λ  = hsμ/hsσᵤ 
    
     uncondU = hsσᵤ* (Λ + normpdf(Λ) / normcdf(Λ)) # kx1
   
end   


#? -- panel FE Orea and Al, truncated normal, , get marginal effect ----
 
function get_marg(::Type{SSFKKET}, 
     pos::NamedTuple, num::NamedTuple, coef::Array{Float64, 1}, 
     Q::Matrix, W::Matrix,Z::Matrix, eigvalu::NamedTuple, rowIDT::Matrix{Any})

        #* Note that Y and X are within-transformed by `getvar`, 
        #* but Q, W, V are still in the original level.

     mm_q = Array{Float64}(undef, num.nofq, num.nofobs)                  
     mm_w = Array{Float64}(undef, num.nofw, num.nofobs)    
     mm_z = Array{Float64}(undef, num.nofz, num.nofobs)  

     T = size(rowIDT,1)

     @inbounds for ttt=1:T
          ind = rowIDT[ttt,1];
          for iii in ind
               @views N = rowIDT[1,2];


               @views marg = ForwardDiff.gradient(marg -> marg_ssfkket(ttt, iii,N, pos,eigvalu, coef, 
                                                  marg[1 : num.nofq],
                                                  marg[num.nofq+1 : num.nofq+num.nofw],
                                                  marg[num.nofq+num.nofw+1 : num.nofq+num.nofw+num.nofz] ),
                              vcat(  Q[iii,:], W[iii,:],Z[iii,:]) );                            

               mm_q[:,iii] = marg[1 : num.nofq]
               mm_w[:,iii] = marg[num.nofq+1 : num.nofq+num.nofw]
               mm_z[:,iii] = marg[num.nofq+num.nofw+1 : end]
          end # iii in ind
     end  #  ttt=1:T

     margeff = DataFrame(mm_q', _dicM[:hscale])
     mm_w = DataFrame(mm_w', _dicM[:σᵤ²])
     mm_z = DataFrame(mm_z', _dicM[:μ]) # the base set

     #* purge off the constant var's marginal effect from the DataFrame
     margeff = nonConsDataFrame(margeff, Q)
     mm_w = nonConsDataFrame(mm_w, W)
     mm_z = nonConsDataFrame(mm_z, Z)

     #* if same var in different equations, add up the marg eff
     margeff = addDataFrame(margeff, mm_w)
     margeff = addDataFrame(margeff, mm_z)

      #* prepare info for printing
      margMean = (; zip(Symbol.(names(margeff)) , round.(mean.(eachcol(margeff)); digits=5))...)

      #* modify variable names to indicate marginal effects
      newname = Symbol.(fill("marg_", (size(margeff,2), 1)) .* names(margeff))
      margeff = rename!(margeff, vec(newname))

     return  margeff, margMean
end  

