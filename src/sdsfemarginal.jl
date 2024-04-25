#########################################################
#                                                       #
#  marginal effects of exogenous determinants on E(u)   #
#                                                       #
#########################################################


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
        Mtau = 1;
        hs = 1*h;
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
     μ = 0
     
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
        Mtau = 1;
        hs = 1*h;
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
     μ = 0  # mu, a scalar
     
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

    return dire,indire
end

# 对theta求导
function IrhoWW(gamma::Float64, rowIDT::Matrix{Any} )

	
     Wy = _dicM[:wy]
     Wx = _dicM[:wx]

     T = size(rowIDT,1)

	
   if length(Wy)==1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
        @views N = rowIDT[1,2];
            
		 wirho = (I(N)-gamma*Wy[1])\I(N)*Wx[1]
		 dire = tr(wirho)/N
		 indire = (sum(wirho) - tr(wirho))/N
    
	else
		NN=0
		dire=0
		indire=0
		for ttt=i:T
			
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
        @views N = rowIDT[1,2];  
         wirho = (I(N)-gamma*Wy[1])\I(N)*Wy[1]*(I(N)-gamma*Wy[1])*dgamma
         dire = tr(wirho)/N
         indire = (sum(wirho) - tr(wirho))/N
   else
        NN=0
        dire=0
        indire=0 
        for ttt=i:T
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
		 
        @views N = rowIDT[1,2];  
		 wirho = (I(N)-gamma*Wy[1])\I(N)*Wy[1]*(I(N)-gamma*Wy[1])*Wx[1]*dgamma
		 dire = tr(wirho)/N
		 indire = (sum(wirho) - tr(wirho))/N
    else
		NN=0
		dire=0
		indire=0
		for ttt=i:T
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



function get_mareffx_single( b::Array{Float64, 1}, dire0::Float64, indire0::Float64, V::Matrix,gamma::Float64,dgamma::Float64,rowIDT::Matrix{Any})

     if(length(b)==1)
         dire = dire0 *b
         indire = indire0*b
         totale = dire + indire
         dire1, indire1 = IrhoWWIrhoW(gamma,dgamma, rowIDT) 			
         ddire = vcat(dire0 , dire1)
         direse = sqrt(ddire'*V*ddire )
     #     direse = sqrt(Complex(ddire'*V*ddire ))
         inddire = vcat(indire0 , indire1)
         indirese = sqrt(inddire'*V*inddire )
         totalese = sqrt((ddire+inddire)'*V*(ddire+inddire))
 
     else
         dire1, indire1 = IrhoWW(gamma, rowIDT) 	
         dire = dire0 *b[1] + dire1*b[2]
         indire = indire0*b[1] + indire1*b[2]
         totale = dire + indire
         dire2, indire2 = IrhoWWIrhoW(gamma,dgamma, rowIDT) 	
         dire3, indire3 = IrhoWWIrhoWW(gamma,dgamma, rowIDT) 	
         ddire = vcat(dire0 , dire1 , (dire2*b[1]+dire3*b[2]))
         direse = sqrt(ddire'*V*ddire )
     #     direse = sqrt(Complex(ddire'*V*ddire ))
         inddire = vcat(indire0 , indire1 , (indire2*b[1]+indire3*b[2]))
         indirese =sqrt(inddire'*V*inddire) 
         totalese = sqrt((ddire+inddire)'*V*(ddire+inddire))
         
     end 
     dire   = hcat(dire, direse)
     indire = hcat(indire , indirese) 
     totale = hcat(totale,totalese )
     
     return dire,indire,totale
 end
         
       
 
 function get_mareffx(
     pos::NamedTuple, coef::Array{Float64, 1},  var_cov_matrix::Matrix,
     eigvalu::NamedTuple ,indices_list::Vector{Any}, rowIDT::Matrix{Any})
    
     gammap = coef[pos.beggamma];
     gamma  = eigvalu.rumin/(1+exp(gammap))+eigvalu.rumax*exp(gammap)/(1+exp(gammap));
     dgamma =  exp(gammap)*((eigvalu.rumax-eigvalu.rumin)/(1+exp(gammap))^2);
    

    totalemat = Array{Any}(undef,0,2)
    diremat   = Array{Any}(undef,0,2)
    indiremat = Array{Any}(undef,0,2)
    
    dire0, indire0= IrhoW(gamma,  rowIDT)
    
    for (symbol, indices) in indices_list
        b = coef[(pos.begx-1).+indices]
        V = var_cov_matrix[vcat((pos.begx-1).+indices,pos.beggamma),vcat((pos.begx-1).+indices,pos.beggamma)]
        dire,indire,totale = get_mareffx_single(b, dire0,indire0,  V,gamma,dgamma,rowIDT)
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
 