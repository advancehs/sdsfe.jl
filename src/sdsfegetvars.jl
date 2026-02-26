########################################################
####       process data for estimation              ####
########################################################



function get_rowIDT(ivar) 

  # Create a matrix of panel information.

  N  = length(unique(ivar)) # N=number of panels
  id = Array{Any}(undef,N,2)
  id[:,1] = unique(ivar)    # list of id with no repetition
  @inbounds for i = 1:N
      @views id[i,2]= sum(ivar.== id[i,1])    # number of periods for the panel
  end
  @views Tᵢ = id[:,2]

  rowID =  Vector{Vector}()  
  @inbounds for i=1:N
      @views cc = findall(x-> x == id[i,1], ivar) # row index of i'th firm
      push!(rowID, cc) # put the id info in the vector; faster than using UnitRange
  end    

  rowIDT = hcat(rowID, Tᵢ) 

  return rowIDT # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods
end

function get_rowIDofT(tvar) 

# Create a matrix of panel information.

N  = size(unique(tvar),1) # N=number of panels
id = Array{Any}(undef,N,2)
id[:,1] = unique(tvar)    # list of id with no repetition
@inbounds for i = 1:N
    @views id[i,2]= sum(tvar.== id[i,1])    # number of periods for the panel
end
@views Tᵢ = id[:,2]

rowID =  Vector{Vector}()  
@inbounds for i=1:N
    @views cc = findall(x-> x == id[i,1], tvar) # row index of i'th firm
    push!(rowID, cc) # put the id info in the vector; faster than using UnitRange
end    

rowIDT = hcat(rowID, Tᵢ) 

return rowIDT # (Nx2): col_1 is panel's row info; col_2 is panel's number of periods
end

#=  replaced by sf_demean()
function get_INVtrM(rowIDT) 

  # Create the within-transformation matrix for panels.
  # The matrix is panel-specific.

  N = size(rowIDT,1)
  INVtrM = Vector{Matrix{Float64}}()  # store the inverted matrix
  @inbounds for i=1:N
      @views trM = Matrix(I, rowIDT[i, 2], rowIDT[i, 2]) .- (1/rowIDT[i,2]) # transformation matrix
    # push!(INVtrM, pinv(trM)) # put the inverted matrix in the vector; note, trM is an idempotent matrix, and so the inverse is itself
      push!(INVtrM, trM) # put the inverted matrix in the vector; note, trM is an idempotent matrix, and so the inverse is itself
    end    
  return INVtrM # (Nx1): ith row is the transformation matrix for the ith panel
end
=#

function sf_demean(a::AbstractArray)  # subtract mean from columns
  return a .- mean(a, dims=1)
end

########################################################
####        process variables for estimation        ####
########################################################

#? ----- truncated normal -----------------

function find_all_indices_ordered(x_Wx)
indices_list = []
for symbol in unique(x_Wx)
    if symbol == :_consssssss
        continue
    end
    push!(indices_list, (symbol, findall(x -> x == symbol, x_Wx)))
end
return indices_list
end

#?--------- panel SSF Orea and Al, truncated normal ----------------

function getvar(::Type{SSFOAT}, dat::DataFrame)
ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])

tvar = dat[:, _dicM[:timevar]]

rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
if _dicM[:wx]!=Nothing  # yuvx
   Wxvar = dat[:, _dicM[:frontierWx]]   
end
qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]
zvar = dat[:, _dicM[:μ]]

Wy = _dicM[:wy]
Wx = _dicM[:wx]
Wu = _dicM[:wu]
Wv = _dicM[:wv]




#* --- model info printout --------- 
modelinfo1 = "spatial stochastic frontier analysis in Orea and Al (2019 JoE), normal and truncated-normal"
modelinfo2 = begin
 """
 * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

 $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,
 
 where vᵢₜ ∼ N(0, σᵥ²),
             σᵥ² = exp(log_σᵥ²) 
                 = exp($(_dicM[:σᵥ²]));
       uᵢₜ ∼ hscaleᵢₜ * uᵢ,
             hscaleᵢₜ = exp($(_dicM[:hscale])),
       uᵢ ∼ N⁺(μ, σᵤ²),
            μ = $(_dicM[:μ])
            σᵤ² = exp(log_σᵤ²) 
                = exp($(_dicM[:σᵤ²]));
 """
end
  
if  Wx!=Nothing   # yuvx
    wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
    T=length(unique(vec(Matrix(tvar))));
    for ttt in 1:T
        if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
       
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
        elseif length(wx) > 1
  
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
           
        end	
    end
end

#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx =  nofq = nofw = nofv = nofz = nofgamma = noftau = nofrho=0  # to make a complete list

nofobs  = nrow(dat)  
if  Wx!=Nothing   # yuvx
nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
  else
nofx    = size(xvar,2) 
  end

nofq    = size(qvar,2)  # h
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
nofz    = size(zvar,2)  # mu
if  Wy!=Nothing    
    nofgamma    = 1 # wy
end
if  Wu!=Nothing  
  noftau    = 1 # wu
end
if  Wv!=Nothing   
  nofrho    = 1 # wv
end

nofpara = nofx + nofq + nofw + nofv  +nofgamma+noftau+nofrho+nofz

nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
        nofgamma=nofgamma,noftau=noftau,nofrho=nofrho,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)

#* positions of the variables/parameters
begx=endx=begq=endq=begw=endw=begv=endv=begz=endz=beggamma=endgamma=begtau =endtau =begrho =endrho =0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begw = endq + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
begz = endv + 1
endz = begz + nofz-1
if  Wy!=Nothing    
    beggamma = endz + 1
    endgamma = beggamma + nofgamma-1
  else
    beggamma = endz 
    endgamma = beggamma + nofgamma
  end
if  Wu!=Nothing    
    begtau = endgamma + 1
    endtau = begtau + noftau-1
  else
    begtau = endgamma
    endtau = begtau + noftau
  end  
if  Wv!=Nothing    
    begrho = endtau + 1
    endrho = begrho + nofrho-1
else
    begrho = endtau
    endrho = begrho + nofrho
end

posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
          beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

#* create equation names and mark positions for making tables
if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                      lnσᵤ² = begw + 1,
                      lnσᵥ² = begv + 1,
                          μ = begz + 1,
                          ρ = beggamma + 1,
                          τ = begtau + 1,
                           γ = begrho + 1 )
    
    else # yu
          eqvec = (frontier = begx + 1, 
                          lnh = begq + 1,
                          lnσᵤ² = begw + 1,
                          lnσᵥ² = begv + 1,
                              μ = begz + 1,
                              ρ = beggamma + 1,
                              τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            μ = begz + 1,
                            ρ = beggamma + 1,
                            γ = begrho + 1 )
    else #y
          eqvec = (frontier = begx + 1, 
          lnh = begq + 1,
          lnσᵤ² = begw + 1,
          lnσᵥ² = begv + 1,
              μ = begz + 1,
              ρ = beggamma + 1 )
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            μ = begz + 1,
                            τ = begtau + 1,
                            γ = begrho + 1 )
    
    else # u
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            μ = begz + 1,
                            τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #v
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            μ = begz + 1,
                            γ = begrho + 1 )
    
    else # 
          eqvec = (frontier = begx + 1, 
          lnh = begq + 1,
          lnσᵤ² = begw + 1,
          lnσᵥ² = begv + 1,
              μ = begz + 1)
    end
end
end 

#* create equation names and mark positions 

if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # yu
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau),  )        
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_ρ = (begrho:endrho) )        
    else #y
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma), )        
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # u
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_τ = (begtau:endtau) )        
    
    end    
else 
    if Wv!=Nothing #v
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_ρ = (begrho:endrho) )        
    
    else # 
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),      
                       coeff_μ = (begz:endz))
  
    end
end
end           
          
           

#* retrieve variable names for making tables
if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
  else
xnames  = names(xvar)
  end
  

qnames  = names(qvar)
wnames  = names(wvar)
vnames  = names(vvar)
znames  = names(zvar)
if  Wy!=Nothing    
  gammanames  = "ρ"
end
if  Wu!=Nothing    
  taunames  = "τ"
end
if  Wv!=Nothing    
  rhonames  = "γ"
end


  if Wy!=Nothing  # yuv
      if Wu!=Nothing 
          if Wv!=Nothing #yuv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames,taunames,rhonames)
          else # yu
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames,taunames)
          end    
      else 
          if Wv!=Nothing #yv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames,rhonames)
          else #y
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames)
          end
      end
  else
      if Wu!=Nothing 
          if Wv!=Nothing #uv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, taunames,rhonames)
          else # u
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, taunames)
          end    
      else 
          if Wv!=Nothing #v
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, rhonames)
          else # 
                varlist = vcat(" ", xnames, qnames, wnames, vnames, znames)
          end
      end
  end 


#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
if  Wx!=Nothing   # yuvx
xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
  else
    xvar  = convert(Array{Float64}, Matrix(xvar))
  end
qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))
zvar  = convert(Array{Float64}, Matrix(zvar))
tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))
envar = ()
ivvar = ()


#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))


# 获得空间矩阵的特征值
rymin=rymax=rumin=rumax=rvmin=rvmax=0.0
if  Wy!=Nothing   # yuvx
    dylams = eigen(Wy[1])
    rymin = 1.0 / minimum(real(dylams.values))
    rymax = 1.0
    if length(Wy) > 1
        for k = 2:length(Wy)
            dylams = eigen(Wy[k])
            if rymin < 1.0 / minimum(real(dylams.values))
                rymin = 1.0 / minimum(real(dylams.values))
            end
        end
    end
  end
if  Wu!=Nothing   # yuvx
    dulams = eigen(Wu[1])
    rumin = 1.0 / minimum(real(dulams.values))
    rumax = 1.0
    if length(Wu) > 1
        for k = 2:length(Wu)
            dulams = eigen(Wu[k])
            if rumin < 1.0 / minimum(real(dulams.values))
                rumin = 1.0 / minimum(real(dulams.values))
            end
        end
    end
  end
  
if  Wv!=Nothing   # yuvx
    dvlams = eigen(Wv[1])
    rvmin = 1.0 / minimum(real(dvlams.values))
    rvmax = 1.0
    if length(Wv) > 1
        for k = 2:length(Wv)
            dvlams = eigen(Wv[k])
            if rvmin < 1.0 / minimum(real(dvlams.values))
                rvmin = 1.0 / minimum(real(dvlams.values))
            end
        end
    end
end

eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)
indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))


return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar,  qvar, wvar, vvar,  zvar, 
                              envar, ivvar, eigvalu, indices_list, indices_listz,rowIDT, varlist


end






function getvar(::Type{SSFOADT}, dat::DataFrame)
ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])

tvar = dat[:, _dicM[:timevar]]

rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
if _dicM[:wx]!=Nothing  # yuvx
   Wxvar = dat[:, _dicM[:frontierWx]]   
end
qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]
zvar = dat[:, _dicM[:μ]]

Wy = _dicM[:wy]
Wx = _dicM[:wx]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

envar = dat[:, _dicM[:envar]]   
  name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
  name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
  name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv

ivvar = dat[:, name_new_ivvar]   


#* --- model info printout --------- 
modelinfo1 = "spatial stochastic frontier analysis in Orea and Al (2019 JoE), normal and truncated-normal"
modelinfo2 = begin
 """
 * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

 $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,
 
 where vᵢₜ ∼ N(0, σᵥ²),
             σᵥ² = exp(log_σᵥ²) 
                 = exp($(_dicM[:σᵥ²]));
       uᵢₜ ∼ hscaleᵢₜ * uᵢ,
             hscaleᵢₜ = exp($(_dicM[:hscale])),
       uᵢ ∼ N⁺(μ, σᵤ²),
            μ = $(_dicM[:μ])
            σᵤ² = exp(log_σᵤ²) 
                = exp($(_dicM[:σᵤ²]));
 """
end
  
if  Wx!=Nothing   # yuvx
    wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
    T=length(unique(vec(Matrix(tvar))));
    for ttt in 1:T
        if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
       
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
        elseif length(wx) > 1
  
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
           
        end	
    end
end

#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx = nofq = nofw = nofv = nofz= nofgamma = noftau = nofrho = nofphi = nofeta = 0  # to make a complete list

nofobs  = nrow(dat)  
if  Wx!=Nothing   # yuvx
nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
  else
nofx    = size(xvar,2) 
  end
nofq    = size(qvar,2)  # h  
nofphi  = size(envar,2)*size(ivvar,2)
nofeta  = size(envar,2)
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
nofz    = size(zvar,2)  # mu 
if  Wy!=Nothing    
    nofgamma    = 1 # wy
end
if  Wu!=Nothing  
  noftau    = 1 # wu
end
if  Wv!=Nothing   
  nofrho    = 1 # wv
end

nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv +nofz +nofgamma+noftau+nofrho

nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
        nofgamma=nofgamma,noftau=noftau,nofrho=nofrho,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)

#* positions of the variables/parameters
begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=begtau =endtau =begrho =endrho =0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begphi = endq+1
endphi = begphi + nofphi-1
begeta = endphi+1
endeta = begeta + nofeta-1

begw = endeta + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
begz = endv + 1
endz = begz + nofz-1
if  Wy!=Nothing    
    beggamma = endz + 1
    endgamma = beggamma + nofgamma-1
  else
    beggamma = endz 
    endgamma = beggamma + nofgamma
  end
if  Wu!=Nothing    
    begtau = endgamma + 1
    endtau = begtau + noftau-1
  else
    begtau = endgamma
    endtau = begtau + noftau
  end  
if  Wv!=Nothing    
    begrho = endtau + 1
    endrho = begrho + nofrho-1
else
    begrho = endtau
    endrho = begrho + nofrho
end

posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
          begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
          beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

#* create equation names and mark positions for making tables
if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                ρ = beggamma + 1,
                                τ = begtau + 1,
                                γ = begrho + 1 )
    
    else # yu
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                ρ = beggamma + 1,
                                τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #yv
      eqvec = (frontier = begx + 1, 

                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                ρ = beggamma + 1,
                                γ = begrho + 1 )
    else #y
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                ρ = beggamma + 1 )
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                τ = begtau + 1,
                                γ = begrho + 1 )
    
    else # u
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1,
                                τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #v
          eqvec = (frontier = begx + 1, 
                          lnh = begq + 1,
                          ϕ        = begphi + 1,
                          η        =  begeta + 1,
                              lnσᵤ² = begw + 1,
                              lnσᵥ² = begv + 1,
                                  μ = begz + 1,
                                  γ = begrho + 1 )
    
    else # 
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        ϕ        = begphi + 1,
                        η        =  begeta + 1,
                            lnσᵤ² = begw + 1,
                            lnσᵥ² = begv + 1,
                                μ = begz + 1)
    end
end
end 

#* create equation names and mark positions 

if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # yu
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau),  )        
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma),
                           coeff_ρ = (begrho:endrho) )        
    else #y
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_γ = (beggamma:endgamma), )        
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # u
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_τ = (begtau:endtau) )        
    
    end    
else 
    if Wv!=Nothing #v
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                       coeff_μ = (begz:endz),
                           coeff_ρ = (begrho:endrho) )        
    
    else # 
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),      
                       coeff_μ = (begz:endz))
  
    end
end
end           



#* retrieve variable names for making tables
if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
    else
  xnames  = names(xvar)
    end
    

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["η_" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  znames  = names(zvar)
  if  Wy!=Nothing    
      gammanames  = "ρ"
    end
  if  Wu!=Nothing    
      taunames  = "τ"
    end
  if  Wv!=Nothing    
      rhonames  = "γ"
    end

  if Wy!=Nothing  # yuv
      if Wu!=Nothing 
          if Wv!=Nothing #yuv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames,taunames,rhonames)
          else # yu
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames,taunames)
          end    
      else 
          if Wv!=Nothing #yv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames,rhonames)
          else #y
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames)
          end
      end
  else
      if Wu!=Nothing 
          if Wv!=Nothing #uv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, taunames,rhonames)
          else # u
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, taunames)
          end    
      else 
          if Wv!=Nothing #v
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, rhonames)
          else # 
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames)
          end
      end
  end 
  

#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
if  Wx!=Nothing   # yuvx
xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
  else
    xvar  = convert(Array{Float64}, Matrix(xvar))
  end
qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))
zvar  = convert(Array{Float64}, Matrix(zvar))
tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))

ivvar  = convert(Array{Float64}, Matrix(ivvar))
envar  = convert(Array{Float64}, Matrix(envar))


#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))


# 获得空间矩阵的特征值
rymin=rymax=rumin=rumax=rvmin=rvmax=0.0
if  Wy!=Nothing   # yuvx
    dylams = eigen(Wy[1])
    rymin = 1.0 / minimum(real(dylams.values))
    rymax = 1.0
    if length(Wy) > 1
        for k = 2:length(Wy)
            dylams = eigen(Wy[k])
            if rymin < 1.0 / minimum(real(dylams.values))
                rymin = 1.0 / minimum(real(dylams.values))
            end
        end
    end
  end
if  Wu!=Nothing   # yuvx
    dulams = eigen(Wu[1])
    rumin = 1.0 / minimum(real(dulams.values))
    rumax = 1.0
    if length(Wu) > 1
        for k = 2:length(Wu)
            dulams = eigen(Wu[k])
            if rumin < 1.0 / minimum(real(dulams.values))
                rumin = 1.0 / minimum(real(dulams.values))
            end
        end
    end
  end
  
if  Wv!=Nothing   # yuvx
    dvlams = eigen(Wv[1])
    rvmin = 1.0 / minimum(real(dvlams.values))
    rvmax = 1.0
    if length(Wv) > 1
        for k = 2:length(Wv)
            dvlams = eigen(Wv[k])
            if rvmin < 1.0 / minimum(real(dvlams.values))
                rvmin = 1.0 / minimum(real(dvlams.values))
            end
        end
    end
end
  

eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar,  qvar, wvar, vvar,  zvar, 
                  envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end





function getvar(::Type{SSFOAH}, dat::DataFrame)
ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])

tvar = dat[:, _dicM[:timevar]]

rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
if _dicM[:wx]!=Nothing  # yuvx
   Wxvar = dat[:, _dicM[:frontierWx]]   
end
qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]

Wy = _dicM[:wy]
Wx = _dicM[:wx]
Wu = _dicM[:wu]
Wv = _dicM[:wv]




#* --- model info printout ---------
modelinfo1 = "spatial stochastic frontier analysis in Orea and Al (2019 JoE), normal and half-normal"
modelinfo2 = begin
 """
 * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

 $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

 where vᵢₜ ∼ N(0, σᵥ²),
             σᵥ² = exp(log_σᵥ²)
                 = exp($(_dicM[:σᵥ²]));
       uᵢₜ ∼ hscaleᵢₜ * uᵢ,
             hscaleᵢₜ = exp($(_dicM[:hscale])),
       uᵢ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)
                = exp($(_dicM[:σᵤ²]));
 """
end

if  Wx!=Nothing   # yuvx
    wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
    T=length(unique(vec(Matrix(tvar))));
    for ttt in 1:T
        if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
        elseif length(wx) > 1

            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[ttt] * xx[rowIDT[ttt,1], :]
           
        end	
    end
end

#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx =  nofq = nofw = nofv = nofz = nofgamma = noftau = nofrho=0  # to make a complete list

nofobs  = nrow(dat)  
if  Wx!=Nothing   # yuvx
nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
  else
nofx    = size(xvar,2) 
  end

nofq    = size(qvar,2)  # h
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
if  Wy!=Nothing    
    nofgamma    = 1 # wy
end
if  Wu!=Nothing  
  noftau    = 1 # wu
end
if  Wv!=Nothing   
  nofrho    = 1 # wv
end
nofpara = nofx + nofq + nofw + nofv + nofz + nofgamma + noftau + nofrho

nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv,  nofz=nofz,
        nofgamma=nofgamma,noftau=noftau,nofrho=nofrho,  nofpara=nofpara, nofmarg = nofq+nofw)

#* positions of the variables/parameters
begx=endx=begq=endq=begw=endw=begv=endv=begz=endz=beggamma=endgamma=begtau =endtau =begrho =endrho =0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begw = endq + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
if  Wy!=Nothing    
    beggamma = endv + 1
    endgamma = beggamma + nofgamma-1
  else
    beggamma = endv 
    endgamma = beggamma + nofgamma
  end
if  Wu!=Nothing    
    begtau = endgamma + 1
    endtau = begtau + noftau-1
  else
    begtau = endgamma
    endtau = begtau + noftau
  end  
if  Wv!=Nothing    
    begrho = endtau + 1
    endrho = begrho + nofrho-1
else
    begrho = endtau
    endrho = begrho + nofrho
end
posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz,
          beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

#* create equation names and mark positions for making tables
if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                      lnσᵤ² = begw + 1,
                      lnσᵥ² = begv + 1,
                          ρ = beggamma + 1,
                          τ = begtau + 1,
                          γ = begrho + 1 )
    
    else # yu
          eqvec = (frontier = begx + 1, 
                          lnh = begq + 1,
                          lnσᵤ² = begw + 1,
                          lnσᵥ² = begv + 1,
                              ρ = beggamma + 1,
                              τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            ρ = beggamma + 1,
                            γ = begrho + 1 )
    else #y
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            ρ = beggamma + 1 )
                  
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            τ = begtau + 1,
                            γ = begrho + 1 )
    
    else # u
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            τ = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #v
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1,
                            γ = begrho + 1 )
    
    else # 
          eqvec = (frontier = begx + 1, 
                        lnh = begq + 1,
                        lnσᵤ² = begw + 1,
                        lnσᵥ² = begv + 1 )
     
    end
end
end 

#* create equation names and mark positions 

if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # yu
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau),  )        
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_ρ = (begrho:endrho) )        
    else #y
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma), )        
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # u
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_τ = (begtau:endtau) )        
    
    end    
else 
    if Wv!=Nothing #v
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_ρ = (begrho:endrho) )        
    
    else # 
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv))        
     
    end
end
end           

#* retrieve variable names for making tables
if  Wx!=Nothing   # yuvx
xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
else
xnames  = names(xvar)
end


qnames  = names(qvar)
wnames  = names(wvar)
vnames  = names(vvar)
if  Wy!=Nothing    
gammanames  = "ρ"
end
if  Wu!=Nothing    
taunames  = "τ"
end
if  Wv!=Nothing    
rhonames  = "γ"
end

  if Wy!=Nothing  # yuv
      if Wu!=Nothing 
          if Wv!=Nothing #yuv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames,taunames,rhonames)
          else # yu
                varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames,taunames)
          end    
      else 
          if Wv!=Nothing #yv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames,rhonames)
          else #y
                varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames)
          end
      end
  else
      if Wu!=Nothing 
          if Wv!=Nothing #uv
                varlist = vcat(" ", xnames, qnames, wnames, vnames, taunames,rhonames)
          else # u
                varlist = vcat(" ", xnames, qnames, wnames, vnames,taunames)
          end    
      else 
          if Wv!=Nothing #v
                varlist = vcat(" ", xnames, qnames, wnames, vnames, rhonames)
          else # 
                varlist = vcat(" ", xnames, qnames, wnames, vnames)
          end
      end
  end 
  



#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
if  Wx!=Nothing   # yuvx
xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
  else
    xvar  = convert(Array{Float64}, Matrix(xvar))
  end
qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))

tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))
zvar = ()
envar = ()
ivvar = ()


#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))


# 获得空间矩阵的特征值
rymin=rymax=rumin=rumax=rvmin=rvmax=0.0
if  Wy!=Nothing   # yuvx
    dylams = eigen(Wy[1])
    rymin = 1.0 / minimum(real(dylams.values))
    rymax = 1.0
    if length(Wy) > 1
        for k = 2:length(Wy)
            dylams = eigen(Wy[k])
            if rymin < 1.0 / minimum(real(dylams.values))
                rymin = 1.0 / minimum(real(dylams.values))
            end
        end
    end
  end
if  Wu!=Nothing   # yuvx
    dulams = eigen(Wu[1])
    rumin = 1.0 / minimum(real(dulams.values))
    rumax = 1.0
    if length(Wu) > 1
        for k = 2:length(Wu)
            dulams = eigen(Wu[k])
            if rumin < 1.0 / minimum(real(dulams.values))
                rumin = 1.0 / minimum(real(dulams.values))
            end
        end
    end
  end
  
if  Wv!=Nothing   # yuvx
    dvlams = eigen(Wv[1])
    rvmin = 1.0 / minimum(real(dvlams.values))
    rvmax = 1.0
    if length(Wv) > 1
        for k = 2:length(Wv)
            dvlams = eigen(Wv[k])
            if rvmin < 1.0 / minimum(real(dvlams.values))
                rvmin = 1.0 / minimum(real(dvlams.values))
            end
        end
    end
end
  
eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

  

return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar,  qvar, wvar, vvar,  zvar, 
                  envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist


end


function getvar(::Type{SSFOADH}, dat::DataFrame)

  
ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])
tvar = dat[:, _dicM[:timevar]]
rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
if _dicM[:wx]!=Nothing  # yuvx
   Wxvar = dat[:, _dicM[:frontierWx]]   
end
qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]
# zvar = dat[:, _dicM[:μ]]

Wy = _dicM[:wy]
Wx = _dicM[:wx]
Wu = _dicM[:wu]
Wv = _dicM[:wv]

envar = dat[:, _dicM[:envar]]
  name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
  name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
  name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv

ivvar = dat[:, name_new_ivvar]


#* --- model info printout ---------
modelinfo1 = "spatial stochastic frontier analysis in Orea and Al (2019 JoE), normal and half-normal"
modelinfo2 = begin
 """
 * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

 $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

 where vᵢₜ ∼ N(0, σᵥ²),
             σᵥ² = exp(log_σᵥ²)
                 = exp($(_dicM[:σᵥ²]));
       uᵢₜ ∼ hscaleᵢₜ * uᵢ,
             hscaleᵢₜ = exp($(_dicM[:hscale])),
       uᵢ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)
                = exp($(_dicM[:σᵤ²]));
 """
end
  
if  Wx!=Nothing   # yuvx
    wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
    T=length(unique(vec(Matrix(tvar))));
    for ttt in 1:T
        if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
       
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
        elseif length(wx) > 1
  
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
           
        end	
    end
end

#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx = nofq = nofw = nofv = nofz= nofgamma = noftau = nofrho = nofphi = nofeta = 0  # to make a complete list

nofobs  = nrow(dat)  
if  Wx!=Nothing   # yuvx
nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
  else
nofx    = size(xvar,2) 
  end
nofq    = size(qvar,2)  # h  
nofphi  = size(envar,2)*size(ivvar,2)
nofeta  = size(envar,2)
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
# nofz    = size(zvar,2)  # mu
if  Wy!=Nothing    
    nofgamma    = 1 # wy
end
if  Wu!=Nothing  
  noftau    = 1 # wu
end
if  Wv!=Nothing   
  nofrho    = 1 # wv
end

nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv  +nofgamma+noftau+nofrho+nofz


nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
        nofgamma=nofgamma,noftau=noftau,nofrho=nofrho,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)

#* positions of the variables/parameters
begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=begtau =endtau =begrho =endrho =0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begphi = endq+1
endphi = begphi + nofphi-1
begeta = endphi+1
endeta = begeta + nofeta-1

begw = endeta + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
# begz = endv + 1
# endz = begz + nofz-1
if  Wy!=Nothing    
    beggamma = endv + 1
    endgamma = beggamma + nofgamma-1
  else
    beggamma = endv 
    endgamma = beggamma + nofgamma
  end
if  Wu!=Nothing    
    begtau = endgamma + 1
    endtau = begtau + noftau-1
  else
    begtau = endgamma
    endtau = begtau + noftau
  end  
if  Wv!=Nothing    
    begrho = endtau + 1
    endrho = begrho + nofrho-1
else
    begrho = endtau
    endrho = begrho + nofrho
end

posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
          begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
          beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

#* create equation names and mark positions for making tables
if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec = (frontier = begx + 1, 
                       lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        ρ   = beggamma + 1,
                        τ   = begtau + 1,
                        γ   = begrho + 1 )
    
    else # yu
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        ρ   = beggamma + 1,
                        τ   = begtau + 1 )
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        ρ   = beggamma + 1,
                        γ   = begrho + 1 )
    else #y
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        ρ   = beggamma + 1)
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        τ   = begtau + 1,
                        γ   = begrho + 1 )
                  
    else # u
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        τ   = begtau + 1)
    
    end    
else 
    if Wv!=Nothing #v
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1,
                        γ   = begrho + 1 )
    
    else # 
          eqvec = (frontier = begx + 1, 
                        lnh  = begq + 1,
                        ϕ   = begphi + 1,
                        η   =  begeta + 1,
                    lnσᵤ²   = begw + 1,
                    lnσᵥ²   = begv + 1)
     
    end
end
end 

#* create equation names and mark positions 

if Wy!=Nothing  # yuv
if Wu!=Nothing 
    if Wv!=Nothing #yuv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # yu
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_τ = (begtau:endtau),  )        
    
    end    
else 
    if Wv!=Nothing #yv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma),
                           coeff_ρ = (begrho:endrho) )        
    else #y
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_γ = (beggamma:endgamma), )        
    
    end
end
else
if Wu!=Nothing 
    if Wv!=Nothing #uv
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_τ = (begtau:endtau), 
                           coeff_ρ = (begrho:endrho) )        
    
    else # u
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_τ = (begtau:endtau) )        
    
    end    
else 
    if Wv!=Nothing #v
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv),
                           coeff_ρ = (begrho:endrho) )        
    
    else # 
          eqvec2 = (coeff_frontier = (begx:endx), 
                  coeff_log_hscale = (begq:endq),
                        coeff_ϕ       = (begphi:endphi),
                        coeff_η       = (begeta:endeta),
                     coeff_log_σᵤ² = (begw:endw),
                     coeff_log_σᵥ² = (begv:endv))        
     
    end
end
end           

#* retrieve variable names for making tables
if  Wx!=Nothing   # yuvx
xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
else
xnames  = names(xvar)
end


qnames  = names(qvar)
ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
etanames = ["η_" * s for s in names(envar)] 
wnames  = names(wvar)
vnames  = names(vvar)
# znames  = names(zvar)
if  Wy!=Nothing    
gammanames  = "ρ"
end
if  Wu!=Nothing    
taunames  = "τ"
end
if  Wv!=Nothing    
rhonames  = "γ"
end
  if Wy!=Nothing  # yuv
      if Wu!=Nothing 
          if Wv!=Nothing #yuv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames,taunames,rhonames)
          else # yu
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames,taunames)
          end    
      else 
          if Wv!=Nothing #yv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames,rhonames)
          else #y
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames)
          end
      end
  else
      if Wu!=Nothing 
          if Wv!=Nothing #uv
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, taunames,rhonames)
          else # u
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames,taunames)
          end    
      else 
          if Wv!=Nothing #v
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, rhonames)
          else # 
                varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames)
          end
      end
  end 



#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
if  Wx!=Nothing   # yuvx
xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
  else
    xvar  = convert(Array{Float64}, Matrix(xvar))
  end
qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))
# zvar  = convert(Array{Float64}, Matrix(zvar))
tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))

ivvar  = convert(Array{Float64}, Matrix(ivvar))
envar  = convert(Array{Float64}, Matrix(envar))
zvar = ()

#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))


# 获得空间矩阵的特征值
rymin=rymax=rumin=rumax=rvmin=rvmax=0.0
if  Wy!=Nothing   # yuvx
    dylams = eigen(Wy[1])
    rymin = 1.0 / minimum(real(dylams.values))
    rymax = 1.0
    if length(Wy) > 1
        for k = 2:length(Wy)
            dylams = eigen(Wy[k])
            if rymin < 1.0 / minimum(real(dylams.values))
                rymin = 1.0 / minimum(real(dylams.values))
            end
        end
    end
  end
if  Wu!=Nothing   # yuvx
    dulams = eigen(Wu[1])
    rumin = 1.0 / minimum(real(dulams.values))
    rumax = 1.0
    if length(Wu) > 1
        for k = 2:length(Wu)
            dulams = eigen(Wu[k])
            if rumin < 1.0 / minimum(real(dulams.values))
                rumin = 1.0 / minimum(real(dulams.values))
            end
        end
    end
  end
  
if  Wv!=Nothing   # yuvx
    dvlams = eigen(Wv[1])
    rvmin = 1.0 / minimum(real(dvlams.values))
    rvmax = 1.0
    if length(Wv) > 1
        for k = 2:length(Wv)
            dvlams = eigen(Wv[k])
            if rvmin < 1.0 / minimum(real(dvlams.values))
                rvmin = 1.0 / minimum(real(dvlams.values))
            end
        end
    end
end
  
eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)
  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
  qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
  
end




function getvar(::Type{SSFKUEH}, dat::DataFrame)


    ivar = dat[:, _dicM[:idvar]] 
    dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])
    tvar = dat[:, _dicM[:timevar]]
    rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year
    
    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
    if _dicM[:wx]!=Nothing  # yuvx
       Wxvar = dat[:, _dicM[:frontierWx]]   
    end
    qvar = dat[:, _dicM[:hscale]]  
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    # zvar = dat[:, _dicM[:μ]]
    
    Wy = _dicM[:wy]
    Wx = _dicM[:wx]
    
    envar = dat[:, _dicM[:envar]]   
      name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
      name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
      name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
    
    ivvar = dat[:, name_new_ivvar]   
    
    
    #* --- model info printout ---------
    modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2020 EJoR), normal and half-normal"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     """
    end

    if  Wx!=Nothing   # yuvx
        wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
        T=length(unique(vec(Matrix(tvar))));
        for ttt in 1:T
            if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度

                xx = Matrix(dat[!, _dicM[:frontierWx]])
                @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]


            elseif length(wx) > 1
      
                xx = Matrix(dat[!, _dicM[:frontierWx]])
                @views wxvar[rowIDT[ttt,1], :] .= Wx[ttt] * xx[rowIDT[ttt,1], :]
               
            end	
        end
    end


    #* --- retrieve and generate important parameters -----
    
    #*   number of obs and number of variables
    nofx = nofq = nofw = nofv = nofz= nofgamma = nofphi = nofeta = 0  # to make a complete list
    
    nofobs  = nrow(dat)  
    if  Wx!=Nothing   # yuvx
    nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
      else
    nofx    = size(xvar,2) 
      end
    nofq    = size(qvar,2)  # h  
    nofphi  = size(envar,2)*size(ivvar,2)
    nofeta  = size(envar,2)
    nofw    = size(wvar,2)  # sigma_u_2
    nofv    = size(vvar,2)  # sigma_v_2
    # nofz    = size(zvar,2)  # mu
    nofgamma    = 1 # wy
  
    
    nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv  +nofgamma+nofz
    
    
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
            nofgamma=nofgamma,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)
    
    #* positions of the variables/parameters
    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma =0
    
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq-1
    begphi = endq+1
    endphi = begphi + nofphi-1
    begeta = endphi+1
    endeta = begeta + nofeta-1
    
    begw = endeta + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = begv + nofv-1
    # begz = endv + 1
    # endz = begz + nofz-1
    beggamma = endv + 1
    endgamma = beggamma + nofgamma-1
      
    
    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
              begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
              beggamma=beggamma, endgamma=endgamma)
    
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
    lnh  = begq + 1,
    ϕ   = begphi + 1,
    η   =  begeta + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1,
    ρ   = beggamma + 1)

   

    
    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
    coeff_log_hscale = (begq:endq),
          coeff_ϕ       = (begphi:endphi),
          coeff_η       = (begeta:endeta),
       coeff_log_σᵤ² = (begw:endw),
       coeff_log_σᵥ² = (begv:endv),
             coeff_γ = (beggamma:endgamma), )        

  
    
    #* retrieve variable names for making tables
    if  Wx!=Nothing   # yuvx
    xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
    else
    xnames  = names(xvar)
    end
    
    
    qnames  = names(qvar)
    ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
    etanames = ["η_" * s for s in names(envar)] 
    wnames  = names(wvar)
    vnames  = names(vvar)
    # znames  = names(zvar)
    if  Wy!=Nothing    
    gammanames  = "ρ"
    end

    varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames)

    
    
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    if  Wx!=Nothing   # yuvx
    xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
      else
        xvar  = convert(Array{Float64}, Matrix(xvar))
      end
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    # zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    
    ivvar  = convert(Array{Float64}, Matrix(ivvar))
    envar  = convert(Array{Float64}, Matrix(envar))
    zvar = ()
    
    #* various functions can and cannot contain a constant, check! ---- *#
    # checkConst(xvar, :frontier, @requireConst(0))
    # checkConst(qvar, :hscale,   @requireConst(0)) 
    # checkConst(wvar, :σᵤ²,      @requireConst(1))
    # checkConst(vvar, :σᵥ²,      @requireConst(1))
    # checkConst(zvar, :μ,        @requireConst(1))
    
    
    # 获得空间矩阵的特征值
    rymin=rymax=0
    if  Wy!=Nothing   # yuvx
        dylams = eigen(Wy[1])
        rymin = 1 / minimum(real(dylams.values))
        rymax = 1
        if length(Wy) > 1
            for k = 2:length(Wy)
                dylams = eigen(Wy[k])
                if rymin < 1 / minimum(real(dylams.values))
                    rymin = 1 / minimum(real(dylams.values))
                end
            end
        end
      end
 
      
    eigvalu = (rymin=rymin, rymax=rymax)
      indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
      indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
      qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
      
    end
    

function getvar(::Type{SSFKUH}, dat::DataFrame)

    
      ivar = dat[:, _dicM[:idvar]] 
      dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])
      tvar = dat[:, _dicM[:timevar]]
      rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year
      
      yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
      xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
      if _dicM[:wx]!=Nothing  # yuvx
          Wxvar = dat[:, _dicM[:frontierWx]]   
      end
      qvar = dat[:, _dicM[:hscale]]  
      wvar = dat[:, _dicM[:σᵤ²]]
      vvar = dat[:, _dicM[:σᵥ²]]
      # zvar = dat[:, _dicM[:μ]]
      
      Wy = _dicM[:wy]
      Wx = _dicM[:wx]

      #* --- model info printout ---------
      modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2020 EJoR), normal and half-normal"
      modelinfo2 = begin
        """
        * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

        $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

        where vᵢₜ ∼ N(0, σᵥ²),
                    σᵥ² = exp(log_σᵥ²)
                        = exp($(_dicM[:σᵥ²]));
              uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                    hscaleᵢₜ = exp($(_dicM[:hscale])),
              uᵢ ∼ N⁺(0, σᵤ²),
                  σᵤ² = exp(log_σᵤ²)
                      = exp($(_dicM[:σᵤ²]));
        """
      end
        
      if  Wx!=Nothing   # yuvx
          wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
          T=length(unique(vec(Matrix(tvar))));
          for ttt in 1:T
              if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
              
                  xx = Matrix(dat[!, _dicM[:frontierWx]])
                  @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
              elseif length(wx) > 1
        
                  xx = Matrix(dat[!, _dicM[:frontierWx]])
                  @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
                  
              end	
          end
      end
      
      #* --- retrieve and generate important parameters -----
      
      #*   number of obs and number of variables
      nofx = nofq = nofw = nofv = nofz= nofgamma  = 0  # to make a complete list
      
      nofobs  = nrow(dat)  
      if  Wx!=Nothing   # yuvx
      nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
        else
      nofx    = size(xvar,2) 
        end
      nofq    = size(qvar,2)  # h  
      nofw    = size(wvar,2)  # sigma_u_2
      nofv    = size(vvar,2)  # sigma_v_2
      # nofz    = size(zvar,2)  # mu
      nofgamma    = 1 # wy
    
      
      nofpara = nofx + nofq + nofw + nofv  +nofgamma+nofz
      
      
      nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofgamma=nofgamma,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)
      
      #* positions of the variables/parameters
      begx=endx=begq=endq=begw=endw=begv=endv=begz=endz=beggamma=endgamma =0
      
      begx = 1
      endx = nofx
      begq = endx + 1
      endq = begq + nofq-1
      begw = endq + 1
      endw = begw + nofw-1
      begv = endw + 1
      endv = begv + nofv-1
      # begz = endv + 1
      # endz = begz + nofz-1
      beggamma = endv + 1
      endgamma = beggamma + nofgamma-1
        
      
      posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
                begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
                beggamma=beggamma, endgamma=endgamma)
      
      #* create equation names and mark positions for making tables
      eqvec = (frontier = begx + 1, 
      lnh  = begq + 1,
  lnσᵤ²   = begw + 1,
  lnσᵥ²   = begv + 1,
      ρ   = beggamma + 1)
  
      
  
      
      #* create equation names and mark positions 
      eqvec2 = (coeff_frontier = (begx:endx), 
      coeff_log_hscale = (begq:endq),
          coeff_log_σᵤ² = (begw:endw),
          coeff_log_σᵥ² = (begv:endv),
                coeff_γ = (beggamma:endgamma), )        
  
    
      
      #* retrieve variable names for making tables
      if  Wx!=Nothing   # yuvx
      xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
      else
      xnames  = names(xvar)
      end
      
      
      qnames  = names(qvar)
      wnames  = names(wvar)
      vnames  = names(vvar)
      # znames  = names(zvar)
      if  Wy!=Nothing    
      gammanames  = "ρ"
      end
  
      varlist = vcat(" ", xnames, qnames , wnames, vnames, gammanames)
  
      
      
      #* Converting the dataframe to matrix in order to do computation
      yvar  = convert(Array{Float64}, Matrix(yvar))
      if  Wx!=Nothing   # yuvx
      xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
        else
          xvar  = convert(Array{Float64}, Matrix(xvar))
        end
      qvar  = convert(Array{Float64}, Matrix(qvar))
      wvar  = convert(Array{Float64}, Matrix(wvar))
      vvar  = convert(Array{Float64}, Matrix(vvar))
      # zvar  = convert(Array{Float64}, Matrix(zvar))
      tvar  = convert(Array{Float64}, Matrix(tvar))
      ivar  = convert(Array{Float64}, Matrix(ivar))
      
      ivvar  = ()
      envar  = ()
      zvar = ()
      
      #* various functions can and cannot contain a constant, check! ---- *#
      # checkConst(xvar, :frontier, @requireConst(0))
      # checkConst(qvar, :hscale,   @requireConst(0)) 
      # checkConst(wvar, :σᵤ²,      @requireConst(1))
      # checkConst(vvar, :σᵥ²,      @requireConst(1))
      # checkConst(zvar, :μ,        @requireConst(1))
      
      
      # 获得空间矩阵的特征值
      rymin=rymax=0
      if  Wy!=Nothing   # yuvx
          dylams = eigen(Wy[1])
          rymin = 1 / minimum(real(dylams.values))
          rymax = 1
          if length(Wy) > 1
              for k = 2:length(Wy)
                  dylams = eigen(Wy[k])
                  if rymin < 1 / minimum(real(dylams.values))
                      rymin = 1 / minimum(real(dylams.values))
                  end
              end
          end
        end
    
        
      eigvalu = (rymin=rymin, rymax=rymax)
        indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
        indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

      return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
        qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
        
end
        
        
    



function getvar(::Type{SSFKUET}, dat::DataFrame)


      ivar = dat[:, _dicM[:idvar]] 
      dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])
      tvar = dat[:, _dicM[:timevar]]
      rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year
      
      yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
      xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
      if _dicM[:wx]!=Nothing  # yuvx
         Wxvar = dat[:, _dicM[:frontierWx]]   
      end
      qvar = dat[:, _dicM[:hscale]]  
      wvar = dat[:, _dicM[:σᵤ²]]
      vvar = dat[:, _dicM[:σᵥ²]]
      zvar = dat[:, _dicM[:μ]]
      
      Wy = _dicM[:wy]
      Wx = _dicM[:wx]
      
      envar = dat[:, _dicM[:envar]]   
        name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
        name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
        name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
      
      ivvar = dat[:, name_new_ivvar]   
      
      
      #* --- model info printout --------- 
      modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2020 EJoR), normal and truncated-normal"
      modelinfo2 = begin
       """
       * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".
      
       $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,
       
       where vᵢₜ ∼ N(0, σᵥ²),
                   σᵥ² = exp(log_σᵥ²) 
                       = exp($(_dicM[:σᵥ²]));
             uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                   hscaleᵢₜ = exp($(_dicM[:hscale])),
             uᵢ ∼ N⁺(μ, σᵤ²),
                  μ = $(_dicM[:μ])
                  σᵤ² = exp(log_σᵤ²) 
                      = exp($(_dicM[:σᵤ²]));
       """
      end
        
      if  Wx!=Nothing   # yuvx
          wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
          T=length(unique(vec(Matrix(tvar))));
          for ttt in 1:T
              if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
             
                  xx = Matrix(dat[!, _dicM[:frontierWx]])
                  @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
              elseif length(wx) > 1
        
                  xx = Matrix(dat[!, _dicM[:frontierWx]])
                  @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
                 
              end	
          end
      end
      
      #* --- retrieve and generate important parameters -----
      
      #*   number of obs and number of variables
      nofx = nofq = nofw = nofv = nofz= nofgamma = nofphi = nofeta = 0  # to make a complete list
      
      nofobs  = nrow(dat)  
      if  Wx!=Nothing   # yuvx
      nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
        else
      nofx    = size(xvar,2) 
        end
      nofq    = size(qvar,2)  # h  
      nofphi  = size(envar,2)*size(ivvar,2)
      nofeta  = size(envar,2)
      nofw    = size(wvar,2)  # sigma_u_2
      nofv    = size(vvar,2)  # sigma_v_2
      nofz    = size(zvar,2)  # mu
      nofgamma    = 1 # wy
    
      
      nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv +nofz +nofgamma
      
      
      nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofgamma=nofgamma,  nofpara=nofpara, nofmarg = nofq+nofw+nofz)
      
      #* positions of the variables/parameters
      begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma =0
      
      begx = 1
      endx = nofx
      begq = endx + 1
      endq = begq + nofq-1
      begphi = endq+1
      endphi = begphi + nofphi-1
      begeta = endphi+1
      endeta = begeta + nofeta-1
      
      begw = endeta + 1
      endw = begw + nofw-1
      begv = endw + 1
      endv = begv + nofv-1
      begz = endv + 1
      endz = begz + nofz-1
      beggamma = endz + 1
      endgamma = beggamma + nofgamma-1
        
      
      posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
                begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
                beggamma=beggamma, endgamma=endgamma)
      
      #* create equation names and mark positions for making tables
      eqvec = (frontier = begx + 1, 
      lnh  = begq + 1,
      ϕ   = begphi + 1,
      η   =  begeta + 1,
  lnσᵤ²   = begw + 1,
  lnσᵥ²   = begv + 1,
      μ   = begz + 1,
      ρ   = beggamma + 1)
  
     
  
      
      #* create equation names and mark positions 
      eqvec2 = (coeff_frontier = (begx:endx), 
      coeff_log_hscale = (begq:endq),
            coeff_ϕ       = (begphi:endphi),
            coeff_η       = (begeta:endeta),
         coeff_log_σᵤ² = (begw:endw),
         coeff_log_σᵥ² = (begv:endv),
         coeff_μ = (begz:endz),
               coeff_γ = (beggamma:endgamma), )        
  
    
      
      #* retrieve variable names for making tables
      if  Wx!=Nothing   # yuvx
      xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
      else
      xnames  = names(xvar)
      end
      
      
      qnames  = names(qvar)
      ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
      etanames = ["η_" * s for s in names(envar)] 
      wnames  = names(wvar)
      vnames  = names(vvar)
      znames  = names(zvar)
      if  Wy!=Nothing    
      gammanames  = "ρ"
      end
  
      varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames)
  
      
      
      #* Converting the dataframe to matrix in order to do computation
      yvar  = convert(Array{Float64}, Matrix(yvar))
      if  Wx!=Nothing   # yuvx
      xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
        else
          xvar  = convert(Array{Float64}, Matrix(xvar))
        end
      qvar  = convert(Array{Float64}, Matrix(qvar))
      wvar  = convert(Array{Float64}, Matrix(wvar))
      vvar  = convert(Array{Float64}, Matrix(vvar))
      zvar  = convert(Array{Float64}, Matrix(zvar))
      tvar  = convert(Array{Float64}, Matrix(tvar))
      ivar  = convert(Array{Float64}, Matrix(ivar))
      
      ivvar  = convert(Array{Float64}, Matrix(ivvar))
      envar  = convert(Array{Float64}, Matrix(envar))
      
      #* various functions can and cannot contain a constant, check! ---- *#
      # checkConst(xvar, :frontier, @requireConst(0))
      # checkConst(qvar, :hscale,   @requireConst(0)) 
      # checkConst(wvar, :σᵤ²,      @requireConst(1))
      # checkConst(vvar, :σᵥ²,      @requireConst(1))
      # checkConst(zvar, :μ,        @requireConst(1))
      
      
      # 获得空间矩阵的特征值
      rymin=rymax=0
      if  Wy!=Nothing   # yuvx
          dylams = eigen(Wy[1])
          rymin = 1 / minimum(real(dylams.values))
          rymax = 1
          if length(Wy) > 1
              for k = 2:length(Wy)
                  dylams = eigen(Wy[k])
                  if rymin < 1 / minimum(real(dylams.values))
                      rymin = 1 / minimum(real(dylams.values))
                  end
              end
          end
        end
   
        
      eigvalu = (rymin=rymin, rymax=rymax)
        indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
        indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
  qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end
      
      
 

function getvar(::Type{SSFKUT}, dat::DataFrame)

ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:timevar][1], _dicM[:idvar][1]])

tvar = dat[:, _dicM[:timevar]]

rowIDT = get_rowIDofT(vec(Matrix(tvar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
if _dicM[:wx]!=Nothing  # yuvx
   Wxvar = dat[:, _dicM[:frontierWx]]   
end
qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]
zvar = dat[:, _dicM[:μ]]

Wy = _dicM[:wy]
Wx = _dicM[:wx]


#* --- model info printout ---------
modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2020 EJoR), normal and truncated-normal"
modelinfo2 = begin
 """
 * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

 $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,
 
 where vᵢₜ ∼ N(0, σᵥ²),
             σᵥ² = exp(log_σᵥ²) 
                 = exp($(_dicM[:σᵥ²]));
       uᵢₜ ∼ hscaleᵢₜ * uᵢ,
             hscaleᵢₜ = exp($(_dicM[:hscale])),
       uᵢ ∼ N⁺(μ, σᵤ²),
            μ = $(_dicM[:μ])
            σᵤ² = exp(log_σᵤ²) 
                = exp($(_dicM[:σᵤ²]));
 """
end
  
if  Wx!=Nothing   # yuvx
    wxvar = zeros(size(dat, 1), length(_dicM[:frontierWx]) )
    T=length(unique(vec(Matrix(tvar))));
    for ttt in 1:T
        if length(Wx) == 1  # 可以传入单个cell的w，则默认cell的长度为时间的长度
       
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
        elseif length(wx) > 1
  
            xx = Matrix(dat[!, _dicM[:frontierWx]])
            @views wxvar[rowIDT[ttt,1], :] .= Wx[1] * xx[rowIDT[ttt,1], :]
           
        end	
    end
end

#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx =  nofq = nofw = nofv = nofz = nofgamma = noftau = nofrho=0  # to make a complete list

nofobs  = nrow(dat)  
if  Wx!=Nothing   # yuvx
nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
  else
nofx    = size(xvar,2) 
  end

nofq    = size(qvar,2)  # h
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
nofz    = size(zvar,2)  # mu
nofgamma    = 1 # wy


nofpara = nofx + nofq + nofw + nofv  +nofgamma+nofz

nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
        nofgamma=nofgamma,nofpara=nofpara, nofmarg = nofq+nofw+nofz)

#* positions of the variables/parameters
begx=endx=begq=endq=begw=endw=begv=endv=begz=endz=beggamma=endgamma =0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begw = endq + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
begz = endv + 1
endz = begz + nofz-1
beggamma = endz + 1
endgamma = beggamma + nofgamma-1

posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
          beggamma=beggamma, endgamma=endgamma )

#* create equation names and mark positions for making tables
eqvec = (frontier = begx + 1, 
lnh = begq + 1,
lnσᵤ² = begw + 1,
lnσᵥ² = begv + 1,
    μ = begz + 1,
    ρ = beggamma + 1 )

#* create equation names and mark positions 

eqvec2 = (coeff_frontier = (begx:endx), 
coeff_log_hscale = (begq:endq),
   coeff_log_σᵤ² = (begw:endw),
   coeff_log_σᵥ² = (begv:endv),
     coeff_μ = (begz:endz),
         coeff_γ = (beggamma:endgamma), )              
          
#* retrieve variable names for making tables
if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["W*" * s for s in names(Wxvar)]   )
  else
xnames  = names(xvar)
  end
  

qnames  = names(qvar)
wnames  = names(wvar)
vnames  = names(vvar)
znames  = names(zvar)
gammanames  = "ρ"


varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames)



#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
if  Wx!=Nothing   # yuvx
xvar  = convert(Array{Float64}, hcat(Matrix(xvar),wxvar))
  else
    xvar  = convert(Array{Float64}, Matrix(xvar))
  end
qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))
zvar  = convert(Array{Float64}, Matrix(zvar))
tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))
envar = ()
ivvar = ()


#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))


# 获得空间矩阵的特征值
rymin=rymax=0
dylams = eigen(Wy[1])
rymin = 1 / minimum(real(dylams.values))
rymax = 1
if length(Wy) > 1
    for k = 2:length(Wy)
        dylams = eigen(Wy[k])
        if rymin < 1 / minimum(real(dylams.values))
            rymin = 1 / minimum(real(dylams.values))
        end
    end
end
  


eigvalu = (rymin=rymin, rymax=rymax)
indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end
    
     
  

function getvar(::Type{SSFKKEH}, dat::DataFrame)

  ivar = dat[:, _dicM[:idvar]] 
  dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
  tvar = dat[:, _dicM[:timevar]]
  rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year
  
  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
  if _dicM[:wx]!=Nothing  # yuvx
     Wxvar = dat[:, _dicM[:frontierWx]]   
  end
  qvar = dat[:, _dicM[:hscale]]  
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]
  # zvar = dat[:, _dicM[:μ]]
  

  envar = dat[:, _dicM[:envar]]   
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
  
  ivvar = dat[:, name_new_ivvar]   
  
  
  #* --- model info printout --------- 
  modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2017 Applied economics), normal and half-normal"
  modelinfo2 = begin
   """
   * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

   $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

   where vᵢₜ ∼ N(0, σᵥ²),
               σᵥ² = exp(log_σᵥ²)
                   = exp($(_dicM[:σᵥ²]));
         uᵢₜ ∼ hscaleᵢₜ * uᵢ,
               hscaleᵢₜ = exp($(_dicM[:hscale])),
         uᵢ ∼ N⁺(0, σᵤ²),
              σᵤ² = exp(log_σᵤ²)
                  = exp($(_dicM[:σᵤ²]));
   """
  end
    
  #* --- retrieve and generate important parameters -----
  #*   number of obs and number of variables
  nofx = nofq = nofw = nofv = nofz=  nofphi = nofeta = 0  # to make a complete list
  
  nofobs  = nrow(dat)  

  nofx    = size(xvar,2) 
  nofq    = size(qvar,2)  # h  
  nofphi  = size(envar,2)*size(ivvar,2)
  nofeta  = size(envar,2)
  nofw    = size(wvar,2)  # sigma_u_2
  nofv    = size(vvar,2)  # sigma_v_2
  # nofz    = size(zvar,2)  # mu

  
  nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv  +nofz

  nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
           nofpara=nofpara, nofmarg = nofq+nofw+nofz)
  
  #* positions of the variables/parameters
  begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz =0
  
  begx = 1
  endx = nofx
  begq = endx + 1
  endq = begq + nofq-1
  begphi = endq+1
  endphi = begphi + nofphi-1
  begeta = endphi+1
  endeta = begeta + nofeta-1
  
  begw = endeta + 1
  endw = begw + nofw-1
  begv = endw + 1
  endv = begv + nofv-1
  # begz = endv + 1
  # endz = begz + nofz-1

    
  
  posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
            begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
  
  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
  lnh  = begq + 1,
  ϕ   = begphi + 1,
  η   =  begeta + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1)

 

  
  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
  coeff_log_hscale = (begq:endq),
        coeff_ϕ       = (begphi:endphi),
        coeff_η       = (begeta:endeta),
     coeff_log_σᵤ² = (begw:endw),
     coeff_log_σᵥ² = (begv:endv) )        


  
  #* retrieve variable names for making tables
  xnames  = names(xvar)

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["η_" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  # znames  = names(zvar)

  varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames)

  
  
  #* Converting the dataframe to matrix in order to do computation
  yvar  = convert(Array{Float64}, Matrix(yvar))
  xvar  = convert(Array{Float64}, Matrix(xvar))
  
  qvar  = convert(Array{Float64}, Matrix(qvar))
  wvar  = convert(Array{Float64}, Matrix(wvar))
  vvar  = convert(Array{Float64}, Matrix(vvar))
  # zvar  = convert(Array{Float64}, Matrix(zvar))
  tvar  = convert(Array{Float64}, Matrix(tvar))
  ivar  = convert(Array{Float64}, Matrix(ivar))
  
  ivvar  = convert(Array{Float64}, Matrix(ivvar))
  envar  = convert(Array{Float64}, Matrix(envar))
  zvar = ()
  
  #* various functions can and cannot contain a constant, check! ---- *#
  # checkConst(xvar, :frontier, @requireConst(0))
  # checkConst(qvar, :hscale,   @requireConst(0)) 
  # checkConst(wvar, :σᵤ²,      @requireConst(1))
  # checkConst(vvar, :σᵥ²,      @requireConst(1))
  # checkConst(zvar, :μ,        @requireConst(1))
  
  
  # 获得空间矩阵的特征值
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
    qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
  end
  

function getvar(::Type{SSFKKH}, dat::DataFrame)


    ivar = dat[:, _dicM[:idvar]]
    dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx

    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    # zvar = dat[:, _dicM[:μ]]

    #* --- model info printout ---------
    modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2017 Applied Economics), normal and half-normal"
    modelinfo2 = begin
      """
      * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

      $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

      where vᵢₜ ∼ N(0, σᵥ²),
                  σᵥ² = exp(log_σᵥ²)
                      = exp($(_dicM[:σᵥ²]));
            uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                  hscaleᵢₜ = exp($(_dicM[:hscale])),
            uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
      """
    end
      
  
    #* --- retrieve and generate important parameters -----
    
    #*   number of obs and number of variables
    nofx = nofq = nofw = nofv = nofz  = 0  # to make a complete list
    
    nofobs  = nrow(dat)  

    nofx    = size(xvar,2) 
    nofq    = size(qvar,2)  # h  
    nofw    = size(wvar,2)  # sigma_u_2
    nofv    = size(vvar,2)  # sigma_v_2
    # nofz    = size(zvar,2)  # mu
  
    
    nofpara = nofx + nofq + nofw + nofv +nofz
    
    
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofpara=nofpara, nofmarg = nofq+nofw+nofz)
    
    #* positions of the variables/parameters
    begx=endx=begq=endq=begw=endw=begv=endv=begz=endz =0
    
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq-1
    begw = endq + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = begv + nofv-1
    # begz = endv + 1
    # endz = begz + nofz-1

      
    
    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
    
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
    lnh  = begq + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
    coeff_log_hscale = (begq:endq),
        coeff_log_σᵤ² = (begw:endw),
        coeff_log_σᵥ² = (begv:endv))        

  
    
    #* retrieve variable names for making tables
    xnames  = names(xvar)

    qnames  = names(qvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    # znames  = names(zvar)

    varlist = vcat(" ", xnames, qnames , wnames, vnames)

    
    
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))

    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    # zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    
    ivvar  = ()
    envar  = ()
    zvar = ()
    
    #* various functions can and cannot contain a constant, check! ---- *#
    # checkConst(xvar, :frontier, @requireConst(0))
    # checkConst(qvar, :hscale,   @requireConst(0)) 
    # checkConst(wvar, :σᵤ²,      @requireConst(1))
    # checkConst(vvar, :σᵥ²,      @requireConst(1))
    # checkConst(zvar, :μ,        @requireConst(1))
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)
  
      indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
      indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
      qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
      
end
      
      
  

function getvar(::Type{SSFKKET}, dat::DataFrame)


    ivar = dat[:, _dicM[:idvar]] 
    dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year
    
    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx

    qvar = dat[:, _dicM[:hscale]]  
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    zvar = dat[:, _dicM[:μ]]

    
    envar = dat[:, _dicM[:envar]]   
      name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
      name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
      name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv
    
    ivvar = dat[:, name_new_ivvar]   
    
    
    #* --- model info printout --------- 
    modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2017 Applied Economics), normal and truncated-normal"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     $(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ])
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     """
    end


    #* --- retrieve and generate important parameters -----

    #*   number of obs and number of variables
    nofx = nofq = nofw = nofv = nofz = nofphi = nofeta = 0  # to make a complete list
    
    nofobs  = nrow(dat)  
    nofx    = size(xvar,2) 

    nofq    = size(qvar,2)  # h  
    nofphi  = size(envar,2)*size(ivvar,2)
    nofeta  = size(envar,2)
    nofw    = size(wvar,2)  # sigma_u_2
    nofv    = size(vvar,2)  # sigma_v_2
    nofz    = size(zvar,2)  # mu
  
    
    nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv +nofz 
    
    
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofpara=nofpara, nofmarg = nofq+nofw+nofz)
    
    #* positions of the variables/parameters
    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz =0
    
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq-1
    begphi = endq+1
    endphi = begphi + nofphi-1
    begeta = endphi+1
    endeta = begeta + nofeta-1
    
    begw = endeta + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = begv + nofv-1
    begz = endv + 1
    endz = begz + nofz-1

      
    
    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
              begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
              )
    
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
    lnh  = begq + 1,
    ϕ   = begphi + 1,
    η   =  begeta + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1,
    μ   = begz + 1)

   

    
    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
    coeff_log_hscale = (begq:endq),
          coeff_ϕ       = (begphi:endphi),
          coeff_η       = (begeta:endeta),
       coeff_log_σᵤ² = (begw:endw),
       coeff_log_σᵥ² = (begv:endv),
       coeff_μ = (begz:endz) )        

  
    
    #* retrieve variable names for making tables
    xnames  = names(xvar)

    qnames  = names(qvar)
    ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
    etanames = ["η_" * s for s in names(envar)] 
    wnames  = names(wvar)
    vnames  = names(vvar)
    znames  = names(zvar)

    varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames)

    
    
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))

    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    
    ivvar  = convert(Array{Float64}, Matrix(ivvar))
    envar  = convert(Array{Float64}, Matrix(envar))
    
    #* various functions can and cannot contain a constant, check! ---- *#
    # checkConst(xvar, :frontier, @requireConst(0))
    # checkConst(qvar, :hscale,   @requireConst(0)) 
    # checkConst(wvar, :σᵤ²,      @requireConst(1))
    # checkConst(vvar, :σᵥ²,      @requireConst(1))
    # checkConst(zvar, :μ,        @requireConst(1))
    
    rymin=rymax=0
    eigvalu = (rymin=rymin, rymax=rymax)
    
    indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
    indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end
    
    


function getvar(::Type{SSFKKT}, dat::DataFrame)

ivar = dat[:, _dicM[:idvar]] 
dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
tvar = dat[:, _dicM[:timevar]]
rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx

qvar = dat[:, _dicM[:hscale]]  
wvar = dat[:, _dicM[:σᵤ²]]
vvar = dat[:, _dicM[:σᵥ²]]
zvar = dat[:, _dicM[:μ]]


#* --- model info printout --------- 
modelinfo1 = "spatial stochastic frontier analysis in Kutlu (2017 Applied Economics), normal and truncated-normal"
modelinfo2 = begin
"""
* In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

$(_dicM[:depvar][1]) = frontier( $(_dicM[:frontier])) + ̃vᵢₜ - ̃uᵢₜ,

where vᵢₜ ∼ N(0, σᵥ²),
           σᵥ² = exp(log_σᵥ²) 
               = exp($(_dicM[:σᵥ²]));
     uᵢₜ ∼ hscaleᵢₜ * uᵢ,
           hscaleᵢₜ = exp($(_dicM[:hscale])),
     uᵢ ∼ N⁺(μ, σᵤ²),
          μ = $(_dicM[:μ])
          σᵤ² = exp(log_σᵤ²) 
              = exp($(_dicM[:σᵤ²]));
"""
end


#* --- retrieve and generate important parameters -----

#*   number of obs and number of variables
nofx =  nofq = nofw = nofv = nofz =0  # to make a complete list

nofobs  = nrow(dat)  
nofx    = size(xvar,2) 


nofq    = size(qvar,2)  # h
nofw    = size(wvar,2)  # sigma_u_2
nofv    = size(vvar,2)  # sigma_v_2
nofz    = size(zvar,2)  # mu


nofpara = nofx + nofq + nofw + nofv +nofz

nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
          nofpara=nofpara, nofmarg = nofq+nofw+nofz)

#* positions of the variables/parameters
begx=endx=begq=endq=begw=endw=begv=endv=begz=endz=0

begx = 1
endx = nofx
begq = endx + 1
endq = begq + nofq-1
begw = endq + 1
endw = begw + nofw-1
begv = endw + 1
endv = begv + nofv-1
begz = endv + 1
endz = begz + nofz-1


posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz )

#* create equation names and mark positions for making tables
eqvec = (frontier = begx + 1, 
lnh = begq + 1,
lnσᵤ² = begw + 1,
lnσᵥ² = begv + 1,
  μ = begz + 1 )

#* create equation names and mark positions 

eqvec2 = (coeff_frontier = (begx:endx), 
coeff_log_hscale = (begq:endq),
 coeff_log_σᵤ² = (begw:endw),
 coeff_log_σᵥ² = (begv:endv),
   coeff_μ = (begz:endz) )              
        
#* retrieve variable names for making tables
xnames  = names(xvar)

qnames  = names(qvar)
wnames  = names(wvar)
vnames  = names(vvar)
znames  = names(zvar)


varlist = vcat(" ", xnames, qnames, wnames, vnames, znames)



#* Converting the dataframe to matrix in order to do computation
yvar  = convert(Array{Float64}, Matrix(yvar))
xvar  = convert(Array{Float64}, Matrix(xvar))

qvar  = convert(Array{Float64}, Matrix(qvar))
wvar  = convert(Array{Float64}, Matrix(wvar))
vvar  = convert(Array{Float64}, Matrix(vvar))
zvar  = convert(Array{Float64}, Matrix(zvar))
tvar  = convert(Array{Float64}, Matrix(tvar))
ivar  = convert(Array{Float64}, Matrix(ivar))
envar = ()
ivvar = ()


#* various functions can and cannot contain a constant, check! ---- *#
# checkConst(xvar, :frontier, @requireConst(0))
# checkConst(qvar, :hscale,   @requireConst(0)) 
# checkConst(wvar, :σᵤ²,      @requireConst(1))
# checkConst(vvar, :σᵥ²,      @requireConst(1))
# checkConst(zvar, :μ,        @requireConst(1))

rymin=rymax=0
eigvalu = (rymin=rymin, rymax=rymax)

indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))


return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end
  
   






# ============ Giannini 2025 (GI) 系列 ============
function getvar(::Type{SSFGIEH}, dat::DataFrame)

    ivar = dat[:, _dicM[:idvar]]
    dat = sort(dat, [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    ivar = dat[:, _dicM[:idvar]]

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    if _dicM[:wx]!=Nothing
       Wxvar = dat[:, _dicM[:frontierWx]]
    end
    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]

    Wy = _dicM[:wy]
    Wx = _dicM[:wx]

    envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])
    ivvar = dat[:, name_new_ivvar]

    modelinfo1 = "spatial SFA in Giannini (2025), first-difference with spatial lag, half-normal, endogeneity"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     Δyᵢₜ = λWΔy + Δxβ + Δvᵢₜ - Δ(hᵢₜuᵢ),

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     frontier: $(_dicM[:frontier])
     endogenous: $(_dicM[:envar])
     instruments: $(_dicM[:ivvar])
     """
    end

    #* variable names for tables
    if Wx!=Nothing
        xnames = vcat(names(xvar), ["W*" * s for s in names(Wxvar)])
    else
        xnames = names(xvar)
    end
    qnames   = names(qvar)
    ivnames  = ["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]
    etanames = ["η_" * s for s in names(envar)]
    wnames   = names(wvar)
    vnames   = names(vvar)
    gammanames = Wy!=Nothing ? "λ" : ""
    varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames)

    #* convert to matrices
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    ivvar = convert(Array{Float64}, Matrix(ivvar))
    envar = convert(Array{Float64}, Matrix(envar))

    nofw_orig = size(wvar, 2)
    nofv_orig = size(vvar, 2)

    # individual-based rowIDT on original data
    rowIDT_orig = get_rowIDT(vec(ivar))

    # __GIEH_PART2__

    # compute frontierWx on original data (before FD)
    if Wx!=Nothing
        rowIDT_t_orig = get_rowIDofT(vec(tvar))
        T_total = size(rowIDT_t_orig, 1)
        wxvar_mat = zeros(size(yvar, 1), length(_dicM[:frontierWx]))
        xx = convert(Array{Float64}, Matrix(dat[!, _dicM[:frontierWx]]))
        for ttt in 1:T_total
            ind_t = rowIDT_t_orig[ttt, 1]
            if length(Wx) == 1
                @views wxvar_mat[ind_t, :] .= Wx[1] * xx[ind_t, :]
            elseif length(Wx) > 1
                @views wxvar_mat[ind_t, :] .= Wx[ttt] * xx[ind_t, :]
            end
        end
        xvar = hcat(xvar, wxvar_mat)
    end

    # --- apply first-difference per individual ---
    ID_orig = size(rowIDT_orig, 1)
    nofobs_fd = 0
    for iidd = 1:ID_orig
        nofobs_fd += rowIDT_orig[iidd, 2] - 1
    end

    yvar_fd    = zeros(nofobs_fd, size(yvar, 2))
    xvar_fd    = zeros(nofobs_fd, size(xvar, 2))
    qvar_cur   = zeros(nofobs_fd, size(qvar, 2))
    qvar_lag   = zeros(nofobs_fd, size(qvar, 2))
    envar_fd   = zeros(nofobs_fd, size(envar, 2))
    ivvar_fd   = zeros(nofobs_fd, size(ivvar, 2))
    ivar_fd    = zeros(nofobs_fd)
    tvar_fd    = zeros(nofobs_fd)

    idx = 0
    for iidd = 1:ID_orig
        ind = rowIDT_orig[iidd, 1]
        T_i = rowIDT_orig[iidd, 2]
        for t = 2:T_i
            idx += 1
            @views yvar_fd[idx, :]  = yvar[ind[t], :]  - yvar[ind[t-1], :]
            @views xvar_fd[idx, :]  = xvar[ind[t], :]  - xvar[ind[t-1], :]
            @views qvar_cur[idx, :] = qvar[ind[t], :]
            @views qvar_lag[idx, :] = qvar[ind[t-1], :]
            @views envar_fd[idx, :] = envar[ind[t], :] - envar[ind[t-1], :]
            @views ivvar_fd[idx, :] = ivvar[ind[t], :] - ivvar[ind[t-1], :]
            ivar_fd[idx] = ivar[ind[t]]
            tvar_fd[idx] = tvar[ind[t]]
        end
    end

    # individual-based rowIDT on FD data
    rowIDT = get_rowIDT(vec(ivar_fd))

    # store qvar_lag in _dicM for likelihood use
    _dicM[:qvar_lag] = qvar_lag

    zvar = ()

    # __GIEH_PART3__

    #* --- parameter counts ---
    nofx = nofq = nofw = nofv = nofz = nofgamma = nofphi = nofeta = 0
    nofobs  = nofobs_fd
    nofx    = size(xvar_fd, 2)
    nofq    = size(qvar_cur, 2)
    nofphi  = size(envar_fd, 2) * size(ivvar_fd, 2)
    nofeta  = size(envar_fd, 2)
    nofw    = nofw_orig
    nofv    = nofv_orig
    nofgamma = 1

    nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv + nofgamma + nofz
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq, nofphi=nofphi, nofeta=nofeta,
              nofw=nofw, nofv=nofv, nofz=nofz, nofgamma=nofgamma,
              nofpara=nofpara, nofmarg=nofq+nofw+nofz)

    #* positions of the variables/parameters
    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=0
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq - 1
    begphi = endq + 1
    endphi = begphi + nofphi - 1
    begeta = endphi + 1
    endeta = begeta + nofeta - 1
    begw = endeta + 1
    endw = begw + nofw - 1
    begv = endw + 1
    endv = begv + nofv - 1
    beggamma = endv + 1
    endgamma = beggamma + nofgamma - 1

    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begphi=begphi, endphi=endphi, begeta=begeta, endeta=endeta,
              begw=begw, endw=endw, begv=begv, endv=endv, begz=begz, endz=endz,
              beggamma=beggamma, endgamma=endgamma)

    eqvec = (frontier=begx+1, lnh=begq+1, ϕ=begphi+1, η=begeta+1,
             lnσᵤ²=begw+1, lnσᵥ²=begv+1, λ=beggamma+1)

    eqvec2 = (coeff_frontier=(begx:endx), coeff_log_hscale=(begq:endq),
              coeff_ϕ=(begphi:endphi), coeff_η=(begeta:endeta),
              coeff_log_σᵤ²=(begw:endw), coeff_log_σᵥ²=(begv:endv),
              coeff_λ=(beggamma:endgamma))

    #* eigenvalues for spatial parameter bounds
    rymin=rymax=0
    if Wy!=Nothing
        dylams = eigen(Wy[1])
        rymin = 1 / minimum(real(dylams.values))
        rymax = 1
        if length(Wy) > 1
            for k = 2:length(Wy)
                dylams = eigen(Wy[k])
                if rymin < 1 / minimum(real(dylams.values))
                    rymin = 1 / minimum(real(dylams.values))
                end
            end
        end
    end
    eigvalu = (rymin=rymin, rymax=rymax)

    indices_list = find_all_indices_ordered(vcat(_dicM[:frontier], _dicM[:frontierWx]))
    indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar_fd, xvar_fd,
        qvar_cur, wvar, vvar, zvar, envar_fd, ivvar_fd, eigvalu, indices_list, indices_listz, rowIDT, varlist

end


# __GI2025_GIH_START__

function getvar(::Type{SSFGIH}, dat::DataFrame)

    ivar = dat[:, _dicM[:idvar]]
    dat = sort(dat, [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    ivar = dat[:, _dicM[:idvar]]

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    if _dicM[:wx]!=Nothing
       Wxvar = dat[:, _dicM[:frontierWx]]
    end
    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]

    Wy = _dicM[:wy]
    Wx = _dicM[:wx]

    modelinfo1 = "spatial SFA in Giannini (2025), first-difference with spatial lag, half-normal"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     Δyᵢₜ = λWΔy + Δxβ + Δvᵢₜ - Δ(hᵢₜuᵢ),

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     frontier: $(_dicM[:frontier])
     """
    end

    if Wx!=Nothing
        xnames = vcat(names(xvar), ["W*" * s for s in names(Wxvar)])
    else
        xnames = names(xvar)
    end
    qnames   = names(qvar)
    wnames   = names(wvar)
    vnames   = names(vvar)
    gammanames = Wy!=Nothing ? "λ" : ""
    varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames)

    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))

    nofw_orig = size(wvar, 2)
    nofv_orig = size(vvar, 2)
    rowIDT_orig = get_rowIDT(vec(ivar))

    # __GIH_PART2__

    # compute frontierWx on original data
    if Wx!=Nothing
        rowIDT_t_orig = get_rowIDofT(vec(tvar))
        T_total = size(rowIDT_t_orig, 1)
        wxvar_mat = zeros(size(yvar, 1), length(_dicM[:frontierWx]))
        xx = convert(Array{Float64}, Matrix(dat[!, _dicM[:frontierWx]]))
        for ttt in 1:T_total
            ind_t = rowIDT_t_orig[ttt, 1]
            if length(Wx) == 1
                @views wxvar_mat[ind_t, :] .= Wx[1] * xx[ind_t, :]
            elseif length(Wx) > 1
                @views wxvar_mat[ind_t, :] .= Wx[ttt] * xx[ind_t, :]
            end
        end
        xvar = hcat(xvar, wxvar_mat)
    end

    # first-difference per individual
    ID_orig = size(rowIDT_orig, 1)
    nofobs_fd = 0
    for iidd = 1:ID_orig
        nofobs_fd += rowIDT_orig[iidd, 2] - 1
    end

    yvar_fd    = zeros(nofobs_fd, size(yvar, 2))
    xvar_fd    = zeros(nofobs_fd, size(xvar, 2))
    qvar_cur   = zeros(nofobs_fd, size(qvar, 2))
    qvar_lag   = zeros(nofobs_fd, size(qvar, 2))
    ivar_fd    = zeros(nofobs_fd)
    tvar_fd    = zeros(nofobs_fd)

    idx = 0
    for iidd = 1:ID_orig
        ind = rowIDT_orig[iidd, 1]
        T_i = rowIDT_orig[iidd, 2]
        for t = 2:T_i
            idx += 1
            @views yvar_fd[idx, :]  = yvar[ind[t], :]  - yvar[ind[t-1], :]
            @views xvar_fd[idx, :]  = xvar[ind[t], :]  - xvar[ind[t-1], :]
            @views qvar_cur[idx, :] = qvar[ind[t], :]
            @views qvar_lag[idx, :] = qvar[ind[t-1], :]
            ivar_fd[idx] = ivar[ind[t]]
            tvar_fd[idx] = tvar[ind[t]]
        end
    end

    rowIDT = get_rowIDT(vec(ivar_fd))
    _dicM[:qvar_lag] = qvar_lag

    envar = ()
    ivvar = ()
    zvar  = ()

    # __GIH_PART3__

    nofx = nofq = nofw = nofv = nofz = nofgamma = nofphi = nofeta = 0
    nofobs  = nofobs_fd
    nofx    = size(xvar_fd, 2)
    nofq    = size(qvar_cur, 2)
    nofw    = nofw_orig
    nofv    = nofv_orig
    nofgamma = 1

    nofpara = nofx + nofq + nofw + nofv + nofgamma + nofz
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq, nofphi=nofphi, nofeta=nofeta,
              nofw=nofw, nofv=nofv, nofz=nofz, nofgamma=nofgamma,
              nofpara=nofpara, nofmarg=nofq+nofw+nofz)

    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=0
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq - 1
    begw = endq + 1
    endw = begw + nofw - 1
    begv = endw + 1
    endv = begv + nofv - 1
    beggamma = endv + 1
    endgamma = beggamma + nofgamma - 1

    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begw=begw, endw=endw, begv=begv, endv=endv, begz=begz, endz=endz,
              beggamma=beggamma, endgamma=endgamma)

    eqvec = (frontier=begx+1, lnh=begq+1,
             lnσᵤ²=begw+1, lnσᵥ²=begv+1, λ=beggamma+1)

    eqvec2 = (coeff_frontier=(begx:endx), coeff_log_hscale=(begq:endq),
              coeff_log_σᵤ²=(begw:endw), coeff_log_σᵥ²=(begv:endv),
              coeff_λ=(beggamma:endgamma))

    rymin=rymax=0
    if Wy!=Nothing
        dylams = eigen(Wy[1])
        rymin = 1 / minimum(real(dylams.values))
        rymax = 1
        if length(Wy) > 1
            for k = 2:length(Wy)
                dylams = eigen(Wy[k])
                if rymin < 1 / minimum(real(dylams.values))
                    rymin = 1 / minimum(real(dylams.values))
                end
            end
        end
    end
    eigvalu = (rymin=rymin, rymax=rymax)

    indices_list = find_all_indices_ordered(vcat(_dicM[:frontier], _dicM[:frontierWx]))
    indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar_fd, xvar_fd,
        qvar_cur, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end


# __GI2025_GIET_START__

function getvar(::Type{SSFGIET}, dat::DataFrame)

    ivar = dat[:, _dicM[:idvar]]
    dat = sort(dat, [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    ivar = dat[:, _dicM[:idvar]]

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    if _dicM[:wx]!=Nothing
       Wxvar = dat[:, _dicM[:frontierWx]]
    end
    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    zvar = dat[:, _dicM[:μ]]

    Wy = _dicM[:wy]
    Wx = _dicM[:wx]

    envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])
    ivvar = dat[:, name_new_ivvar]

    modelinfo1 = "spatial SFA in Giannini (2025), first-difference with spatial lag, truncated-normal, endogeneity"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     Δyᵢₜ = λWΔy + Δxβ + Δvᵢₜ - Δ(hᵢₜuᵢ),

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ])
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     frontier: $(_dicM[:frontier])
     endogenous: $(_dicM[:envar])
     instruments: $(_dicM[:ivvar])
     """
    end

    if Wx!=Nothing
        xnames = vcat(names(xvar), ["W*" * s for s in names(Wxvar)])
    else
        xnames = names(xvar)
    end
    qnames   = names(qvar)
    ivnames  = ["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]
    etanames = ["η_" * s for s in names(envar)]
    wnames   = names(wvar)
    vnames   = names(vvar)
    znames   = names(zvar)
    gammanames = Wy!=Nothing ? "λ" : ""
    varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames, gammanames)

    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    ivvar = convert(Array{Float64}, Matrix(ivvar))
    envar = convert(Array{Float64}, Matrix(envar))

    nofw_orig = size(wvar, 2)
    nofv_orig = size(vvar, 2)
    nofz_orig = size(zvar, 2)
    rowIDT_orig = get_rowIDT(vec(ivar))

    # __GIET_PART2__

    if Wx!=Nothing
        rowIDT_t_orig = get_rowIDofT(vec(tvar))
        T_total = size(rowIDT_t_orig, 1)
        wxvar_mat = zeros(size(yvar, 1), length(_dicM[:frontierWx]))
        xx = convert(Array{Float64}, Matrix(dat[!, _dicM[:frontierWx]]))
        for ttt in 1:T_total
            ind_t = rowIDT_t_orig[ttt, 1]
            if length(Wx) == 1
                @views wxvar_mat[ind_t, :] .= Wx[1] * xx[ind_t, :]
            elseif length(Wx) > 1
                @views wxvar_mat[ind_t, :] .= Wx[ttt] * xx[ind_t, :]
            end
        end
        xvar = hcat(xvar, wxvar_mat)
    end

    ID_orig = size(rowIDT_orig, 1)
    nofobs_fd = 0
    for iidd = 1:ID_orig
        nofobs_fd += rowIDT_orig[iidd, 2] - 1
    end

    yvar_fd    = zeros(nofobs_fd, size(yvar, 2))
    xvar_fd    = zeros(nofobs_fd, size(xvar, 2))
    qvar_cur   = zeros(nofobs_fd, size(qvar, 2))
    qvar_lag   = zeros(nofobs_fd, size(qvar, 2))
    envar_fd   = zeros(nofobs_fd, size(envar, 2))
    ivvar_fd   = zeros(nofobs_fd, size(ivvar, 2))
    ivar_fd    = zeros(nofobs_fd)
    tvar_fd    = zeros(nofobs_fd)

    idx = 0
    for iidd = 1:ID_orig
        ind = rowIDT_orig[iidd, 1]
        T_i = rowIDT_orig[iidd, 2]
        for t = 2:T_i
            idx += 1
            @views yvar_fd[idx, :]  = yvar[ind[t], :]  - yvar[ind[t-1], :]
            @views xvar_fd[idx, :]  = xvar[ind[t], :]  - xvar[ind[t-1], :]
            @views qvar_cur[idx, :] = qvar[ind[t], :]
            @views qvar_lag[idx, :] = qvar[ind[t-1], :]
            @views envar_fd[idx, :] = envar[ind[t], :] - envar[ind[t-1], :]
            @views ivvar_fd[idx, :] = ivvar[ind[t], :] - ivvar[ind[t-1], :]
            ivar_fd[idx] = ivar[ind[t]]
            tvar_fd[idx] = tvar[ind[t]]
        end
    end

    rowIDT = get_rowIDT(vec(ivar_fd))
    _dicM[:qvar_lag] = qvar_lag

    # __GIET_PART3__

    nofx = nofq = nofw = nofv = nofz = nofgamma = nofphi = nofeta = 0
    nofobs  = nofobs_fd
    nofx    = size(xvar_fd, 2)
    nofq    = size(qvar_cur, 2)
    nofphi  = size(envar_fd, 2) * size(ivvar_fd, 2)
    nofeta  = size(envar_fd, 2)
    nofw    = nofw_orig
    nofv    = nofv_orig
    nofz    = nofz_orig
    nofgamma = 1

    nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv + nofz + nofgamma
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq, nofphi=nofphi, nofeta=nofeta,
              nofw=nofw, nofv=nofv, nofz=nofz, nofgamma=nofgamma,
              nofpara=nofpara, nofmarg=nofq+nofw+nofz)

    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=0
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq - 1
    begphi = endq + 1
    endphi = begphi + nofphi - 1
    begeta = endphi + 1
    endeta = begeta + nofeta - 1
    begw = endeta + 1
    endw = begw + nofw - 1
    begv = endw + 1
    endv = begv + nofv - 1
    begz = endv + 1
    endz = begz + nofz - 1
    beggamma = endz + 1
    endgamma = beggamma + nofgamma - 1

    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begphi=begphi, endphi=endphi, begeta=begeta, endeta=endeta,
              begw=begw, endw=endw, begv=begv, endv=endv, begz=begz, endz=endz,
              beggamma=beggamma, endgamma=endgamma)

    eqvec = (frontier=begx+1, lnh=begq+1, ϕ=begphi+1, η=begeta+1,
             lnσᵤ²=begw+1, lnσᵥ²=begv+1, μ=begz+1, λ=beggamma+1)

    eqvec2 = (coeff_frontier=(begx:endx), coeff_log_hscale=(begq:endq),
              coeff_ϕ=(begphi:endphi), coeff_η=(begeta:endeta),
              coeff_log_σᵤ²=(begw:endw), coeff_log_σᵥ²=(begv:endv),
              coeff_μ=(begz:endz), coeff_λ=(beggamma:endgamma))

    rymin=rymax=0
    if Wy!=Nothing
        dylams = eigen(Wy[1])
        rymin = 1 / minimum(real(dylams.values))
        rymax = 1
        if length(Wy) > 1
            for k = 2:length(Wy)
                dylams = eigen(Wy[k])
                if rymin < 1 / minimum(real(dylams.values))
                    rymin = 1 / minimum(real(dylams.values))
                end
            end
        end
    end
    eigvalu = (rymin=rymin, rymax=rymax)

    indices_list = find_all_indices_ordered(vcat(_dicM[:frontier], _dicM[:frontierWx]))
    indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar_fd, xvar_fd,
        qvar_cur, wvar, vvar, zvar, envar_fd, ivvar_fd, eigvalu, indices_list, indices_listz, rowIDT, varlist

end


# __GI2025_GIT_START__

function getvar(::Type{SSFGIT}, dat::DataFrame)

    ivar = dat[:, _dicM[:idvar]]
    dat = sort(dat, [_dicM[:idvar][1], _dicM[:timevar][1]])
    tvar = dat[:, _dicM[:timevar]]
    ivar = dat[:, _dicM[:idvar]]

    yvar = dat[:, _dicM[:depvar]]
    xvar = dat[:, _dicM[:frontier]]
    if _dicM[:wx]!=Nothing
       Wxvar = dat[:, _dicM[:frontierWx]]
    end
    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    zvar = dat[:, _dicM[:μ]]

    Wy = _dicM[:wy]
    Wx = _dicM[:wx]

    modelinfo1 = "spatial SFA in Giannini (2025), first-difference with spatial lag, truncated-normal"
    modelinfo2 = begin
     """
     * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

     Δyᵢₜ = λWΔy + Δxβ + Δvᵢₜ - Δ(hᵢₜuᵢ),

     where vᵢₜ ∼ N(0, σᵥ²),
                 σᵥ² = exp(log_σᵥ²)
                     = exp($(_dicM[:σᵥ²]));
           uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                 hscaleᵢₜ = exp($(_dicM[:hscale])),
           uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ])
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
     frontier: $(_dicM[:frontier])
     """
    end

    if Wx!=Nothing
        xnames = vcat(names(xvar), ["W*" * s for s in names(Wxvar)])
    else
        xnames = names(xvar)
    end
    qnames   = names(qvar)
    wnames   = names(wvar)
    vnames   = names(vvar)
    znames   = names(zvar)
    gammanames = Wy!=Nothing ? "λ" : ""
    varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames)

    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))
    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))

    nofw_orig = size(wvar, 2)
    nofv_orig = size(vvar, 2)
    nofz_orig = size(zvar, 2)
    rowIDT_orig = get_rowIDT(vec(ivar))

    # __GIT_PART2__

    if Wx!=Nothing
        rowIDT_t_orig = get_rowIDofT(vec(tvar))
        T_total = size(rowIDT_t_orig, 1)
        wxvar_mat = zeros(size(yvar, 1), length(_dicM[:frontierWx]))
        xx = convert(Array{Float64}, Matrix(dat[!, _dicM[:frontierWx]]))
        for ttt in 1:T_total
            ind_t = rowIDT_t_orig[ttt, 1]
            if length(Wx) == 1
                @views wxvar_mat[ind_t, :] .= Wx[1] * xx[ind_t, :]
            elseif length(Wx) > 1
                @views wxvar_mat[ind_t, :] .= Wx[ttt] * xx[ind_t, :]
            end
        end
        xvar = hcat(xvar, wxvar_mat)
    end

    ID_orig = size(rowIDT_orig, 1)
    nofobs_fd = 0
    for iidd = 1:ID_orig
        nofobs_fd += rowIDT_orig[iidd, 2] - 1
    end

    yvar_fd    = zeros(nofobs_fd, size(yvar, 2))
    xvar_fd    = zeros(nofobs_fd, size(xvar, 2))
    qvar_cur   = zeros(nofobs_fd, size(qvar, 2))
    qvar_lag   = zeros(nofobs_fd, size(qvar, 2))
    ivar_fd    = zeros(nofobs_fd)
    tvar_fd    = zeros(nofobs_fd)

    idx = 0
    for iidd = 1:ID_orig
        ind = rowIDT_orig[iidd, 1]
        T_i = rowIDT_orig[iidd, 2]
        for t = 2:T_i
            idx += 1
            @views yvar_fd[idx, :]  = yvar[ind[t], :]  - yvar[ind[t-1], :]
            @views xvar_fd[idx, :]  = xvar[ind[t], :]  - xvar[ind[t-1], :]
            @views qvar_cur[idx, :] = qvar[ind[t], :]
            @views qvar_lag[idx, :] = qvar[ind[t-1], :]
            ivar_fd[idx] = ivar[ind[t]]
            tvar_fd[idx] = tvar[ind[t]]
        end
    end

    rowIDT = get_rowIDT(vec(ivar_fd))
    _dicM[:qvar_lag] = qvar_lag

    envar = ()
    ivvar = ()

    # __GIT_PART3__

    nofx = nofq = nofw = nofv = nofz = nofgamma = nofphi = nofeta = 0
    nofobs  = nofobs_fd
    nofx    = size(xvar_fd, 2)
    nofq    = size(qvar_cur, 2)
    nofw    = nofw_orig
    nofv    = nofv_orig
    nofz    = nofz_orig
    nofgamma = 1

    nofpara = nofx + nofq + nofw + nofv + nofz + nofgamma
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq, nofphi=nofphi, nofeta=nofeta,
              nofw=nofw, nofv=nofv, nofz=nofz, nofgamma=nofgamma,
              nofpara=nofpara, nofmarg=nofq+nofw+nofz)

    begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz=beggamma=endgamma=0
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq - 1
    begw = endq + 1
    endw = begw + nofw - 1
    begv = endw + 1
    endv = begv + nofv - 1
    begz = endv + 1
    endz = begz + nofz - 1
    beggamma = endz + 1
    endgamma = beggamma + nofgamma - 1

    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begw=begw, endw=endw, begv=begv, endv=endv, begz=begz, endz=endz,
              beggamma=beggamma, endgamma=endgamma)

    eqvec = (frontier=begx+1, lnh=begq+1,
             lnσᵤ²=begw+1, lnσᵥ²=begv+1, μ=begz+1, λ=beggamma+1)

    eqvec2 = (coeff_frontier=(begx:endx), coeff_log_hscale=(begq:endq),
              coeff_log_σᵤ²=(begw:endw), coeff_log_σᵥ²=(begv:endv),
              coeff_μ=(begz:endz), coeff_λ=(beggamma:endgamma))

    rymin=rymax=0
    if Wy!=Nothing
        dylams = eigen(Wy[1])
        rymin = 1 / minimum(real(dylams.values))
        rymax = 1
        if length(Wy) > 1
            for k = 2:length(Wy)
                dylams = eigen(Wy[k])
                if rymin < 1 / minimum(real(dylams.values))
                    rymin = 1 / minimum(real(dylams.values))
                end
            end
        end
    end
    eigvalu = (rymin=rymin, rymax=rymax)

    indices_list = find_all_indices_ordered(vcat(_dicM[:frontier], _dicM[:frontierWx]))
    indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))

    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar_fd, xvar_fd,
        qvar_cur, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist

end



function getvar(::Type{SSFWHEH}, dat::DataFrame)

  ivar = dat[:, _dicM[:idvar]]
  dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
  tvar = dat[:, _dicM[:timevar]]
  rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
  if _dicM[:wx]!=Nothing  # yuvx
     Wxvar = dat[:, _dicM[:frontierWx]]
  end
  qvar = dat[:, _dicM[:hscale]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]
  # zvar = dat[:, _dicM[:μ]]


  envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv

  ivvar = dat[:, name_new_ivvar]


  #* --- model info printout ---------
  modelinfo1 = "stochastic frontier analysis in Wang and Ho (2010) with endogeneity in Kutlu et al. (2019), normal and half-normal"
  modelinfo2 = begin
   """
   * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

   $(_dicM[:depvar][1]) = αᵢ + frontier( $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,

   where vᵢₜ ∼ N(0, σᵥ²),
               σᵥ² = exp(log_σᵥ²)
                   = exp($(_dicM[:σᵥ²]));
         uᵢₜ ∼ hscaleᵢₜ * uᵢ,
               hscaleᵢₜ = exp($(_dicM[:hscale])),
         uᵢ ∼ N⁺(0, σᵤ²),
              σᵤ² = exp(log_σᵤ²)
                  = exp($(_dicM[:σᵤ²]));
   αᵢ: individual fixed effects (removed by within-transformation)
   endogenous: $(_dicM[:envar])
   instruments: $(_dicM[:ivvar])
   """
  end
    
  #* --- retrieve and generate important parameters -----
  #*   number of obs and number of variables
  nofx = nofq = nofw = nofv = nofz=  nofphi = nofeta = 0  # to make a complete list
  
  nofobs  = nrow(dat)  

  nofx    = size(xvar,2) 
  nofq    = size(qvar,2)  # h  
  nofphi  = size(envar,2)*size(ivvar,2)
  nofeta  = size(envar,2)
  nofw    = size(wvar,2)  # sigma_u_2
  nofv    = size(vvar,2)  # sigma_v_2
  # nofz    = size(zvar,2)  # mu

  
  nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv  +nofz

  nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
           nofpara=nofpara, nofmarg = nofq+nofw+nofz)
  
  #* positions of the variables/parameters
  begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz =0
  
  begx = 1
  endx = nofx
  begq = endx + 1
  endq = begq + nofq-1
  begphi = endq+1
  endphi = begphi + nofphi-1
  begeta = endphi+1
  endeta = begeta + nofeta-1
  
  begw = endeta + 1
  endw = begw + nofw-1
  begv = endw + 1
  endv = begv + nofv-1
  # begz = endv + 1
  # endz = begz + nofz-1

    
  
  posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
            begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
  
  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
  lnh  = begq + 1,
  ϕ   = begphi + 1,
  η   =  begeta + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1)

 

  
  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
  coeff_log_hscale = (begq:endq),
        coeff_ϕ       = (begphi:endphi),
        coeff_η       = (begeta:endeta),
     coeff_log_σᵤ² = (begw:endw),
     coeff_log_σᵥ² = (begv:endv) )        


  
  #* retrieve variable names for making tables
  xnames  = names(xvar)

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["η_" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  # znames  = names(zvar)

  varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames)

  
  
  #* Converting the dataframe to matrix in order to do computation
  yvar  = convert(Array{Float64}, Matrix(yvar))
  xvar  = convert(Array{Float64}, Matrix(xvar))
  
  qvar  = convert(Array{Float64}, Matrix(qvar))
  wvar  = convert(Array{Float64}, Matrix(wvar))
  vvar  = convert(Array{Float64}, Matrix(vvar))
  # zvar  = convert(Array{Float64}, Matrix(zvar))
  tvar  = convert(Array{Float64}, Matrix(tvar))
  ivar  = convert(Array{Float64}, Matrix(ivar))
  
  ivvar  = convert(Array{Float64}, Matrix(ivvar))
  envar  = convert(Array{Float64}, Matrix(envar))
  zvar = ()
  
  #* various functions can and cannot contain a constant, check! ---- *#
  # checkConst(xvar, :frontier, @requireConst(0))
  # checkConst(qvar, :hscale,   @requireConst(0)) 
  # checkConst(wvar, :σᵤ²,      @requireConst(1))
  # checkConst(vvar, :σᵥ²,      @requireConst(1))
  # checkConst(zvar, :μ,        @requireConst(1))
  
  
  # 获得空间矩阵的特征值
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))
  # println(yvar,"yvar")
  ID = size(rowIDT,1)
  @floop begin
  @inbounds  for iidd=1:ID  
  @views T = rowIDT[iidd,2];
         onecol = ones(T, 1);
         IMT = (I(T)-onecol*inv(onecol'*onecol)*onecol');
  @views ind = rowIDT[iidd,1];
          yvar[ind,:] =IMT * yvar[ind,:];
          xvar[ind,:] =IMT * xvar[ind,:];
          envar[ind,:] =IMT * envar[ind,:];
          ivvar[ind,:] =IMT * ivvar[ind,:];
        end # for ttt=1:ID
      end # begin


  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
    qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
  end
  

function getvar(::Type{SSFWHH}, dat::DataFrame)

  dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])

    ivar = dat[:, _dicM[:idvar]]
    tvar = dat[:, _dicM[:timevar]]
    rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx

    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    # zvar = dat[:, _dicM[:μ]]

    #* --- model info printout ---------
    modelinfo1 = "stochastic frontier analysis in Wang and Ho (2010), normal and half-normal"
    modelinfo2 = begin
      """
      * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

      $(_dicM[:depvar][1]) = αᵢ + frontier( $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,

      where vᵢₜ ∼ N(0, σᵥ²),
                  σᵥ² = exp(log_σᵥ²)
                      = exp($(_dicM[:σᵥ²]));
            uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                  hscaleᵢₜ = exp($(_dicM[:hscale])),
            uᵢ ∼ N⁺(0, σᵤ²),
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
      αᵢ: individual fixed effects (removed by within-transformation)
      """
    end


    #* --- retrieve and generate important parameters -----
    
    #*   number of obs and number of variables
    nofx = nofq = nofw = nofv = nofz  = 0  # to make a complete list
    
    nofobs  = nrow(dat)  

    nofx    = size(xvar,2) 
    nofq    = size(qvar,2)  # h  
    nofw    = size(wvar,2)  # sigma_u_2
    nofv    = size(vvar,2)  # sigma_v_2
    # nofz    = size(zvar,2)  # mu
  
    
    nofpara = nofx + nofq + nofw + nofv +nofz
    
    
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofpara=nofpara, nofmarg = nofq+nofw+nofz)
    
    #* positions of the variables/parameters
    begx=endx=begq=endq=begw=endw=begv=endv=begz=endz =0
    
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq-1
    begw = endq + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = begv + nofv-1
    # begz = endv + 1
    # endz = begz + nofz-1

      
    
    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
    
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
    lnh  = begq + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
    coeff_log_hscale = (begq:endq),
        coeff_log_σᵤ² = (begw:endw),
        coeff_log_σᵥ² = (begv:endv))        

  
    
    #* retrieve variable names for making tables
    xnames  = names(xvar)

    qnames  = names(qvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    # znames  = names(zvar)

    varlist = vcat(" ", xnames, qnames , wnames, vnames)

    
    
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))

    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    # zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    
    ivvar  = ()
    envar  = ()
    zvar = ()
    
    #* various functions can and cannot contain a constant, check! ---- *#
    # checkConst(xvar, :frontier, @requireConst(0))
    # checkConst(qvar, :hscale,   @requireConst(0)) 
    # checkConst(wvar, :σᵤ²,      @requireConst(1))
    # checkConst(vvar, :σᵥ²,      @requireConst(1))
    # checkConst(zvar, :μ,        @requireConst(1))
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)
  
      indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
      indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))
      # println(yvar,"yvar")
      yvar2 = zeros(nofobs, 1 )
      xvar2 = zeros(nofobs, nofx )

      ID = size(rowIDT,1)
      @floop begin
      @inbounds  for iidd=1:ID  
      @views T = rowIDT[iidd,2];
             onecol = ones(T, 1);
             IMT = (I(T)-onecol*inv(onecol'*onecol)*onecol');
      @views ind = rowIDT[iidd,1];
              # yvar2[ind] =sf_demean( yvar[ind,:]);
              # xvar2[ind,:] =sf_demean( xvar[ind,:]);
              yvar2[ind,:] =IMT * yvar[ind,:];
              xvar2[ind,:] =IMT * xvar[ind,:];

              # envar[ind] =IMT * envar[ind];
              # ivvar[ind] =IMT * ivvar[ind];
            end # for ttt=1:ID
          end # begin
    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar2, xvar2, 
      qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
      
end
      










function getvar(::Type{SSFWHET}, dat::DataFrame)

  dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])
  ivar = dat[:, _dicM[:idvar]]
  tvar = dat[:, _dicM[:timevar]]
  rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

  yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
  xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx
  if _dicM[:wx]!=Nothing  # yuvx
     Wxvar = dat[:, _dicM[:frontierWx]]
  end
  qvar = dat[:, _dicM[:hscale]]
  wvar = dat[:, _dicM[:σᵤ²]]
  vvar = dat[:, _dicM[:σᵥ²]]
  zvar = dat[:, _dicM[:μ]]


  envar = dat[:, _dicM[:envar]]
    name_xuvar = unique(union(_dicM[:frontier], _dicM[:hscale]), dims=1)  #  frontier + h (xu) 中的变量
    name_exovar = unique(setdiff(name_xuvar, _dicM[:envar]), dims=1)  # xu中的所以外生变量
    name_new_ivvar = union(name_exovar, _dicM[:ivvar])  # xu中的所以外生变量 + iv

  ivvar = dat[:, name_new_ivvar]


  #* --- model info printout ---------
  modelinfo1 = "stochastic frontier analysis in Wang and Ho (2010) with endogeneity in Kutlu et al. (2019), normal and truncated-normal"
  modelinfo2 = begin
   """
   * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

   $(_dicM[:depvar][1]) = αᵢ + frontier( $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,

   where vᵢₜ ∼ N(0, σᵥ²),
               σᵥ² = exp(log_σᵥ²)
                   = exp($(_dicM[:σᵥ²]));
         uᵢₜ ∼ hscaleᵢₜ * uᵢ,
               hscaleᵢₜ = exp($(_dicM[:hscale])),
         uᵢ ∼ N⁺(μ, σᵤ²),
              μ = $(_dicM[:μ])
              σᵤ² = exp(log_σᵤ²)
                  = exp($(_dicM[:σᵤ²]));
   αᵢ: individual fixed effects (removed by within-transformation)
   endogenous: $(_dicM[:envar])
   instruments: $(_dicM[:ivvar])
   """
  end
    
  #* --- retrieve and generate important parameters -----
  #*   number of obs and number of variables
  nofx = nofq = nofw = nofv = nofz=  nofphi = nofeta = 0  # to make a complete list
  
  nofobs  = nrow(dat)  

  nofx    = size(xvar,2) 
  nofq    = size(qvar,2)  # h  
  nofphi  = size(envar,2)*size(ivvar,2)
  nofeta  = size(envar,2)
  nofw    = size(wvar,2)  # sigma_u_2
  nofv    = size(vvar,2)  # sigma_v_2
  nofz    = size(zvar,2)  # mu

  
  nofpara = nofx + nofq + nofphi + nofeta + nofw + nofv  +nofz

  nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofphi=nofphi,nofeta=nofeta,nofw=nofw, nofv=nofv, nofz=nofz, 
           nofpara=nofpara, nofmarg = nofq+nofw+nofz)
  
  #* positions of the variables/parameters
  begx=endx=begq=endq=begphi=endphi=begeta=endeta=begw=endw=begv=endv=begz=endz =0
  
  begx = 1
  endx = nofx
  begq = endx + 1
  endq = begq + nofq-1
  begphi = endq+1
  endphi = begphi + nofphi-1
  begeta = endphi+1
  endeta = begeta + nofeta-1
  
  begw = endeta + 1
  endw = begw + nofw-1
  begv = endw + 1
  endv = begv + nofv-1
  begz = endv + 1
  endz = begz + nofz-1

    
  
  posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
            begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
  
  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
  lnh  = begq + 1,
  ϕ   = begphi + 1,
  η   =  begeta + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1,
μ       = begz + 1
)

 

  
  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
  coeff_log_hscale = (begq:endq),
        coeff_ϕ       = (begphi:endphi),
        coeff_η       = (begeta:endeta),
     coeff_log_σᵤ² = (begw:endw),
     coeff_log_σᵥ² = (begv:endv) ,
     coeff_μ      = (begz:endz)

     )        


  
  #* retrieve variable names for making tables
  xnames  = names(xvar)

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["η_" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  znames  = names(zvar)

  varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, znames)

  
  
  #* Converting the dataframe to matrix in order to do computation
  yvar  = convert(Array{Float64}, Matrix(yvar))
  xvar  = convert(Array{Float64}, Matrix(xvar))
  
  qvar  = convert(Array{Float64}, Matrix(qvar))
  wvar  = convert(Array{Float64}, Matrix(wvar))
  vvar  = convert(Array{Float64}, Matrix(vvar))
  zvar  = convert(Array{Float64}, Matrix(zvar))
  tvar  = convert(Array{Float64}, Matrix(tvar))
  ivar  = convert(Array{Float64}, Matrix(ivar))
  
  ivvar  = convert(Array{Float64}, Matrix(ivvar))
  envar  = convert(Array{Float64}, Matrix(envar))
  # zvar = ()
  
  #* various functions can and cannot contain a constant, check! ---- *#
  # checkConst(xvar, :frontier, @requireConst(0))
  # checkConst(qvar, :hscale,   @requireConst(0)) 
  # checkConst(wvar, :σᵤ²,      @requireConst(1))
  # checkConst(vvar, :σᵥ²,      @requireConst(1))
  # checkConst(zvar, :μ,        @requireConst(1))
  
  
  # 获得空间矩阵的特征值
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
  indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))
  # println(yvar,"yvar")
  ID = size(rowIDT,1)
  @floop begin
  @inbounds  for iidd=1:ID  
  @views T = rowIDT[iidd,2];
         onecol = ones(T, 1);
         IMT = (I(T)-onecol*inv(onecol'*onecol)*onecol');
  @views ind = rowIDT[iidd,1];
          yvar[ind,:] =IMT * yvar[ind,:];
          xvar[ind,:] =IMT * xvar[ind,:];
          envar[ind,:] =IMT * envar[ind,:];
          ivvar[ind,:] =IMT * ivvar[ind,:];
        end # for ttt=1:ID
      end # begin


  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, 
    qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
  end
  

function getvar(::Type{SSFWHT}, dat::DataFrame)

  dat = sort(dat,  [_dicM[:idvar][1], _dicM[:timevar][1]])

    ivar = dat[:, _dicM[:idvar]]
    tvar = dat[:, _dicM[:timevar]]
    rowIDT = get_rowIDT(vec(Matrix(ivar)))   # rowIDT (Nx2): col_1 is panel's row info; col_2 is panel's number of id in each year

    yvar = dat[:, _dicM[:depvar]]   # still a DataFrame
    xvar = dat[:, _dicM[:frontier]]  ## 如果有wx，则要在这里合并一下，x和wx

    qvar = dat[:, _dicM[:hscale]]
    wvar = dat[:, _dicM[:σᵤ²]]
    vvar = dat[:, _dicM[:σᵥ²]]
    zvar = dat[:, _dicM[:μ]]

    #* --- model info printout ---------
    modelinfo1 = "stochastic frontier analysis in Wang and Ho (2010), normal and truncated-normal"
    modelinfo2 = begin
      """
      * In the case of type(cost), "- uᵢₜ" below is changed to "+ uᵢₜ".

      $(_dicM[:depvar][1]) = αᵢ + frontier( $(_dicM[:frontier])) + vᵢₜ - uᵢₜ,

      where vᵢₜ ∼ N(0, σᵥ²),
                  σᵥ² = exp(log_σᵥ²)
                      = exp($(_dicM[:σᵥ²]));
            uᵢₜ ∼ hscaleᵢₜ * uᵢ,
                  hscaleᵢₜ = exp($(_dicM[:hscale])),
            uᵢ ∼ N⁺(μ, σᵤ²),
                μ = $(_dicM[:μ])
                σᵤ² = exp(log_σᵤ²)
                    = exp($(_dicM[:σᵤ²]));
      αᵢ: individual fixed effects (removed by within-transformation)
      """
    end


    #* --- retrieve and generate important parameters -----

    #*   number of obs and number of variables
    nofx = nofq = nofw = nofv = nofz  = 0  # to make a complete list
    
    nofobs  = nrow(dat)  

    nofx    = size(xvar,2) 
    nofq    = size(qvar,2)  # h  
    nofw    = size(wvar,2)  # sigma_u_2
    nofv    = size(vvar,2)  # sigma_v_2
    nofz    = size(zvar,2)  # mu
  
    
    nofpara = nofx + nofq + nofw + nofv +nofz
    
    
    nofvar = (nofobs=nofobs, nofx=nofx, nofq=nofq,nofw=nofw, nofv=nofv, nofz=nofz, 
              nofpara=nofpara, nofmarg = nofq+nofw+nofz)
    
    #* positions of the variables/parameters
    begx=endx=begq=endq=begw=endw=begv=endv=begz=endz =0
    
    begx = 1
    endx = nofx
    begq = endx + 1
    endq = begq + nofq-1
    begw = endq + 1
    endw = begw + nofw-1
    begv = endw + 1
    endv = begv + nofv-1
    begz = endv + 1
    endz = begz + nofz-1

      
    
    posvec = (begx=begx, endx=endx, begq=begq, endq=endq,
              begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz)
    
    #* create equation names and mark positions for making tables
    eqvec = (frontier = begx + 1, 
    lnh  = begq + 1,
lnσᵤ²   = begw + 1,
lnσᵥ²   = begv + 1,
μ       = begz + 1
)

    #* create equation names and mark positions 
    eqvec2 = (coeff_frontier = (begx:endx), 
    coeff_log_hscale = (begq:endq),
        coeff_log_σᵤ² = (begw:endw),
        coeff_log_σᵥ² = (begv:endv),
        coeff_μ      = (begz:endz)
        )        

  
    
    #* retrieve variable names for making tables
    xnames  = names(xvar)

    qnames  = names(qvar)
    wnames  = names(wvar)
    vnames  = names(vvar)
    znames  = names(zvar)

    varlist = vcat(" ", xnames, qnames , wnames, vnames,znames)

    
    
    #* Converting the dataframe to matrix in order to do computation
    yvar  = convert(Array{Float64}, Matrix(yvar))
    xvar  = convert(Array{Float64}, Matrix(xvar))

    qvar  = convert(Array{Float64}, Matrix(qvar))
    wvar  = convert(Array{Float64}, Matrix(wvar))
    vvar  = convert(Array{Float64}, Matrix(vvar))
    zvar  = convert(Array{Float64}, Matrix(zvar))
    tvar  = convert(Array{Float64}, Matrix(tvar))
    ivar  = convert(Array{Float64}, Matrix(ivar))
    
    ivvar  = ()
    envar  = ()
    # zvar = ()
    
    #* various functions can and cannot contain a constant, check! ---- *#
    # checkConst(xvar, :frontier, @requireConst(0))
    # checkConst(qvar, :hscale,   @requireConst(0)) 
    # checkConst(wvar, :σᵤ²,      @requireConst(1))
    # checkConst(vvar, :σᵥ²,      @requireConst(1))
    # checkConst(zvar, :μ,        @requireConst(1))
  rymin=rymax=0
  eigvalu = (rymin=rymin, rymax=rymax)
  
      indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))
      indices_listz = find_all_indices_ordered(vcat(_dicM[:hscale]))
      # println(yvar,"yvar")
      yvar2 = zeros(nofobs, 1 )
      xvar2 = zeros(nofobs, nofx )

      ID = size(rowIDT,1)
      @floop begin
      @inbounds  for iidd=1:ID  
      @views T = rowIDT[iidd,2];
             onecol = ones(T, 1);
             IMT = (I(T)-onecol*inv(onecol'*onecol)*onecol');
      @views ind = rowIDT[iidd,1];
              # yvar2[ind] =sf_demean( yvar[ind,:]);
              # xvar2[ind,:] =sf_demean( xvar[ind,:]);
              yvar2[ind,:] =IMT * yvar[ind,:];
              xvar2[ind,:] =IMT * xvar[ind,:];

              # envar[ind] =IMT * envar[ind];
              # ivvar[ind] =IMT * ivvar[ind];
            end # for ttt=1:ID
          end # begin
    return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar2, xvar2, 
      qvar, wvar, vvar, zvar, envar, ivvar, eigvalu, indices_list, indices_listz, rowIDT, varlist
      
end
