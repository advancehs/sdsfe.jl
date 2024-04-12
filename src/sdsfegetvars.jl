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



#?--------- panel SSF Orea and Al, truncated normal ----------------


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
              @views wxvar[rowIDT[ttt,1], :] .= Wx[ttt] * xx[rowIDT[ttt,1], :]
             
          end	
      end
  end

  #* --- retrieve and generate important parameters -----

  #*   number of obs and number of variables
  nofx =  nofq = nofw = nofv = nofz = 0  # to make a complete list

  nofobs  = nrow(dat)  
  if  Wx!=Nothing   # yuvx
  nofx    = size(xvar,2) + size(wxvar,2)  # nofx: number of x + wx vars
    else
  nofx    = size(xvar,2) 
    end
  
  nofq    = size(qvar,2)  # h
  nofw    = size(wvar,2)  # sigma_u_2
  nofv    = size(vvar,2)  # sigma_v_2
  nofgamma    = 1 # wy
  noftau    = 1 # wu
  nofrho    = 1 # wv

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

  beggamma = endv + 1
  endgamma = beggamma + nofgamma-1
  begtau = endgamma + 1
  endtau = begtau + noftau-1
  begrho = endtau + 1
  endrho = begrho + nofrho-1

  posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz,
            beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
         log_hscale = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1,
              gamma = beggamma + 1,
                tau = begtau + 1,
                rho = begrho + 1 )

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
          coeff_log_hscale = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv),
                   coeff_γ = (beggamma:endgamma),
                   coeff_τ = (begtau:endtau),
                   coeff_ρ = (begrho:endrho) )             

  #* retrieve variable names for making tables
  if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["s" * s for s in names(Wxvar)]   )
    else
  xnames  = names(xvar)
    end
    

  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  gammanames  = "γ"
  taunames  = "tau"
  rhonames  = "rho"

  varlist = vcat(" ", xnames, qnames, wnames, vnames, gammanames,taunames,rhonames)
 


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

  dulams = eigen(Wu[1])
  rumin = 1 / minimum(real(dulams.values))
  rumax = 1
  if length(Wu) > 1
      for k = 2:length(Wu)
          dulams = eigen(Wu[k])
          if rumin < 1 / minimum(real(dulams.values))
              rumin = 1 / minimum(real(dulams.values))
          end
      end
  end

  dvlams = eigen(Wv[1])
  rvmin = 1 / minimum(real(dvlams.values))
  rvmax = 1
  if length(Wv) > 1
      for k = 2:length(Wv)
          dvlams = eigen(Wv[k])
          if rvmin < 1 / minimum(real(dvlams.values))
              rvmin = 1 / minimum(real(dvlams.values))
          end
      end
  end

  eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)
    
  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx])) 

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, qvar, wvar, vvar, zvar, 
        envar, ivvar, eigvalu, indices_list, rowIDT, varlist
end









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
  nofx =  nofq = nofw = nofv = nofz = 0  # to make a complete list

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
  noftau    = 1 # wu
  nofrho    = 1 # wv

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
  beggamma = endz + 1
  endgamma = beggamma + nofgamma-1
  begtau = endgamma + 1
  endtau = begtau + noftau-1
  begrho = endtau + 1
  endrho = begrho + nofrho-1

  posvec = (begx=begx, endx=endx, begq=begq, endq=endq, begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
            beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
         log_hscale = begq + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1,
                  μ = begz + 1,
              gamma = beggamma + 1,
                tau = begtau + 1,
                rho = begrho + 1 )

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
          coeff_log_hscale = (begq:endq),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv),
                   coeff_μ = (begz:endz),
                   coeff_γ = (beggamma:endgamma),
                   coeff_τ = (begtau:endtau),
                   coeff_ρ = (begrho:endrho) )             

  #* retrieve variable names for making tables
  if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["s" * s for s in names(Wxvar)]   )
    else
  xnames  = names(xvar)
    end
    

  qnames  = names(qvar)
  wnames  = names(wvar)
  vnames  = names(vvar)
  znames  = names(zvar)
  gammanames  = "γ"
  taunames  = "tau"
  rhonames  = "rho"

  varlist = vcat(" ", xnames, qnames, wnames, vnames, znames, gammanames,taunames,rhonames)
 


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

  dulams = eigen(Wu[1])
  rumin = 1 / minimum(real(dulams.values))
  rumax = 1
  if length(Wu) > 1
      for k = 2:length(Wu)
          dulams = eigen(Wu[k])
          if rumin < 1 / minimum(real(dulams.values))
              rumin = 1 / minimum(real(dulams.values))
          end
      end
  end

  dvlams = eigen(Wv[1])
  rvmin = 1 / minimum(real(dvlams.values))
  rvmax = 1
  if length(Wv) > 1
      for k = 2:length(Wv)
          dvlams = eigen(Wv[k])
          if rvmin < 1 / minimum(real(dvlams.values))
              rvmin = 1 / minimum(real(dvlams.values))
          end
      end
  end



  eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)
  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx]))


  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, qvar, wvar, vvar, zvar, 
        envar, ivvar, eigvalu, indices_list, rowIDT, varlist


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
  # nofz    = size(zvar,2)  # mu
  nofgamma    = 1 # wy
  noftau    = 1 # wu
  nofrho    = 1 # wv

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
  beggamma = endv + 1
  endgamma = beggamma + nofgamma-1
  begtau = endgamma + 1
  endtau = begtau + noftau-1
  begrho = endtau + 1
  endrho = begrho + nofrho-1

  posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
            begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
            beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
         log_hscale = begq + 1,
                 iv = begphi + 1,
                  η = begeta + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1,
                  # μ = begz + 1,
              gamma = beggamma + 1,
                tau = begtau + 1,
                rho = begrho + 1 )

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
          coeff_log_hscale = (begq:endq),
                  coeff_iv = (begphi : endphi),
                  coeff_η  = (begeta : endeta),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv),
                   # coeff_μ = (begz:endz),
                   coeff_γ = (beggamma:endgamma),
                   coeff_τ = (begtau:endtau),
                   coeff_ρ = (begrho:endrho) )             

  #* retrieve variable names for making tables
  if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["s" * s for s in names(Wxvar)]   )
    else
  xnames  = names(xvar)
    end
    

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["eta" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  # znames  = names(zvar)
  gammanames  = "γ"
  taunames  = "tau"
  rhonames  = "rho"

  varlist = vcat(" ", xnames, qnames, ivnames, etanames, wnames, vnames, gammanames,taunames,rhonames)
 


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

  dulams = eigen(Wu[1])
  rumin = 1 / minimum(real(dulams.values))
  rumax = 1
  if length(Wu) > 1
      for k = 2:length(Wu)
          dulams = eigen(Wu[k])
          if rumin < 1 / minimum(real(dulams.values))
              rumin = 1 / minimum(real(dulams.values))
          end
      end
  end

  dvlams = eigen(Wv[1])
  rvmin = 1 / minimum(real(dvlams.values))
  rvmax = 1
  if length(Wv) > 1
      for k = 2:length(Wv)
          dvlams = eigen(Wv[k])
          if rvmin < 1 / minimum(real(dvlams.values))
              rvmin = 1 / minimum(real(dvlams.values))
          end
      end
  end

  eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)

  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx])) 

    
  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, qvar, wvar, vvar, zvar, 
    envar, ivvar, eigvalu, indices_list, rowIDT, varlist
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
  nofgamma    = 1 # wy
  noftau    = 1 # wu
  nofrho    = 1 # wv

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
  begz = endv + 1
  endz = begz + nofz-1
  beggamma = endz + 1
  endgamma = beggamma + nofgamma-1
  begtau = endgamma + 1
  endtau = begtau + noftau-1
  begrho = endtau + 1
  endrho = begrho + nofrho-1

  posvec = (begx=begx, endx=endx, begq=begq, endq=endq,begphi=begphi,endphi=endphi,begeta=begeta,endeta=endeta, 
            begw=begw, endw=endw,begv=begv, endv=endv, begz=begz, endz=endz, 
            beggamma=beggamma, endgamma=endgamma,begtau=begtau, endtau=endtau,begrho=begrho, endrho=endrho )

  #* create equation names and mark positions for making tables
  eqvec = (frontier = begx + 1, 
         log_hscale = begq + 1,                
                 iv = begphi + 1,
                  η = begeta + 1,
            log_σᵤ² = begw + 1,
            log_σᵥ² = begv + 1,
                  μ = begz + 1,
              gamma = beggamma + 1,
                tau = begtau + 1,
                rho = begrho + 1 )

  #* create equation names and mark positions 
  eqvec2 = (coeff_frontier = (begx:endx), 
          coeff_log_hscale = (begq:endq),
                  coeff_iv = (begphi : endphi),
                  coeff_η  = (begeta : endeta),
             coeff_log_σᵤ² = (begw:endw),
             coeff_log_σᵥ² = (begv:endv),
                   coeff_μ = (begz:endz),
                   coeff_γ = (beggamma:endgamma),
                   coeff_τ = (begtau:endtau),
                   coeff_ρ = (begrho:endrho) )             

  #* retrieve variable names for making tables
  if  Wx!=Nothing   # yuvx
  xnames  = vcat(names(xvar),   ["s" * s for s in names(Wxvar)]   )
    else
  xnames  = names(xvar)
    end
    

  qnames  = names(qvar)
  ivnames =["$(ai)_$(bi)" for ai in names(envar) for bi in names(ivvar)]   
  etanames = ["eta" * s for s in names(envar)] 
  wnames  = names(wvar)
  vnames  = names(vvar)
  znames  = names(zvar)
  gammanames  = "γ"
  taunames  = "tau"
  rhonames  = "rho"

  varlist = vcat(" ", xnames, qnames,  ivnames, etanames, wnames, vnames, znames, gammanames,taunames,rhonames)
 


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

  dulams = eigen(Wu[1])
  rumin = 1 / minimum(real(dulams.values))
  rumax = 1
  if length(Wu) > 1
      for k = 2:length(Wu)
          dulams = eigen(Wu[k])
          if rumin < 1 / minimum(real(dulams.values))
              rumin = 1 / minimum(real(dulams.values))
          end
      end
  end

  dvlams = eigen(Wv[1])
  rvmin = 1 / minimum(real(dvlams.values))
  rvmax = 1
  if length(Wv) > 1
      for k = 2:length(Wv)
          dvlams = eigen(Wv[k])
          if rvmin < 1 / minimum(real(dvlams.values))
              rvmin = 1 / minimum(real(dvlams.values))
          end
      end
  end



  eigvalu = (rymin=rymin, rymax=rymax, rumin=rumin, rumax=rumax, rvmin=rvmin, rvmax=rvmax)
    
  indices_list = find_all_indices_ordered(vcat(_dicM[:frontier],_dicM[:frontierWx])) 

  return modelinfo1, modelinfo2, posvec, nofvar, eqvec, eqvec2, yvar, xvar, qvar, wvar, vvar, zvar, 
        envar, ivvar, eigvalu, indices_list, rowIDT, varlist

end


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
