########################################################
####                sfmodel_spec()                  ####
########################################################

"""
sfmodel_spec(<keyword arguments>)

Provide specifications of the stochastic frontier model, including the type of model
and names of variables or matrix used in estimating the model. Two ways to
specify: Method 1: Use DataFrame as input, and Method 2: use matrix as input.

# Method 1 (DataFrame input)
Variables come from a DataFrame, and column names of the Dataframe are used in
the variable input. With this method,
equations are identified by macros but not functions (e.g., `@depvar()` but
not `depvar()`).

## Arguments of Method 1
- `sfdist(::Vararg)`: the distribution assumption of the one-sided stochastic
  variable (aka inefficiency term) of the model;
  possible choices include `truncated` (or `trun`, `t`), `half` (or `h`),
  `exponential` (or `expo`, `e`), and `trun_scale` (or `trun_scaling`, `ts`).
- `sftype(::Vararg)`: whether the model is a `production` (or `prod`) frontier
  or a `cost` frontier.
- `sfpanel(::Vararg)`: the type of panel model. Choices include `TFE_WH2010`
  (true fixed effect model of Wang and Ho 2010 JE), `TFE_CSW2014` (true fixed
  model of Chen, Schmidt, and Wang 2014 JE),  `TRE` (true random effect model
  of Greene 2004), `TimeDecay` (time decay model of Battese and Coelli 1992).
- `@depvar(::Vararg)`: the dependent variable from a DataFrame.
- `@frontier(::Vararg)`: a list of variables, separated by commas, in the frontier function.
- `@μ(::Vararg)` or `@mu(::Vararg)`: a list of variable, separated by comma,
  in the linear function of μ. (`sftype(trun)` only).
- `@σᵥ²(::Vararg)` or `@sigma_v_2(::Vararg)`: a list of variable, separated by comma, in the σᵥ²
  equation.
- `@σᵤ²(::Vararg)` or `@sigma_u_2(::Vararg)`: a list of variable, separated by comma, in the σᵤ²
  equation.
- `@σₐ²(::Vararg)` or `@sigma_a_2(::Vararg)`: a list of variable, separated by comma, in the σₐ²
  equation. `sfpanel(TRE)` only.
- `@gamma(::Vararg)`: a list of variables, separated by commas, in the gamma
  equation. `sfpanel(TimeDecay)` only.
- `@timevar(::Vararg)`: the variable containing the time period information.
  Panel data model only.
- `@idvar(::Vararg)`: the variable identifying each individual. Panel data
  model only.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_spec() is generated."
  on the screen after `sfmodel_spec()` is successfully executed.  


# Method 2 (matrix/vector input)
Data of the variables are
provided by individual matrices or vectors, and names of the mat/vec are used
in the equations. With this method, equations are identified by functions but
not macros (e.g., `depvar()` but not `@depvar()`). Note that if, for instance,
the name of `depvar` or `σᵤ²` has been used elsewhere in the program, using
these names to read in mat/vec will cause name conflict (`MethodError: objects
of type ... are not callable`). The workaround is to fully qualify the function
names, e.g., `SFrontiers.depvar`, `SFrontiers.σᵤ²`, etc. Or, use the alias (if
available), e.g., `sigma_u_2` instead of `σᵤ²`.

## Arguments of Method 2
- `sfdist(::Vararg)`: the distribution assumption on the inefficiency term;
  possible choices include `truncated` (or `trun`, `t`), `half` (or `h`),
  `exponential` (or `expo`, `e`), and `trun_scale` (or `trun_scaling`, `ts`).
- `sftype(::Vararg)`: whether the model is a `production` (or `prod`) frontier
  or a `cost` frontier.
- `sfpanel(::Vararg)`: the type of panel model. Choices include `TFE_WH2010`
  (true fixed effect model of Wang and Ho 2010 JE), `TFE_CSW2014` (true fixed
  model of Chen, Schmidt, and Wang 2014 JE),  `TRE` (true random effect model
  of Greene 2004), `TimeDecay` (time decay model of Battese and Coelli 1992).
- `depvar(::Matrix)`: Matrix or vector of the dependent variable.
- `frontier(::Matrix)`: matrix or vector for frontier function.
- `μ(::Matrix)` or `mu(::Matrix)`: matrix or vector for the (linear) μ equation (`trun` type only).
- `σᵤ²(::Matrix)` or `sigma_u_2(::Matrix)`: matrix or vector for the σᵤ² equation.
- `σₐ²(::Matrix)` or `sigma_a_2(::Matrix)`: matrix or vector for the σₐ² equation.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_spec() is generated."
  on the screen after `sfmodel_spec()` is successfully executed.

# Examples
```julia-repl
sfmodel_spec(sftype(prod), sfdist(trun),
             @depvar(output), 
             @frontier(land, , labor, bull, year, _cons), 
             @μ(age, school, year, _cons),
             @σᵤ²(age, school, year, _cons),
             @σᵥ²(_cons));

sfmodel_spec(sfpanel(TRE), sftype(prod), sfdist(half),
             @timevar(yr), @idvar(id),
             @depvar(y), 
             @frontier(x1, x2, _cons), 
             @σₐ²(_cons),
             @σᵤ²(_cons),
             @σᵥ²(_cons),
             message = false);
```
"""


function sfmodel_spec(arg::Vararg; message::Bool=false) 

  global _dicM 
         _dicM = Dict{Symbol, Any}()  # nullify and initiate new dictionaries when a new model is specified

# 在给定的 Julia 代码中，_dicM = Dict{Symbol, Any}() 这一行创建了一个空的字典。
# 在 Julia 中，Dict{Symbol, Any} 是一个字典类型的声明，其中 Symbol 表示键的类型，Any 表示值的类型，
# 这意味着这个字典可以接受任何类型的值。
# 因此，_dicM 是一个名为 _dicM 的变量，它被初始化为一个空的字典，可以用于存储任意类型的键值对。

       #* -- creates default values ---

      for k in (:panel, :timevar, :idvar,  :dist, :type, :depvar, :frontier,:frontierWx, :margeffu, :mareffx, :counterfact,
                  :μ, :hscale,  :σᵤ², :σᵥ²,  :hasDF, :transfer, :misc) # take out :data, :η, :λ, :τ, in this revision
          _dicM[k] = nothing
      end
   
      #* -- replace the defaults with user's values ---          

      for d in :($(arg))
          _dicM[d[1]] = d[2]
      end 

      (haskey(_dicM, :wy))    ||  (_dicM[:wy] = :($(wy()))[2])
      (haskey(_dicM, :wx))    ||  (_dicM[:wx] = :($(wx()))[2])
      (haskey(_dicM, :wu))    ||  (_dicM[:wu] = :($(wu()))[2])
      (haskey(_dicM, :wv))    ||  (_dicM[:wv] = :($(wv()))[2])



      #* ==== Method 2, matrix input (such as in simulations), create a DataFrame

         _dicM[:hasDF]    = true
         _dicM[:transfer] = false


      if typeof(_dicM[:depvar]) != Array{Symbol,1} # not a DataFrame

         _dicM[:hasDF] = false 

         isa(_dicM[:depvar][1], Vector) || isa(_dicM[:depvar][1], Matrix) || throw("
         `depvar()` has to be a Vector or Matrix (e.g., Array{Float64, 1} or Array{Float64, 2}). 
         Check with `isa(your_thing, Matrix)` or `isa(your_thing, Vector)`. 
         Try `convert()`, `reshape()`, `Matrix()`, or something similar.")

         comDF = _dicM[:depvar][1]  # create the first data column of comDF
         varname = [:depvar]
         _dicM[:depvar] = [:depvar]

         for k in (:timevar, :idvar, :frontier,:frontierWx, :margeffu, :mareffx, :counterfact, :μ, :hscale,  :σᵤ², :σᵥ²) 
             if _dicM[k] !== nothing # if not nothing, must be Array
                isa(_dicM[k], Vector) || isa(_dicM[k][1], Vector) || isa(_dicM[k][1], Matrix) || throw("
                   `k` has to be a Vector or Matrix (e.g., Array{Float64, 1} or Array{Float64, 2}). 
                   Check with `isa(your_thing, Matrix)` or `isa(your_thing, Vector)`. 
                   To convert, try `convert()`, `reshape()`, `Matrix()`, or something similar.")
 
                (isa(_dicM[k], Vector)  && length(_dicM[k][1]) == 1) ?  _dicM[k] = [_dicM[k]] : nothing # ugly fix for pure vector input
 
                @views comDF = hcat(comDF, _dicM[k][1]) # combine the data
                aa = Symbol[]
                for i in 1:size(_dicM[k][1], 2)
                    push!(aa, Symbol(String(k)*"_var$(i)")) # create name for the data
                end                  
                varname = vcat(varname, aa) # combine the dataname
                _dicM[k] = aa
             end 
         end # for k in (...)


         comDF = DataFrame(comDF, varname)
         _dicM[:sdf] = comDF

      end # if typeof(...)

   
      #* -- check the model identifier and the model type ---

      (_dicM[:dist] !== nothing) || throw("You need to specify dist().")
      (_dicM[:type] !== nothing) || throw("You need to specify type().")

      #* --- get the model identifier -------

      s = uppercase(String(_dicM[:dist][1])[1:1])
      # direc = uppercase(String(_dicM[:direction][1]))  ## 使用direction判断方向

      global tagD
      if _dicM[:panel] !== nothing  #  panel 
        

        if (_dicM[:panel] == [:SSF_OA2019]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFOAT}}()
          tagD[:modelid] = SSFOAT
        elseif (_dicM[:panel] == [:SSF_OA2019]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFOAH}}()
          tagD[:modelid] = SSFOAH 
        elseif (_dicM[:panel] == [:SSF_KU2020]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFKUT}}()
          tagD[:modelid] = SSFKUT
        elseif (_dicM[:panel] == [:SSF_KU2020]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFKUH}}()
          tagD[:modelid] = SSFKUH 
        elseif (_dicM[:panel] == [:SSF_KUE2020]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFKUET}}()
          tagD[:modelid] = SSFKUET
        elseif (_dicM[:panel] == [:SSF_KUE2020]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFKUEH}}()
          tagD[:modelid] = SSFKUEH 
        elseif (_dicM[:panel] == [:SSF_KK2017]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFKKT}}()
          tagD[:modelid] = SSFKKT
        elseif (_dicM[:panel] == [:SSF_KK2017]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFKKH}}()
          tagD[:modelid] = SSFKKH 
        elseif (_dicM[:panel] == [:SSF_KKE2017]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFKKET}}()
          tagD[:modelid] = SSFKKET
        elseif (_dicM[:panel] == [:SSF_KKE2017]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFKKEH}}()
          tagD[:modelid] = SSFKKEH 


        elseif (_dicM[:panel] == [:SSF_WH2010]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFWHT}}()
          tagD[:modelid] = SSFWHT
        elseif (_dicM[:panel] == [:SSF_WH2010]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFWHH}}()
          tagD[:modelid] = SSFWHH 
        elseif (_dicM[:panel] == [:SSF_WHE2010]) && (s=="T") 
          tagD = Dict{Symbol, Type{SSFWHET}}()
          tagD[:modelid] = SSFWHET
        elseif (_dicM[:panel] == [:SSF_WHE2010]) && (s=="H") 
          tagD = Dict{Symbol, Type{SSFWHEH}}()
          tagD[:modelid] = SSFWHEH 


        elseif (_dicM[:panel] == [:SSF_OAD2024]) && (s=="T")
          tagD = Dict{Symbol, Type{SSFOADT}}()
          tagD[:modelid] = SSFOADT
        elseif (_dicM[:panel] == [:SSF_OAD2024]) && (s=="H")
          tagD = Dict{Symbol, Type{SSFOADH}}()
          tagD[:modelid] = SSFOADH

        elseif (_dicM[:panel] == [:SSF_GI2025]) && (s=="T")
          tagD = Dict{Symbol, Type{SSFGIT}}()
          tagD[:modelid] = SSFGIT
        elseif (_dicM[:panel] == [:SSF_GI2025]) && (s=="H")
          tagD = Dict{Symbol, Type{SSFGIH}}()
          tagD[:modelid] = SSFGIH
        elseif (_dicM[:panel] == [:SSF_GIE2025]) && (s=="T")
          tagD = Dict{Symbol, Type{SSFGIET}}()
          tagD[:modelid] = SSFGIET
        elseif (_dicM[:panel] == [:SSF_GIE2025]) && (s=="H")
          tagD = Dict{Symbol, Type{SSFGIEH}}()
          tagD[:modelid] = SSFGIEH

        else
            throw("The `sfpanel()` and/or `sfdist()` are not specified correctly.")
        end
      end 
      #* ---- check if the model has the correct syntax ---

      # SFrontiers.checksyn(tagD[:modelid])!!!!!!!!!!!!!!!!!!!!!!!!

      #* ----- make return ----------- 
      # if message 
      #   printstyled("A dictionary from sfmodel_spec() is generated.\n"; color = :green)  
      # end  
      return _dicM # for debugging purpose

end  # end of sfmodel_spec()





########################################################
####                sfmodel_init()                  ####
########################################################
"""
    sfmodel_init(<keyword arguments>)

Provide initial values for the stochastic frontier model estimation. The
values could be a vector or scalars. It creates a global dictionary `_dicINI`. Optional.

# Arguments
- `all_init(::Union{Vector, Real})`: initial values of all the parameters in the model
- `frontier(::Union{Vector, Real})`: initial values of parameters in
  the `frontier()` function
- `μ(::Union{Vector, Real})` or `mu(::Union{Vector, Real})`: initial values of
  parameters in the `μ` function
- `hscale(::Union{Vector, Real})`: initial values of parameters in the `hscale()` function
- `gamma(::Union{Vector, Real})`: initial values of parameters in the `gamma()` function
- `σᵤ²(::Union{Vector, Real})` or `sigma_u_2(::Union{Vector, Real})`: initial values of parameters in the
   `σᵤ²` function
- `σᵥ²(::Union{Vector, Real})` or `sigma_v_2(::Union{Vector, Real})`: initial values of parameters in the
   `σᵥ²` function    
- `σₐ²(::Union{Vector, Real})` or `sigma_a_2(::Union{Vector, Real})`: initial
  values of parameters in the `σₐ²` function
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_init() is generated."
  on the screen after `sfmodel_init()` is successfully executed.

# Remarks
- Equations do not have to follow specific orders.
- `sfmodel_init(...)` is optional but is highly recommended. If it is not
  specified or is specified as an empty set, default values are used.
- It is not necessary to specify a complete set of equations. A partial list 
  or even empty lists are acceptable. Default values will be substituted for the
  missing equations.
- The generated `_dicINI` is inheritable in the sense that an exiting
  `_dicINI` (from the previous run of the same or a different model, for
  example) will be used if the current model does not have its own
  `sfmodel_init(...)`. This design has advantages in a simulations study where
  `sfmodel_init(...)` needs to be specified only once.

# Examples
```julia-repl
b_ini = ones(2)*0.2
sfmodel_init( # frontier(bb),             # may skip and use default
             μ(b_ini),                    # may use a vector
             σᵤ²(-0.1, -0.1),  
             σᵥ²(-0.1) )                   

sfmodel_init(all_init(0.1, 0.2, 0.5, 0.0, -0.1, -0.1, -0.1),
             message = false)             
```
"""
function sfmodel_init(arg::Vararg; message::Bool =false) # create a dictionary of inital vectors

   global _dicINI
          _dicINI = Dict{Symbol, Any}()

    for d in :($(arg))
        _dicINI[d[1]] = d[2]
    end        

    #* If has the key, creates the alias key with the same value.
    
    !(haskey(_dicINI, :μ))      || (_dicINI[:eqz] = _dicINI[:μ])
    !(haskey(_dicINI, :hscale)) || (_dicINI[:eqq] = _dicINI[:hscale]) 
    !(haskey(_dicINI, :σᵤ²))    || (_dicINI[:eqw] = _dicINI[:σᵤ²])
    !(haskey(_dicINI, :σᵥ²))    || (_dicINI[:eqv] = _dicINI[:σᵥ²])

    if message 
      printstyled("A dictionary from sfmodel_init() is generated.\n"; color = :green) 
    end
    return _dicINI # for debugging purpose
end    

########################################################
####                sfmodel_opt()                   ####
########################################################
"""
    sfmodel_opt(<keyword arguments>)

Provide options to the optimization algorithms for the maiximum likelihood
estimation. It creates a global dictionary `_dicOPT`. Optional. The `Optim`
package is used for the optimization, and a subset of
`Optim`'s keywords are directly accessible from this API. 

# Arguments
- `warmstart_solver(algorithm)`: The algorithm used in the first-stage ("warmstart")
  optimization process, which serves the purpose of improving upon the initial
  values for the second-stage ("main") estimation. The default is
  `NelderMead()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `Newton()`, `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.
  Non-gradient based algorithms are recommended for the warmstart solver. 
- `warmstart_maxIT(::Int64)`: The iteration limit for the warmstart. Default
  is 100.
- `main_solver(algorithm)`: The algorithm used in the main opimization process.
  The default is `Newton()`. Others include `SimulatedAnnealing()`, `SAMIN()`, `ParticleSwarm()`,
  `ConjugateGradient()`, `GradientDescent()`, `BFGS()`, `LBFGS()`,
  `NewtonTrustRegion()`, and `IPNewton()`. See
  http://julianlsolvers.github.io/Optim.jl/stable/ for details.
- `main_maxIT(::Int64)`: The iteration limit for the main estimation. Default
  is 2000.
- `tolerance(::Float64)`: The convergence criterion ("tolerance") based on the
  absolute value of gradients. Default is 1.0e-8. For non-gradient algorithms,
  it controls the main convergence tolerance, which is solver specific. 
  See `Optim`'s `g_tol` option for more information.
- `verbose(::Bool)`: Print on screen (`true`, the default) the information of
  the model and the optimization results.
- `banner(::Bool)`: Print on screen (`true`, the default) a banner to serve as
  a visual indicator of the start of the estimation.
- `ineff_index(::Bool)`: Whether to compute the Jondrow et al. (1982)
  inefficiency index and the Battese and Coelli (1988) efficiency index. The
  defauis `true`.
- `marginal(::Bool)`: Whether to compute the marginal effects of the exogenous
  determinants of inefficiency (if any).
- `table_format()`: The format to print the coefficient tables on the screen:
  `text` (default), `html`, or `latex`. A wrapper of `PrettyTables.jl`'s
  `backend` option.
- message::Bool: Whether printing (=true) or not (=false, the default) the
  confirmation message "A dictionary from sfmodel_opt() is generated."
  on the screen after `sfmodel_opt()` is successfully executed.


# Remarks
- `sfmodel_opt(...)` is optional. It can be omitted entirely, or specifying
  only a partial list of the keywords.
- If any of the keywords are missing, default values are used.
- If warmstart is not needed, you need to give empty keyword values to
  warmstart related keys. E.g., either `warmstart_solver()` or
  `warmstart_maxIT()`, or both. Omitting the keyword entirely (i.e., not
  writing down `warmstart_solver` or `warmstart_maxIT`) will not skip the
  warmstart, but will reinstate the default. 
- Users do not need to provide gradient or Hessian functions even if 
  gradient-based optimization algorithms are used. The package uses automatic
  differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation) to 
  compute the derivatives. It is not numerical finite differentiation. It is
  fast and as accurate as the symbolic differentiation.
- The `_dicOPT` is inheritable in the sense that an exiting `_dicOPT` (from
  the previous run of the same or a different model, for example) will be used
  if the current model does not have its own `sfmodel_opt(...)`. This design
  has advantages in simulation studies where `sfmodel_opt(...)` needs to be
  specified only once.

# Examples
```julia-repl
sfmodel_opt(warmstart_solver(NelderMead()),   
            warmstart_maxIT(200),
            main_solver(Newton()), 
            main_maxIT(2000), 
            tolerance(1e-8),
            message = false)
```
"""
function sfmodel_opt(arg::Vararg; message::Bool=false) # create a dictionary of maximization options

  global _dicOPT
         _dicOPT = Dict{Symbol, Any}()

  #* -- creates the default ---

  _dicOPT[:warmstart_solver] = :(NelderMead())
  _dicOPT[:warmstart_maxIT]  =  100
  _dicOPT[:main_solver]      = :(Newton())
  _dicOPT[:main_maxIT]       =  2000
  _dicOPT[:tolerance]        =  1.0e-8
  _dicOPT[:autodiff_mode]    = :(forward)
  _dicOPT[:verbose]          =  true
  _dicOPT[:banner]           =  true
  _dicOPT[:ineff_index]      =  true
  _dicOPT[:margeffu]         =  true
  _dicOPT[:counterfact]      =  true

  if tagD[:modelid] in (SSFOAT,SSFOAH,SSFOADT,SSFOADH,SSFKUEH,SSFKUET,SSFKUH,SSFKUT,SSFGIH,SSFGIT,SSFGIEH,SSFGIET)
    _dicOPT[:mareffx]        =  true
    _dicOPT[:margeffu]       =  true
  elseif tagD[:modelid] in (SSFKKEH,SSFKKET,SSFKKH,SSFKKT,SSFWHEH,SSFWHET,SSFWHH,SSFWHT)
    _dicOPT[:mareffx]        =  false
    _dicOPT[:margeffu]       =  false
  end


  _dicOPT[:table_format]     = :(text)

  #* -- replace the defaults with the user's value ---

  for d in :($(arg))
      _dicOPT[d[1]] = d[2]
  end    
  
  #* ---- error checking --

  if (_dicOPT[:main_solver] === nothing) || (_dicOPT[:main_maxIT] === nothing) || (_dicOPT[:tolerance] === nothing)
       throw("You cannot give empty keyword values to `main_solver()`, `main_maxIT()`, or `tolerance()`. If you want to use the default, you may do so by dropping (not emptying keyword values) the keywords.")
  end
  
  if message 
    printstyled("A dictionary from sfmodel_opt() is generated.\n"; color = :green)  
  end  
  return _dicOPT # for debugging purpose

end    # end of sfmodel_opt()







########################################################
###                 sfmodel_fit()                   ####
########################################################
"""
    sfmodel_fit(<keyword arguments>)

Maximum likelihood estimation of the stochastic frontier model specified 
in `sfmodel_spec(...)`. Estimate the model parameters, calculate Jondrow et al. 
(1982) inefficiency index and Battese and Coelli (1988) efficiency index, 
compute marginal effects of inefficiency determinants (if any).
Return a dictionary with results.

# Arguments
- `useData(::DataFrame)`: The DataFrame used with the Method 1 of
  `sfmodel_spec(...)`. If use Method 2 of `sfmodel_spec(...)` (viz., data
  is supplied by individual matrices), do not need this keyword argument.

# Remarks
- Use `Optim.jl` to carry out the estimation.
- Users do not need to provide gradient or Hessian functions even if 
  gradient-based optimization algorithms are used. The package uses automatic
  differentiation (https://en.wikipedia.org/wiki/Automatic_differentiation) to 
  compute the derivatives. AD is not numerical finite differentiation. AD is
  fast and as accurate as the symbolic differentiation.

# Examples
```julia-repl
sfmodel_fit(useData(df))    # Method 1
sfmodel_fit()               # Method 2
```
"""
function sfmodel_fit()
     # For Method 2 of `sfmodel_spec()`.
   
    !(_dicM[:hasDF]) || throw("Need to specify DataFrame in `sfmodel_fit()`.")
    
    _dicM[:transfer] = true
    sfmodel_fit(_dicM[:sdf])
   
end    

function sfmodel_fit(sfdat::DataFrame) #, D1::Dict = _dicM, D2::Dict = _dicINI, D3::Dict = _dicOPT)
    
  (_dicM[:hasDF] || _dicM[:transfer])  || throw("You provided matrix in `sfmodel_spec()` so you cannot specify a DataFrame in `sfmodel_fit()`. Leave it blank.")

  sfdat[!, :_consssssss] .= 1.0;
  Wy = _dicM[:wy];
  Wu = _dicM[:wu];
  Wv = _dicM[:wv];
  Wx = _dicM[:wx];
  if (Wy!=Nothing) | (Wu!=Nothing) | (Wv!=Nothing) | (Wx!=Nothing)
    sfdat = sort(sfdat,  [_dicM[:timevar][1], _dicM[:idvar][1]])
  else
    sfdat = sort(sfdat,  [_dicM[:idvar][1], _dicM[:timevar][1]])

  end

  # println(sfdat)
 #* for simulation, add a flag
 redflag::Bool = 0

#* ###### Check if the OPT dictionary exists #####

    @isdefined(_dicINI) || sfmodel_init()  # if not exist, create one with default values
    @isdefined(_dicOPT) || sfmodel_opt()  
    

  #  if _dicOPT[:banner] 
  #     printstyled("\n###------------------------------------###\n"; color=:yellow)
  #     printstyled("###  Estimating SF models using Julia  ###\n"; color=:yellow)
  #     printstyled("###------------------------------------###\n\n"; color=:yellow)
  #   end  


#* ##### Get variables from dataset #######
  
   # pos: (begx, endx, begz, endz, ...); variables' positions in the parameter vector.
   # num: (nofobs, nofx, ..., nofpara); number of variables in each equation
   # eqvec: ("frontier"=2, "μ"=6,...); named tuple of equation names and equation position in the table
   # eqvec2: (xeq=(1,3), zeq=(4,5),...); named tuple of equation and parameter positions, for sfmodel_predict
   # varlist: ("x1", "x2",...); variable names for making table

   (minfo1, minfo2, pos, num, eqvec, eqvec2, yvar, xvar,  qvar, wvar, vvar, zvar, envar, ivvar,
         eigvalu, indices_list, indices_listz, rowIDT, varlist) = getvar(tagD[:modelid], sfdat)
  
    # println(rowIDT)

#* ### print preliminary information ########

  if _dicOPT[:verbose] 

    printstyled("*********************************\n "; color=:cyan)
    printstyled("      Model Specification:\n"; color=:cyan); 
    printstyled("*********************************\n"; color=:cyan)

    print("Model type: "); printstyled(minfo1; color=:yellow); println();println()
    printstyled(minfo2; color=:yellow); println()
  end

#* ##### Get the type parameter #######

   _porc::Int64 = 1     

   if (_dicM[:type] == [:cost]) 
       _porc = -1
   end

#* ########## Process initial value dictionary  #####
   #* --- Get OLS results and other auxiliary values. --- #

  
    β0 = xvar \ yvar;  # OLS estiamte, uses a pivoted QR factorization;

    #* 根据空间权重矩阵的特征值计算初始值
    r0 = 0.3
    ry = (r0 - eigvalu.rymin) / (eigvalu.rymax - eigvalu.rymin)
    pgammaini = log(ry / (1.0 - ry))
    if Wu!=Nothing 
      ru = (r0 - eigvalu.rumin) / (eigvalu.rumax - eigvalu.rumin)
      ptauini = log(ru / (1.0 - ru))
    end
    if Wv!=Nothing 
      rv = (r0 - eigvalu.rvmin) / (eigvalu.rvmax - eigvalu.rvmin)
      prhoini = log(rv / (1.0 - rv))
    end
    #* --- Create the dictionary -----------

   if (:all_init in keys(_dicINI))
       sf_init = _dicINI[:all_init]
   else
       #*  Create ini vectors from user's values; if none, use the default.--- #      
       b_ini  = get(_dicINI, :frontier, β0)
       t_ini  = get(_dicINI, :eqq, ones(num.nofq) * 1.0)
       d2_ini = get(_dicINI, :eqw, log.(ones(num.nofw) * 0.1))
       g_ini  = get(_dicINI, :eqv, log.(ones(num.nofv) * 0.1))
       d1_ini = get(_dicINI, :eqz, ones(num.nofz) * 0.1)
       gammma_ini = get(_dicINI, :eqgamma, pgammaini)
       if Wu!=Nothing 
        tau_ini = get(_dicINI, :eqz, ptauini)
       end
       if Wv!=Nothing 
        rho_ini = get(_dicINI, :eqz, prhoini)
       end
       #*  Make it Array{Float64,1}; otherwise Array{Float64,2}. ---#     
       #*       Could also use sf_init[:,1]. *#
      if tagD[:modelid] == SSFOAH
          if Wy!=Nothing  # yuv
              if Wu!=Nothing 
                  if Wv!=Nothing #yuv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, tau_ini, rho_ini)  
                  else # yu
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, tau_ini)  
                  end    
              else 
                  if Wv!=Nothing #yv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, rho_ini)  
                  else #y
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini)  
                  end
              end
          else
              if Wu!=Nothing 
                  if Wv!=Nothing #uv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, tau_ini, rho_ini)  
                  else # u
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, tau_ini)  
                  end    
              else 
                  if Wv!=Nothing #v
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, rho_ini)  
                  else # 
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini)  
                  end
              end
          end            
      elseif tagD[:modelid] == SSFOAT
          if Wy!=Nothing  # yuv
              if Wu!=Nothing 
                  if Wv!=Nothing #yuv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, tau_ini, rho_ini)  
                  else # yu
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, tau_ini)  
                  end    
              else 
                  if Wv!=Nothing #yv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, rho_ini)  
                  else #y
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini)  
                  end
              end
          else
              if Wu!=Nothing 
                  if Wv!=Nothing #uv
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, tau_ini, rho_ini)  
                  else # u
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, tau_ini)  
                  end    
              else 
                  if Wv!=Nothing #v
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, rho_ini)  
                  else # 
                       sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini)  
                  end
              end
          end    
      elseif tagD[:modelid] == SSFKUEH
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini)  
          sf_init0 = vec(sf_init0)   
       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFKUH, sfdat)
          mfun0 = optimize(rho -> ssdkuh(yvar0, xvar0, qvar0,  Wy, _porc, num0, pos0, rho,  eigvalu0, rowIDT0 ) ,
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 1000, # different from search run
                                           store_trace = false,
                                           show_trace  = false))

          sf_init1  = Optim.minimizer(mfun0)

          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, gammma_ini1)  
          
      elseif tagD[:modelid] == SSFKUH
        if (haskey(_dicM, :envar)) 
          error("You can not specify @envar with model SSFKUH!")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini)  
      elseif tagD[:modelid] == SSFKUET
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini)  
          sf_init0 = vec(sf_init0)   

       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFKUT, sfdat)
          mfun0 = optimize(rho -> ssdkut(yvar0, xvar0, qvar0, Wy, _porc, num0, pos0, rho,  eigvalu0, rowIDT0 ) ,
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 1000, # different from search run
                                           store_trace = false,
                                           show_trace  = false))
          sf_init1  = Optim.minimizer(mfun0)
             
          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
          gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1)  

      elseif tagD[:modelid] == SSFKUT
        if (haskey(_dicM, :envar)) 
          error("You can not specify @envar with model SSFKUT!")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini, d1_ini, g_ini, gammma_ini)  

      elseif tagD[:modelid] == SSFKKEH
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, )  
          sf_init0 = vec(sf_init0)   
       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFKKH, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFKKH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600, # different from search run
                                           store_trace = false,
                                           show_trace  = false))

          sf_init1  = Optim.minimizer(mfun0)

          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1)  
          
      elseif tagD[:modelid] == SSFKKH
        if (haskey(_dicM, :envar)) 
          error("You can not specify @envar with model SSFKKH!")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini)  
      elseif tagD[:modelid] == SSFKKET
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini)  
          sf_init0 = vec(sf_init0)   

       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFKKT, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFKKT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600, # different from search run
                                           store_trace = false,
                                           show_trace  = false))
          sf_init1  = Optim.minimizer(mfun0)
             
          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1)  

      elseif tagD[:modelid] == SSFKKT
        if (haskey(_dicM, :envar)) 
          error("You can not specify @envar with model SSFKKT!")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini, d1_ini, g_ini)  


      elseif tagD[:modelid] == SSFWHEH
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, )  
          sf_init0 = vec(sf_init0)   
       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFWHH, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFWHH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600, # different from search run
                                           store_trace = false,
                                           show_trace  = false))

          sf_init1  = Optim.minimizer(mfun0)

          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1)  
          
      elseif tagD[:modelid] == SSFWHH
        if (haskey(_dicM, :envar)) 
          error("You can not specify @envar with model SSFWHH")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini,  g_ini)  
      elseif tagD[:modelid] == SSFWHET
          sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini)  
          sf_init0 = vec(sf_init0)   

       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFWHT, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFWHT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,         # initial values  
                                           NelderMead(),       # different from search run
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600, # different from search run
                                           store_trace = false,
                                           show_trace  = false))
          sf_init1  = Optim.minimizer(mfun0)
             
          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]

          sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1)  

      elseif tagD[:modelid] == SSFWHT
        if (haskey(_dicM, :envar))
          error("You can not specify @envar with model SSFWHT")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini, d1_ini, g_ini)

      # ============ Giannini 2025 (GI) 系列 ============
      elseif tagD[:modelid] == SSFGIEH
          sf_init0 = vcat(b_ini, t_ini, d2_ini, g_ini, gammma_ini)
          sf_init0 = vec(sf_init0)
       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFGIH, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFGIH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,
                                           NelderMead(),
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600,
                                           store_trace = false,
                                           show_trace  = false))
          sf_init1  = Optim.minimizer(mfun0)
          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1]
          sf_init = vcat(b_ini1, t_ini1, phi_ini, eta_ini, d2_ini1, g_ini1, gammma_ini1)

      elseif tagD[:modelid] == SSFGIH
        if (haskey(_dicM, :envar))
          error("You can not specify @envar with model SSFGIH")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini, g_ini, gammma_ini)

      elseif tagD[:modelid] == SSFGIET
          sf_init0 = vcat(b_ini, t_ini, d2_ini, g_ini, d1_ini, gammma_ini)
          sf_init0 = vec(sf_init0)
       (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
          eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFGIT, sfdat)
          mfun0 = optimize(rho -> LL_T(SSFGIT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                            _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                           sf_init0,
                                           NelderMead(),
                                           Optim.Options(g_tol = 1.0e-6,
                                           iterations  = 600,
                                           store_trace = false,
                                           show_trace  = false))
          sf_init1  = Optim.minimizer(mfun0)
          b_ini1 = sf_init1[1:num.nofx]
          t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
          phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
          eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
          d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
          g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
          d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
          gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
          sf_init = vcat(b_ini1, t_ini1, phi_ini, eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1)

      elseif tagD[:modelid] == SSFGIT
        if (haskey(_dicM, :envar))
          error("You can not specify @envar with model SSFGIT")
        end
          sf_init = vcat(b_ini, t_ini, d2_ini, g_ini, d1_ini, gammma_ini)

      elseif tagD[:modelid] == SSFOADH
          if Wy!=Nothing  # yuv
              if Wu!=Nothing 
                  if Wv!=Nothing #yuv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, tau_ini, rho_ini)  
                       sf_init0 = vec(sf_init0)   
                      
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0,  envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+3]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, gammma_ini1, tau_ini1, rho_ini1)  
  
                  else # yu
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, tau_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1  = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, gammma_ini1, tau_ini1)  
  
                  end    
              else 
                  if Wv!=Nothing #yv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini, rho_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, gammma_ini1, rho_ini1)  
                  else #y
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, gammma_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
  
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, gammma_ini1)  
                  end
              end
          else
              if Wu!=Nothing 
                  if Wv!=Nothing #uv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, tau_ini, rho_ini) 
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, tau_ini1, rho_ini1)  
                      
                  else # u
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, tau_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, tau_ini1)  
                  end    
              else 
                  if Wv!=Nothing #v
                      
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, rho_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                      
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, rho_ini1)  
                  else  # 
  
                     sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini)  
                       sf_init0 = vec(sf_init0)   
                          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAH, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAH, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1 )  
                  end
              end
          end  # Wy!=Nothing  # yuv

      elseif tagD[:modelid] == SSFOADT
             if Wy!=Nothing  # yuv
              if Wu!=Nothing 
                  if Wv!=Nothing #yuv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, tau_ini, rho_ini)  
                      
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat) 
  
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0,  envar0, ivvar0, 
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+3]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1, tau_ini1, rho_ini1)  
  
                  else # yu
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, tau_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1  = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1, tau_ini1)  
  
                  end    
              else 
                  if Wv!=Nothing #yv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini, rho_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1, rho_ini1)  
                  else #y
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, gammma_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       gammma_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
  
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, gammma_ini1)  
                  end
              end
          else
              if Wu!=Nothing 
                  if Wv!=Nothing #uv
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, tau_ini, rho_ini) 
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0, 
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+2]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, tau_ini1, rho_ini1)  
                      
                  else # u
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, tau_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
                         
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqeta, ones(num.nofphi)*0.1)  #  get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       tau_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, tau_ini1)  
                  end    
              else 
                  if Wv!=Nothing #v
                      
                       sf_init0 = vcat(b_ini, t_ini, d2_ini,  g_ini, d1_ini, rho_ini)  
                       sf_init0 = vec(sf_init0)   
          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                      
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                       rho_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+num.nofz+1]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1, rho_ini1)  
                  else  # 
  
                     sf_init0 = vcat(b_ini, t_ini, d2_ini, g_ini, d1_ini)  
                       sf_init0 = vec(sf_init0)   
                          
                    (minfo10, minfo20, pos0, num0, eqvec0, eqvec20, yvar0, xvar0,  qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                       eigvalu0, indices_list0, indices_listz0, rowIDT0, varlist0) = getvar(SSFOAT, sfdat)
                       mfun0 = optimize(rho -> LL_T(SSFOAT, yvar0, xvar0, qvar0, wvar0, vvar0, zvar0, envar0, ivvar0,
                                                              _porc, num0, pos0, rho, eigvalu0, rowIDT0, _dicM[:misc]),
                                                        sf_init0,         # initial values  
                                                        NelderMead(),       # different from search run
                                                        Optim.Options(g_tol = 1.0e-6,
                                                        iterations  = 100, # different from search run
                                                        store_trace = false,
                                                        show_trace  = false))
  
                       sf_init1  = Optim.minimizer(mfun0)
             
                       b_ini1 = sf_init1[1:num.nofx]
                       t_ini1  = sf_init1[num.nofx+1: num.nofx+num.nofq]
                       phi_ini = get(_dicINI, :eqphi, vec(ivvar \ envar))
                       eta_ini = get(_dicINI, :eqeta, ones(num.nofeta)*0.1)
                       d2_ini1 = sf_init1[num.nofx+num.nofq+1: num.nofx+num.nofq+num.nofw]
                       g_ini1  = sf_init1[num.nofx+num.nofq+num.nofw+1: num.nofx+num.nofq+num.nofw+num.nofv]
                       d1_ini1 = sf_init1[num.nofx+num.nofq+num.nofw+num.nofv+1: num.nofx+num.nofq+num.nofw+num.nofv+num.nofz]
                  
                       sf_init = vcat(b_ini1, t_ini1, phi_ini,eta_ini, d2_ini1, g_ini1, d1_ini1 )  
                  end
              end
          end  # Wy!=Nothing  # yuv
      end # tagD[:modelid] == SSFOAH
          
          sf_init = vec(sf_init)   
   end # if :all_init


  
#* ############ Misc.  ################     
   # --- check if the number of initial values is correct 
      (length(sf_init) == num.nofpara) ||  throw("The number of initial values does not match the number of parameters to be estimated. Make sure the number of init values in sfmodel_init() matches the number of variabls in sfmodel_spec().") 

   # --- Make sure there is no numerical issue arising from int vs. Float64.
      sf_init = convert(Array{Float64,1}, sf_init) 
  

#* ############# process optimization dictionary  #######

       if (_dicOPT[:warmstart_solver] === nothing) || (_dicOPT[:warmstart_maxIT] === nothing)
           do_warmstart_search = 0
       else 
           do_warmstart_search = 1
           sf_ini_algo  = eval(_dicOPT[:warmstart_solver])  # warmstart search algorithms
           sf_ini_maxit = _dicOPT[:warmstart_maxIT]         # warmstart search iter limit
       end    
           
   # ---- main maximization algorithm -----
       sf_algo  = eval(_dicOPT[:main_solver])    # main algorithm
       sf_maxit = _dicOPT[:main_maxIT] 
       sf_tol   = _dicOPT[:tolerance] 
       sf_table = _dicOPT[:table_format]
      # 兼容 PrettyTables 3.x: 将 Symbol 转换为 Val 类型
      if sf_table == :(text) || sf_table == :text
          sf_table = Val(:text)
      elseif sf_table == :(html) || sf_table == :html
          sf_table = Val(:html)
      elseif sf_table == :(latex) || sf_table == :latex
          sf_table = Val(:latex)
      end
       automode = _dicOPT[:autodiff_mode]
       # 兼容新版 Optim.jl: 将 Symbol 转换为 ADTypes 对象
       if automode == :forward || automode == :(forward)
           automode = ADTypes.AutoForwardDiff()
       elseif automode == :finite || automode == :(finite)
           automode = ADTypes.AutoFiniteDiff()
       end

#* ########  Start the Estimation  ##########

  #* ----- Define the problem's Hessian -----#

   _Hessian = TwiceDifferentiable(rho -> LL_T(tagD[:modelid], 
                         yvar, xvar, qvar, wvar, vvar, zvar, envar, ivvar,
                            _porc, num, pos, rho,
                              eigvalu, rowIDT, _dicM[:misc]),
                   sf_init;               
                  autodiff = automode) ;
  

  #* ---- Make placeholders for dictionary recording purposes *#

  sf_init_1st_dic  = 0
  sf_init_2nd_dic  = 0
  sf_ini_algo_dic  = nothing
  sf_ini_maxit_dic = 0
  sf_total_iter    = 0

  _run = 1  # a counter; use the -if- instead of -for- to avoid using global variables

  if (do_warmstart_search == 1) && (_run == 1)  

      if _dicOPT[:verbose] 
          printstyled("The warmstart run...\n\n"; color = :green)
      end

      sf_init_1st_dic  = copy(sf_init) # for dict recording
      sf_ini_algo_dic  = sf_ini_algo
      sf_ini_maxit_dic = copy(sf_ini_maxit)

      # @time  
             mfun = optimize(_Hessian,
                              sf_init,         # initial values  
                              sf_ini_algo,                   
                              Optim.Options(g_tol = sf_tol,
                                           iterations  = sf_ini_maxit, 
                                           store_trace = true,
                                           show_trace  = false))


      sf_total_iter += Optim.iterations(mfun) # for later use

      sf_init = Optim.minimizer(mfun)  # save as initials for the next run
      _run    = 2                      # modify the flag

      if _dicOPT[:verbose] 
          println()
          print("$mfun \n")
          print("The warmstart results are:\n"); printstyled(Optim.minimizer(mfun); color=:yellow); println("\n")
      end

 end  # if  (do_warmstart_search == 1) && (_run == 1)  

 if (do_warmstart_search == 0 ) || (_run == 2) # either no warmstart run and go straight here, or the 2nd run

     sf_init_2nd_dic = copy(sf_init) # for dict recording 

     if _dicOPT[:verbose] 
         println()
         printstyled("Starting the optimization run...\n\n" ; color = :green)
     end 
     
     # @time 
            mfun = optimize(_Hessian,
                             sf_init,         # initial values  
                            sf_algo,       # different from search run
                            Optim.Options(g_tol = sf_tol,
                                          iterations  = sf_maxit, # different from search run
                                          store_trace = true,
                                          show_trace  = false))
     sf_total_iter += Optim.iterations(mfun)

     if _dicOPT[:verbose] 
           println()
           print("$mfun \n")  
           print("The resulting coefficient vector is:\n"); printstyled(Optim.minimizer(mfun); color=:yellow); println("\n")
     end 


     if isnan(Optim.g_residual(mfun)) || (Optim.g_residual(mfun) >100) 
          redflag = 1
          printstyled("Note that the estimation may not have converged properly. The gradients are problematic (too large, > 0.1, or others).\n\n", color = :red)
     end 


     if Optim.iteration_limit_reached(mfun) 
           redflag = 1
           printstyled("Caution: The number of iterations reached the limit.\n\n"; color= :red)  
     end  

 end     # if (do_warmstart_search == 0 )....

#* ###### Post-estimation process ############### 
    _coevec            = Optim.minimizer(mfun)  # coef. vec.
  
    _coevec_adj = []
    for i in 1:length(eqvec2)
        # println(typeof(keys(eqvec2)[i] ))

        if keys(eqvec2)[i] ∉ [:coeff_γ, :coeff_τ, :coeff_ρ] 
        _coevec_adj = vcat(_coevec_adj, _coevec[eqvec2[i]])
        else
            if keys(eqvec2)[i] == :coeff_γ
                ss =  _coevec[eqvec2[i]][1]
                gamma_adj = eigvalu.rymin/(1+exp(ss))+eigvalu.rymax*exp(ss)/(1+exp(ss))
                _coevec_adj = vcat(_coevec_adj, gamma_adj)
            end
            if keys(eqvec2)[i] == :coeff_τ
                ss =  _coevec[eqvec2[i]][1]
                tau_adj = eigvalu.rumin/(1+exp(ss))+eigvalu.rumax*exp(ss)/(1+exp(ss))
                _coevec_adj = vcat(_coevec_adj, tau_adj)
            end
            if keys(eqvec2)[i] == :coeff_ρ
                ss =  _coevec[eqvec2[i]][1]
                rho_adj = eigvalu.rvmin/(1+exp(ss))+eigvalu.rvmax*exp(ss)/(1+exp(ss))
                _coevec_adj = vcat(_coevec_adj, rho_adj)         
            end
        end
    end
  
  # numerical_hessian = Calculus.hessian(rho -> LL_T(tagD[:modelid], 
  #                        yvar, xvar, qvar, wvar, vvar, zvar, envar, ivvar,
  #                         Wy, Wu, Wv, _porc, num, pos, rho,
  #                             eigvalu,   rowIDT, _dicM[:misc]), _coevec)
  # numerical_hessian = ForwardDiff.hessian(rho -> LL_T(tagD[:modelid], 
  #                        yvar, xvar, qvar, wvar, vvar, zvar, envar, ivvar,
  #                          _porc, num, pos, rho,
  #                             eigvalu,   rowIDT, _dicM[:misc]), _coevec)

  # numerical_hessian  = hessian!(_Hessian, _coevec)  # Hessain

    if automode == :forward
      numerical_hessian  = hessian!(_Hessian, _coevec)  # Hessain

      var_cov_matrix =  try
                          pinv(Symmetric(numerical_hessian))
                        catch err 
                          numerical_hessian = Calculus.hessian(rho -> LL_T(tagD[:modelid], 
                                              yvar, xvar, qvar, wvar, vvar, zvar, envar, ivvar,
                                                _porc, num, pos, rho,
                                                    eigvalu,   rowIDT, _dicM[:misc]), _coevec)

                          var_cov_matrix =  try
                                              pinv(Symmetric(numerical_hessian))
                                            catch err 
                                              redflag = 1
                                              # checkCollinear(tagD[:modelid], xvar,  qvar, wvar, vvar,zvar) # check if it is b/c of multi-collinearity in the data         
                                              throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
                                            end
                        end
    else

      numerical_hessian = Calculus.hessian(rho -> LL_T(tagD[:modelid], 
                            yvar, xvar, qvar, wvar, vvar, zvar, envar, ivvar,
                              _porc, num, pos, rho,
                                  eigvalu,   rowIDT, _dicM[:misc]), _coevec)
      var_cov_matrix =  try
                          pinv(Symmetric(numerical_hessian))
                        catch err 
                          redflag = 1
                          # checkCollinear(tagD[:modelid], xvar,  qvar, wvar, vvar,zvar) # check if it is b/c of multi-collinearity in the data         
                          throw("The Hessian matrix is not invertible, indicating the model does not converge properly. The estimation is abort.")
                        end
    end
    if !issymmetric(var_cov_matrix)
        # 如果矩阵不对称，则将其转换为对称矩阵
        var_cov_matrix = Symmetric(var_cov_matrix)
    end
  # println("############################################################")

  
  #* ------ Check if the matrix is invertible. ----


                    
        #* In some cases the matrix is invertible but the resulting diagonal
        #*    elements are negative. Check.

        # if !all( diag(var_cov_matrix) .> 0 ) # not all are positive
        #      redflag = 1
        #      printstyled("Some of the diagonal elements of the var-cov matrix are non-positive, indicating problems in the convergence. The estimation is abort.\n\n"; color = :red)
        #      # checkCollinear(tagD[:modelid], xvar, zvar, qvar, wvar, vvar) # check if it is b/c of multi-collinearity in the data
        # end              

   #* ------- JLMS and BC index -------------------

   if _dicOPT[:ineff_index] 
      @views _jlms,_jlms_direct,_jlms_indirect = jlmsbc(tagD[:modelid], yvar, xvar,  qvar, wvar, vvar,  zvar, envar, ivvar,
                                       _porc, num, pos, _coevec,  eigvalu, rowIDT)

      _jlmsM = mean.(eachcol(_jlms))
      _jlms_directM = mean.(eachcol(_jlms_direct))
      _jlms_indirectM = mean.(eachcol(_jlms_indirect))

      _te = exp.(-_jlms)
      _te_direct = exp.(-_jlms_direct)
      _te_indirect = exp.(-_jlms_indirect)

      _teM = mean.(eachcol(_te))
      _te_directM = mean.(eachcol(_te_direct))
      _te_indirectM = mean.(eachcol(_te_indirect))

   else
      _jlms, _jlms_direct, _jlms_indirect  = nothing, nothing, nothing
      _jlmsM, _jlms_directM, _jlms_indirectM = nothing, nothing, nothing
      _te, _te_direct, _te_indirect = nothing, nothing, nothing
      _teM,_te_directM,_te_indirectM = nothing, nothing, nothing

   end 


   #* ---- marginal effect on E(u) -------------- 
   Random.seed!(123)

   if _dicOPT[:margeffu]
      jlms0 = jlmsbc0(tagD[:modelid], yvar, xvar,  qvar, wvar, vvar,  zvar, envar, ivvar,
                              _porc, num, pos, _coevec,  eigvalu, rowIDT)

      totalematu, dirematu, indirematu = get_margeffu(jlms0, pos, _coevec,  var_cov_matrix,  eigvalu, indices_listz, rowIDT )
   else
      totalematu, dirematu, indirematu = nothing, nothing, nothing
   end      
  
   #* ---- marginal effect on x or wx --------------

   if _dicOPT[:mareffx]
      if Wy!=Nothing && _dicM[:wx]!=Nothing
      totalemat, diremat, indiremat = get_mareffx( pos, _coevec,  var_cov_matrix, eigvalu, indices_list, rowIDT )
      else
      totalemat, diremat, indiremat = nothing, nothing, nothing
      end
   else
      totalemat, diremat, indiremat = nothing, nothing, nothing
   end
   #* ---- Counterfactual analysis -------------- 
   if _dicOPT[:counterfact]  
   @views (_counterfacttotal, _counterfactdire,_counterfactindire) = counterfactindex(  tagD[:modelid], yvar, xvar, qvar, wvar, vvar,  zvar, envar, ivvar,
                                     _porc, num, pos, _coevec, eigvalu, rowIDT )
   end
  
   #* ------- Make Table ------------------
   for i in 1:size(var_cov_matrix, 1)
    if var_cov_matrix[i, i] < 0
        var_cov_matrix[i, i] = Inf
    end
    end
   stddev  = sqrt.(diag(var_cov_matrix)) # standard error

   stddev_adj = []
  for i in 1:length(eqvec2)
      # println(typeof(keys(eqvec2)[i] ))

      if keys(eqvec2)[i] ∉ [:coeff_γ, :coeff_τ, :coeff_ρ] 
      stddev_adj = vcat(stddev_adj, stddev[eqvec2[i]])
      else
          if keys(eqvec2)[i] == :coeff_γ
              ss =  stddev[eqvec2[i]][1]
              stddev_gamma_adj =abs(ss*(eigvalu.rymax-eigvalu.rymin)*exp(ss)/(1+exp(ss))^2  ) 
              stddev_adj = vcat(stddev_adj, stddev_gamma_adj)
          end
          if keys(eqvec2)[i] == :coeff_τ
              ss =  stddev[eqvec2[i]][1]
              stddev_tau_adj = abs(ss*(eigvalu.rumax-eigvalu.rumin)*exp(ss)/(1+exp(ss))^2  ) 
              stddev_adj = vcat(stddev_adj, stddev_tau_adj)
          end
          if keys(eqvec2)[i] == :coeff_ρ
              ss =  stddev[eqvec2[i]][1]
              stddev_rho_adj = abs(ss*(eigvalu.rvmax-eigvalu.rvmin)*exp(ss)/(1+exp(ss))^2  ) 
              stddev_adj = vcat(stddev_adj, stddev_rho_adj)         
          end
      end
  end
  
   t_stats = _coevec_adj ./ stddev_adj          # t statistics
   p_value = zeros(num.nofpara)   # p values
   ci_low  = zeros(num.nofpara) # confidence interval
   ci_upp  = zeros(num.nofpara) 
   tt      = cquantile(Normal(0,1), 0.025)

   for i = 1:num.nofpara 
       @views p_value[i,1] = pvalue(TDist(num.nofobs - num.nofpara), t_stats[i,1]; tail=:both)
       @views ci_low[i,1] = _coevec_adj[i,1] - tt*stddev_adj[i,1]
       @views ci_upp[i,1] = _coevec_adj[i,1] + tt*stddev_adj[i,1]
   end  

     #* Build the table columns *#

  table = zeros(num.nofpara, 7)  # 7 columns in the table
  table[:,2] = _coevec_adj   # estiamted coefficients
  table[:,3] = stddev_adj    # std deviation
  table[:,4] = t_stats   # t statistic
  table[:,5] = p_value   # p value
  table[:,6] = ci_low
  table[:,7] = ci_upp

  table      = [" " "Coef." "Std. Err." "z" "P>|z|" "95%CI_l" "95%CI_u"; table]  # add to top of the table

     #*  creating a column of function names 

  table[:, 1] .= ""
  for i in 1:length(eqvec)
      @views j = eqvec[i]
      @views table[j,1] = keys(eqvec)[i]
  end

     #*  Add the column of variable names

  table = hcat(varlist, table)                      # combine the variable names column (hcat, horizontal concatenate; see also vcat)
  table[:,1], table[:,2] = table[:,2], table[:,1]   # swap the first name column and the function column
  table[1,2] = "Var."

  stas = zeros(6, 7)
  stas = hcat([ "Median Efficiency"; "Num of obs." ;"Log of Likelihood" ; "AIC";  "BIC"; "HQC"], stas )
  stas[:,2:8] .= "";
  if _dicOPT[:ineff_index] 
    stas[1,3] = _teM[1]
  end
  
  stas[2,3] = num.nofobs
  if tagD[:modelid] in (SSFOADT,SSFOADH,SSFKUEH,SSFKUET,SSFKKEH,SSFKKET,SSFWHEH,SSFWHET,SSFGIEH,SSFGIET)
    llkkkk = -1* prtlloglike(tagD[:modelid], yvar, xvar,  qvar, wvar, vvar,  zvar, envar, ivvar, _porc, num, pos, _coevec,  eigvalu, rowIDT)
    stas[3,3] = llkkkk
    stas[4,3] = (-2)* (llkkkk)+2*(num.nofpara-num.nofphi-num.nofeta)
    stas[5,3] = (-2)* (llkkkk)+log(num.nofobs)*(num.nofpara-num.nofphi-num.nofeta)
    stas[6,3] = (-2)* (llkkkk)+log(log(num.nofobs))*2*(num.nofpara-num.nofphi-num.nofeta)

  else
    llkkkk = -Optim.minimum(mfun)
    stas[3,3] = -Optim.minimum(mfun)
    stas[4,3] = (-2)* (-Optim.minimum(mfun))+2*num.nofpara
    stas[5,3] = (-2)* (-Optim.minimum(mfun))+log(num.nofobs)*num.nofpara
    stas[6,3] = (-2)* (-Optim.minimum(mfun))+log(log(num.nofobs))*2*num.nofpara

  end
  table= vcat(table, stas)  

  if tagD[:modelid] in (SSFOADT,SSFOADH,SSFKUEH,SSFKUET,SSFKKEH,SSFKKET,SSFWHET,SSFWHEH,SSFGIEH,SSFGIET)

    row_indices = setdiff(1:size(table, 1), pos.begphi+1:pos.endphi+1)
    table_show = table[row_indices, :]
    # println(_coevec_adj,"ssssssssssss")

    coef_indices = setdiff(1:size(_coevec_adj, 1), pos.begphi:pos.endphi)

    _coevec_adj_show = _coevec_adj[coef_indices]

  else
    table_show = table
    _coevec_adj_show = _coevec_adj
  end

   # * ------ Print Results ----------- *#

   if _dicOPT[:verbose] 

       printstyled("*********************************\n "; color=:cyan)
       printstyled("      Estimation Results:\n"; color=:cyan); 
       printstyled("*********************************\n"; color=:cyan)

       print("Model type: "); printstyled(minfo1; color=:yellow); println()
       print("Number of observations: "); printstyled(num.nofobs; color=:yellow); println()
       print("Number of total iterations: "); printstyled(sf_total_iter; color=:yellow); println()
       if Optim.converged(mfun) 
           print("Converged successfully: "); printstyled(Optim.converged(mfun); color=:yellow); println()
       elseif Optim.converged(mfun) == false
           print("Converged successfully: "); printstyled(Optim.converged(mfun); color=:red); println()
           redflag = 1
       end         
       print("Log-likelihood value: "); printstyled(round(-1*Optim.minimum(mfun); digits=5); color=:yellow); println()
       println()
   
       pretty_table(table_show[2:end,:],
                    header=["", "Var.", "Coef.", "Std.Err.", "z", "P>|z|",
                            "95%CI_l", "95%CI_u"],
                    formatters = (ft_printf("%5.4f", 3:8),),
                    compact_printing = true,
                    backend = sf_table,
                    display_size = (-1, -1))
       println()


       # *----- Auxiliary Table, log parameters to original scales --------

       auxtable = Array{Any}(undef,2,3)
       rn = 0 # row index

       if size(wvar,2) == 1 # single variable
          varstd = sqrt(sum((wvar .- sum(wvar)/length(wvar)).^2)/length(wvar)) 
          if varstd  <= 1e-7 # constant, assuming =1 anyway
              rn += 1
              auxtable[rn, 1] = :σᵤ²
              auxtable[rn, 2] = exp(_coevec[pos.begw])
              auxtable[rn, 3] = exp(_coevec[pos.begw])*stddev[pos.begw]
          end
       end

       if size(vvar,2) == 1 # single variable
          varstd = sqrt(sum((vvar .- sum(vvar)/length(vvar)).^2)/length(vvar)) 
          if varstd  <= 1e-7 # constant
              rn += 1
              auxtable[rn, 1] = :σᵥ²
              auxtable[rn, 2] = exp(_coevec[pos.begv])
              auxtable[rn, 3] = exp(_coevec[pos.begv])*stddev[pos.begv]
          end
       end

       if rn >= 1  # table is non-empty
           println("Convert the constant log-parameter to its original scale, e.g., σ² = exp(log_σ²):")
           pretty_table(auxtable[1:rn,:],
                        header=["", "Coef.", "Std.Err."],
                        formatters = (ft_printf("%5.4f", 2:3),),
                        compact_printing = true,
                        backend = sf_table,
                        display_size = (-1, -1))

           print("\nTable format: "); printstyled("$(sf_table)"; color=:yellow); println(". Use sfmodel_opt() to choose between text, html, and latex.")
           println()
       end

       printstyled("***** Additional Information *********\n"; color=:cyan)



      #  if length(margMinfo) >= 1
      #     print("* The sample mean of inefficiency determinants' marginal effects on E(u): " ); printstyled(margMinfo; color=:yellow); println("")
      #     println("* Marginal effects of the inefficiency determinants at the observational level are saved in the return. See the follows.\n")
      #  end

       println("* Use `name.list` to see saved results (keys and values) where `name` is the return specified in `name = sfmodel_fit(..)`. Values may be retrieved using the keys. For instance:")
       println("   ** `name.loglikelihood`: the log-likelihood value of the model;")
       println("   ** `name.jlms`: Jondrow et al. (1982) inefficiency index;")
       println("   ** `name.bc`: Battese and Coelli (1988) efficiency index;")
       println("   ** `name.marginal`: a DataFrame with variables' (if any) marginal effects on E(u).")
       println("* Use `keys(name)` to see available keys.")

       printstyled("**************************************\n\n\n"; color=:cyan)

   end  # if_verbose

#* ########### create a dictionary and make a tuple for return ########### *#
   
    _dicRES = OrderedDict{Symbol, Any}()     
    _dicRES[:converged]          = Optim.converged(mfun)
    _dicRES[:iter_limit_reached] = Optim.iteration_limit_reached(mfun)
    _dicRES[:_______________] = "___________________"  #33
    _dicRES[:n_observations]  = num.nofobs
    _dicRES[:loglikelihood]   = llkkkk
    _dicRES[:table]           = [table][1]
    _dicRES[:table_show]      = [table_show][1]
    _dicRES[:coeff]           = _coevec_adj
    _dicRES[:coeff_show]      = _coevec_adj_show
    _dicRES[:std_err]         = stddev_adj
    _dicRES[:var_cov_mat]     = [var_cov_matrix][1]
    _dicRES[:jlms]            = _jlms
    _dicRES[:jlms_direct]     = _jlms_direct
    _dicRES[:jlms_indirect]   = _jlms_indirect

    _dicRES[:te]              = _te
    _dicRES[:te_direct]       = _te_direct
    _dicRES[:te_indirect]     = _te_indirect

    _dicRES[:totalematu]      = totalematu
    _dicRES[:dirematu]        = dirematu
    _dicRES[:indirematu]      = indirematu

    _dicRES[:totalemat]        = totalemat
    _dicRES[:diremat]         = diremat
    _dicRES[:indiremat]       = indiremat


    if _dicOPT[:counterfact]  

      _dicRES[:counterfacttotal] =_counterfacttotal
      _dicRES[:counterfactdire]  =_counterfactdire
      _dicRES[:counterfactindire] =_counterfactindire
    end


    
    _dicRES[:_____________] = "___________________"  #31      
    _dicRES[:model]         = minfo1      
     #     _dicRES[:data]          = "$sfdat"
    _dicRES[:depvar]        = _dicM[:depvar]
    _dicRES[:frontier]      = _dicM[:frontier]
    _dicRES[:μ]             = _dicM[:μ]
    _dicRES[:hscale]        = _dicM[:hscale]        
    _dicRES[:σᵤ²]           = _dicM[:σᵤ²]
    _dicRES[:σᵥ²]           = _dicM[:σᵥ²]
    _dicRES[:log_σᵤ²]       = _dicM[:σᵤ²] 
    _dicRES[:log_σᵥ²]       = _dicM[:σᵥ²]
    _dicRES[:type]          = _dicM[:type]
    _dicRES[:dist]          = _dicM[:dist]
    _dicRES[:PorC]          = _porc
    _dicRES[:timevar]       = _dicM[:timevar]  
    _dicRES[:idvar]         = _dicM[:idvar] # for bootstrap marginal effect
    _dicRES[:table_format]  = _dicOPT[:table_format]
    _dicRES[:autodiff_mode]  = _dicOPT[:autodiff_mode]
    _dicRES[:modelid]       = tagD[:modelid]
    _dicRES[:verbose]       = _dicOPT[:verbose]
    _dicRES[:hasDF]         = _dicM[:hasDF]
    _dicRES[:transfer]      = _dicM[:transfer]

  for i in 1:length(eqvec2)
      _dicRES[keys(eqvec2)[i]] = _coevec_adj[eqvec2[i]]
  end

    _dicRES[:________________]  = "___________________" #34
    _dicRES[:Hessian]           = [numerical_hessian][1]
    _dicRES[:gradient_norm]     = Optim.g_residual(mfun)
  # _dicRES[:trace]             = Optim.trace(mfun)     # comment out because not very informative and size could be large
    _dicRES[:actual_iterations] = Optim.iterations(mfun)
    _dicRES[:______________] = "______________________" #32
    _dicRES[:warmstart_solver] = sf_ini_algo_dic
    _dicRES[:warmstart_ini]    = sf_init_1st_dic
    _dicRES[:warmstart_maxIT]  = sf_ini_maxit_dic
    _dicRES[:main_solver]      = sf_algo
    _dicRES[:main_ini]         = sf_init_2nd_dic
    _dicRES[:main_maxIT]       = sf_maxit
    _dicRES[:tolerance]        = sf_tol
    _dicRES[:eqpo]             = eqvec2
    _dicRES[:eigvalu]          = eigvalu

    _dicRES[:redflag]          = redflag

   #* ----- Delete optional keys that have value nothing, 

       for k in (:μ, :hscale) 
           if _dicRES[k] === nothing 
              delete!(_dicRES, k)
           end 
       end

   #* ----- Create a NamedTuple from the dic as the final output; 
   #* -----     put the dic in the tuple.

       _ntRES = NamedTuple{Tuple(keys(_dicRES))}(values(_dicRES))
       _ntRES = (; _ntRES..., list    = _dicRES)

   #* ---- Create a gloal dictionary for sf_predict ---- 

      _eqncoe = Dict{Symbol, Vector}()  # nullify and initiate new dictionaries when a new model is specified

      for i in 1:length(eqvec2)
          _eqncoe[keys(eqvec)[i]]  = _coevec_adj[eqvec2[i]] # for sf_predict
      end

#* ############  make returns  ############ *#

    return _ntRES
  # return margeff, margMinfo
end # sfmodel_fit

