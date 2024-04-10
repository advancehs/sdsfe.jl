module sdsfe

export sfmodel_spec, sfmodel_init, sfmodel_opt, 
       sfmodel_fit, sfmodel_predict, drawTrun,
       genTrun, sf_demean, sfmodel_boot_marginal,
       # likelihood functions 
        LL_T, simple_check, simple_check2,
       # macros for _spec; 
        depvar, timevar, idvar, wy, wu, wv,
        sfdist, sftype,
        @sfdist, @sftype,
        sddirection,
        @sfdirection,
        @model, @depvar, 
        @frontier,@envar,@ivvar,
        @μ, @mu,  @σᵤ², @sigma_u_2,  @σᵥ², @sigma_v_2, 

        @hscale,  
        @sfpanel, 
        sfpanel,
        @timevar, @idvar,

        @eq,
       # functions for sfmodel_init 
        frontier, 
        μ, mu, σᵤ², sigma_u_2, σᵥ², sigma_v_2, 
  
        misc, 
        hscale,   
        all_init,
       # functions for sfmodel_opt
        warmstart_solver, warmstart_maxIT,
        main_solver, main_maxIT, tolerance, verbose, banner,
        ineff_index, marginal, table_format,
       # functions for sfmodel_fit
        useData,
        sfmodel_CI,
       # MoM Test
        sfmodel_MoMTest,
       # functions for JLMS and BC index
        jlmsbc, jlmsbc_marg,
       # the table for regular and mixed Chi-square test
        sfmodel_MixTable, sfmodel_ChiSquareTable,
       # struct 
        Sfmodeltype,

        Trun, truncated, trun, t, Half, half, h, s,
        MoM,
        production, cost, #* not prod, b/c conflict with Base
        text, html, latex,
        SSFOAT,SSFOAH, SSFOADT,SSFOADH, #* export for testing purpose
        SSF_OA2019,SSF_OAD2024,
      # Optim's algorithms  
        NelderMead, SimulatedAnnealing, SAMIN, ParticleSwarm,
        ConjugateGradient, GradientDescent, BFGS, LBFGS,
        Newton, NewtonTrustRegion, IPNewton, 
      # sfmodel_mtest()
        pickConsNameFromDF
 



using DataFrames
using DataStructures             # for OrderedDict
using Distributions              # for TDist, Normal
using FLoops                     # multithreading
using ForwardDiff                # for marginal effect
using HypothesisTests            # for pvalue()
using KahanSummation             # for time decay model, true random effect model
using LinearAlgebra              # extract diagnol and Matrix(I,...)
using NLSolversBase              # for hessian!
using Optim
using PrettyTables               # making tables 
using QuadGK                     # for TFE_CSW2014 model
using Random                     # for sfmodel_boot_marginal
using RowEchelon                 # for checkCollinear, check multi-collinearity
using SpecialFunctions           # for erfi used in sfmodel_MoMTest 
using StatsFuns                  # for normlogpdf(), normlogcdf()
using Statistics                 #
using CSV
using XLSX
using ForneyLab
using MAT
using LineSearches
using Calculus 


#############################
##   Define Model Types    ##
#############################


abstract type Sfmodeltype end
  struct Trun      <: Sfmodeltype end
  struct truncated <: Sfmodeltype end
  struct trun      <: Sfmodeltype end
  struct t         <: Sfmodeltype end 
  struct Half      <: Sfmodeltype end
  struct half      <: Sfmodeltype end
  struct h         <: Sfmodeltype end 
  struct s <: Sfmodeltype end  


  struct SSFOAT    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFOAH   <: Sfmodeltype end # panel true random effect model, truncated normal
  struct SSFOADT    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFOADH   <: Sfmodeltype end # panel true random effect model, truncated normal

abstract type PorC end
  struct production <: PorC end
  struct cost <: PorC end

abstract type PanelModel end
  struct SSF_OA2019 <: PanelModel end
  struct SSF_OAD2024 <: PanelModel end




  
abstract type TableFormat end
  struct text   <: TableFormat end
  struct html   <: TableFormat end
  struct latex  <: TableFormat end



################################################
##    include other files; order important    ##
################################################


include("sdsfemacfun.jl")
include("sdsfeloglikefun.jl")
include("sdsfeutil.jl")
# include("sdsfemtest.jl")
include("sdsfegetvars.jl")
include("sdsfeindex.jl")
# include("sdsfepredict.jl")
include("sdsfemarginal.jl")
include("sdsfemainfun.jl")





end # module sdsfe


