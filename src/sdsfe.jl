module sdsfe

export sfmodel_spec, sfmodel_init, sfmodel_opt, counterfact,
       sfmodel_fit, sfmodel_predict, drawTrun,
       genTrun, sf_demean, sfmodel_boot_marginal,
       sfmodel_henderson45, sfmodel_henderson45_y, henderson_45degree,
       # likelihood functions 
        LL_T, simple_check, simple_check2,
       # macros for _spec; 
        depvar, timevar, idvar, wy, wx, wu, wv,
        sfdist, sftype,
        @sfdist, @sftype,
        sddirection,
        @sfdirection,
        @model, @depvar, 
        @frontier,@envar,@ivvar,@frontierWx,
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
        ineff_index, mareffx,margeffu, table_format,autodiff_mode, cfindices,
       # functions for sfmodel_fit
        useData,
        sfmodel_CI,
       # MoM Test
        sfmodel_MoMTest,
       # functions for JLMS and BC index
        jlmsbc, jlmsbc_marg,
       # functions for partial loglikehood
        prtlloglike,
       # functions for counterfact 
        counterfactindex,
       # the table for regular and mixed Chi-square test
        sfmodel_MixTable, sfmodel_ChiSquareTable,
       # struct 
        Sfmodeltype,

        Trun, truncated, trun, t, Half, half, h, s,
        MoM,
        production, cost, #* not prod, b/c conflict with Base
        text, html, latex, forward, finite,
        SSFOAT,SSFOAH, SSFOADT,SSFOADH,
        SSFKUT,SSFKUH,SSFKUET,SSFKUEH, SSFKKT,SSFKKH,SSFKKET,SSFKKEH,        #* export for testing purpose
        SSF_OA2019,SSF_OAD2024,SSF_KU2020,SSF_KUE2020,SSF_KK2017,SSF_KKE2017,SSF_WH2010,SSF_WHE2010,
        SSFWHT,SSFWHH,SSFWHET,SSFWHEH,
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
using BlockDiagonals            # for diagnol matrix
using ADTypes
using Printf
using Plots

# Compatibility shim: ft_printf was removed in PrettyTables 3.x
function ft_printf(fmt_str::String, columns::AbstractVector{Int})
    fmt = Printf.Format(fmt_str)
    return (v, i, j) -> begin
        if j in columns && typeof(v) <: Number
            return Printf.format(fmt, v)
        else
            return v
        end
    end
end
ft_printf(fmt_str::String, columns::UnitRange{Int}) = ft_printf(fmt_str, collect(columns))

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
  struct SSFKUH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFKUT   <: Sfmodeltype end # panel true random effect model, truncated normal
  struct SSFKUEH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFKUET   <: Sfmodeltype end # panel true random effect model, truncated normal
  struct SSFKKH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFKKT   <: Sfmodeltype end # panel true random effect model, truncated normal
  struct SSFKKEH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFKKET   <: Sfmodeltype end # panel true random effect model, truncated normal

  struct SSFWHH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFWHT   <: Sfmodeltype end # panel true random effect model, truncated normal
  struct SSFWHEH    <: Sfmodeltype end # panel true random effect model, half normal
  struct SSFWHET   <: Sfmodeltype end # panel true random effect model, truncated normal

abstract type PorC end
  struct production <: PorC end
  struct cost <: PorC end

abstract type PanelModel end
  struct SSF_OA2019 <: PanelModel end
  struct SSF_OAD2024 <: PanelModel end
  struct SSF_KU2020 <: PanelModel end
  struct SSF_KUE2020 <: PanelModel end
  struct SSF_KK2017 <: PanelModel end
  struct SSF_KKE2017 <: PanelModel end
  struct SSF_WH2010 <: PanelModel end
  struct SSF_WHE2010 <: PanelModel end



  
abstract type TableFormat end
  struct text   <: TableFormat end
  struct html   <: TableFormat end
  struct latex  <: TableFormat end

  abstract type AutoDiffMode end
  struct forward   <: AutoDiffMode end
  struct finite   <: AutoDiffMode end

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
include("sdsfepartialloglike.jl")
include("sdsfecountf.jl")
include("sdsfehenderson.jl")





end # module sdsfe


