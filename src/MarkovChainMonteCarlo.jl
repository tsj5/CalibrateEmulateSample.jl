module MarkovChainMonteCarlo

using ..GaussianProcessEmulator
using ..ParameterDistributionStorage

using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions
using Random

using AbstractMCMC
using MCMCChains
import AdvancedMH

export 
    GPDensityModel,
    VariableStepProposal,
    VariableStepMHSampler,
    MCMC,
    get_stepsize,
    set_stepsize!,
    accept_ratio,
    get_posterior,
    find_mcmc_step!,
    sample_posterior!

# ------------------------------------------------------------------------------------------
# Define Gaussian process model

"""
    GPDensityModel

Factory which constructs `AdvancedMH.DensityModel` objects given a [`GaussianProcess`](@ref).
The role of the `DensityModel` is to return the log-likelihood of the data (here, as 
summarized by the Gaussian process) given input model parameters.
"""
function GPDensityModel(gp::GaussianProcess{FT}, obs_sample::Vector{FT}) where {FT <: AbstractFloat}
    # recall predict() written to return multiple `N_samples`: expects input to be a Matrix
    # with `N_samples` columns. Returned g is likewise a Matrix, and g_cov is a Vector of 
    # `N_samples` covariance matrices. For MH, N_samples is always 1, so we have to 
    # reshape()/re-cast input/output. 
    # transform_to_real = false means we work in the standardized space.
    return AdvancedMH.DensityModel(
        function (θ)
            g, g_cov = GaussianProcessEmulator.predict(gp, reshape(θ,:,1), transform_to_real=false)
            return logpdf(MvNormal(obs_sample, g_cov[1]), vec(g))
       end
    )
end

# ------------------------------------------------------------------------------------------
# Extend proposal distribution object to allow for tunable stepsize

"""
    VariableStepProposal

`AdvancedMH.Proposal` object responsible for generating the random walk steps used for new 
parameter proposals in the Metropolis-Hastings algorithm. Adds a separately adjustable 
`stepsize` parameter to the implementation from 
[AdvancedMH](https://github.com/TuringLang/AdvancedMH.jl).

This is allowed to be mutable, in order to dynamically change the `stepsize`. Since no
sampler state gets stored here, this choice is made out of convenience, not necessity.

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct VariableStepProposal{issymmetric, P, FT<:AbstractFloat} <: AdvancedMH.Proposal{P}
    "Distribution from which to draw IID random walk steps. Assumed zero-mean."
    proposal::P
    "Scaling factor applied to all samples drawn from `proposal`."
    stepsize::FT
end

# Boilerplate from AdvancedMH; 
# AdvancedMH extends Base.rand to draw from Proposal objects
const SymmetricVariableStepProposal{P, FT} = VariableStepProposal{true, P, FT}
VariableStepProposal(proposal, step) = VariableStepProposal{false}(proposal, step)
function VariableStepProposal{issymmetric}(proposal, step) where {issymmetric}
    return VariableStepProposal{issymmetric, typeof(proposal), typeof(step)}(proposal, step)
end

function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    proposal::VariableStepProposal{issymmetric, <:Union{Distribution,AbstractArray}, <:AbstractFloat},
    ::AbstractMCMC.AbstractModel
) where {issymmetric}
    return proposal.stepsize * rand(rng, proposal)
end

function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    proposal::VariableStepProposal{issymmetric, <:Union{Distribution,AbstractArray}, <:AbstractFloat},
    ::AbstractMCMC.AbstractModel,
    t
) where {issymmetric}
    return t + proposal.stepsize * rand(rng, proposal)
end

# q-factor used in density ratios: this *only* comes into play for samplers that don't obey
# detailed balance. For `MetropolisHastings` sampling this method isn't used.
function AdvancedMH.q(
    proposal::VariableStepProposal{issymmetric, <:Union{Distribution,AbstractArray}, <:AbstractFloat},
    t,
    t_cond
) where {issymmetric}
    return logpdf(proposal, (t - t_cond) / proposal.stepsize)
end

"""
    VariableStepMHSampler

Constructor for a Metropolis-Hastings sampler that uses the [`VariableStepProposal`](@ref)
Proposal object.

- `cov` - fixed covariance used to generate multivariate normal RW steps.
- `stepsize` - MH stepsize, applied as a constant uniform scaling to all samples. 
"""
function VariableStepMHSampler(cov::Matrix{FT}, stepsize::FT = 1.0) where {FT<:AbstractFloat}
    return AdvancedMH.MetropolisHastings(
        VariableStepProposal{false}(MvNormal(zeros(size(cov)[1]), cov), stepsize)
    )
end

# ------------------------------------------------------------------------------------------
# Record MH accept/reject in Transition object


# ------------------------------------------------------------------------------------------
# Top-level structure

"""
    standardize_obs

Logic for decorrelating observational inputs using SVD.
"""
function standardize_obs(
    obs_sample::Vector{FT},
    obs_noise_cov::Array{FT, 2};
    norm_factor::Union{Array{FT, 1}, Nothing} = nothing,
    svd = true,
    truncate_svd = 1.0
) where {FT}
    if norm_factor !== nothing
        obs_sample = obs_sample ./ norm_factor
        obs_noise_cov = obs_noise_cov ./ (norm_factor .* norm_factor)
    end
    # We need to transform obs_sample into the correct space 
    if svd
        println("Applying SVD to decorrelating outputs, if not required set svdflag=false")
        obs_sample, _ = svd_transform(obs_sample, obs_noise_cov; truncate_svd=truncate_svd)
    else
        println("Assuming independent outputs.")
    end
    return (obs_sample, obs_noise_cov)
end

"""
    MCMC

Top-level object to hold the `AdvancedMH.DensityModel` and Sampler objects, as well as 
arguments to be passed to the sampler.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct MCMC
    model::AbstractMCMC.AbstractModel
    sampler::AbstractMCMC.AbstractSampler
    sample_kwargs::NamedTuple
end

"""
    MCMC

Constructor for [`MCMC`](@ref) which performs obs standardization and takes keywords 
compatible with previous implementation.

- `obs_sample`: A single sample from the observations. Can, e.g., be picked from an Obs 
  struct using get_obs_sample.
- `obs_noise_cov`: Covariance of the observational noise.
- `prior`: array of length `N_parameters` containing the parameters' prior distributions.
- `step`: MCMC step size.
- `param_init`: Starting point for MCMC sampling.
- `max_iter`: Number of MCMC steps to take during sampling.
- `burnin`: Initial number of MCMC steps to discard (pre-convergence).
- `standardize`: Whether to use SVD to standardize observations.
- `norm_factor`: Optional factor by which to rescale observations.
- `svd`: Whether to use SVD to decorrelate observations.
- `truncate_svd`: Threshold for retaining singular values.
"""
function MCMC(
    obs_sample::Vector{FT},
    obs_noise_cov::Array{FT, 2},
    prior::ParameterDistribution;
    step::FT,
    param_init::Vector{FT},
    max_iter::IT,
    burnin::IT,
    standardize=false,
    norm_factor::Union{Array{FT, 1}, Nothing}=nothing,
    svd = true,
    truncate_svd=1.0
) where {FT<:AbstractFloat, IT<:Integer}
    if standardize
        obs_sample, obs_noise_cov = standardize_obs(
            obs_sample, obs_noise_cov;
            norm_factor=norm_factor, svd=svd, truncate_svd=truncate_svd
        )
    end
    model = GPDensityModel(gp, obs_sample)
    sampler = VariableStepMHSampler(get_cov(prior), step)
    return MCMC(
        model, sampler, (;
        :iter => max_iter,
        :init_params => deepcopy(param_init),
        :discard_initial => burnin,
        :chain_type => MCMCChains.Chains
    ))
end

"""
    get_stepsize

Returns the current MH `stepsize`, assuming we're using [`VariableStepProposal`](@ref) to
generate proposals. Throws an error for Samplers/Proposals without a `stepsize` field.
"""
function get_stepsize(mcmc::MCMC)
    if hasproperty(mcmc.sampler, :proposal)
        if hasproperty(mcmc.sampler.proposal, :stepsize)
            return mcmc.sampler.proposal.stepsize
        else
            throw("Proposal is of unrecognized type: $(typeof(mcmc.sampler.proposal)).")
        end
    else
        throw("Sampler is of unrecognized type: $(typeof(mcmc.sampler)).")
    end
end

"""
    set_stepsize!

Sets the MH `stepsize` to a new value, assuming we're using [`VariableStepProposal`](@ref) 
to generate proposals. Throws an error for Samplers/Proposals without a `stepsize` field.
"""
function set_stepsize!(mcmc::MCMC, new_step)
    if hasproperty(mcmc.sampler, :proposal)
        if hasproperty(samp.proposal, :stepsize)
            mcmc.sampler.proposal.stepsize = new_step
        else
            throw("Proposal is of unrecognized type: $(typeof(mcmc.sampler.proposal)).")
        end
    else
        throw("Sampler is of unrecognized type: $(typeof(mcmc.sampler)).")
    end
end

# ------------------------------------------------------------------------------------------
# still to be updated

function accept_ratio(mcmc::MCMC)
    return mcmc.accept[1] / mcmc.iter[1]
end

function find_mcmc_step!(mcmc_test::MCMC, gp::GaussianProcess{FT}; max_iter=2000) where {FT}
    step = mcmc_test.step[1]
    mcmc_accept = false
    doubled = false
    halved = false
    countmcmc = 0

    println("Begin step size search")
    println("iteration 0; current parameters ", mcmc_test.param')
    flush(stdout)
    it = 0
    local acc_ratio
    while mcmc_accept == false

        param = reshape(mcmc_test.param, :, 1)
        gp_pred, gp_predvar = predict(gp, param )
        if ndims(gp_predvar[1]) != 0
            mcmc_sample!(mcmc_test, vec(gp_pred), diag(gp_predvar[1]))
        else
            mcmc_sample!(mcmc_test, vec(gp_pred), vec(gp_predvar))
        end
        it += 1
        if it % max_iter == 0
            countmcmc += 1
            acc_ratio = accept_ratio(mcmc_test)
            println("iteration ", it, "; acceptance rate = ", acc_ratio,
                    ", current parameters ", param)
            flush(stdout)
            if countmcmc == 20
                println("failed to choose suitable stepsize in ", countmcmc,
                        "iterations")
                exit()
            end
            it = 0
            if doubled && halved
                step *= 0.75
                reset_with_step!(mcmc_test, step)
                doubled = false
                halved = false
            elseif acc_ratio < 0.15
                step *= 0.5
                reset_with_step!(mcmc_test, step)
                halved = true
            elseif acc_ratio>0.35
                step *= 2.0
                reset_with_step!(mcmc_test, step)
                doubled = true
            else
                mcmc_accept = true
            end
            if mcmc_accept == false
                println("new step size: ", step)
                flush(stdout)
            end
        end

    end

    return mcmc_test.step[1]
end

function get_posterior(mcmc::MCMC)
    #Return a parameter distributions object
    parameter_slices = batch(mcmc.prior)
    posterior_samples = [Samples(mcmc.posterior[slice,mcmc.burnin+1:end]) for slice in parameter_slices]
    flattened_constraints = get_all_constraints(mcmc.prior)
    parameter_constraints = [flattened_constraints[slice] for slice in parameter_slices] #live in same space as prior
    parameter_names = get_name(mcmc.prior) #the same parameters as in prior
    posterior_distribution = ParameterDistribution(posterior_samples, parameter_constraints, parameter_names)
    return posterior_distribution
    
end

function sample_posterior!(mcmc::MCMC,
                           gp::GaussianProcess{FT},
                           max_iter::IT) where {FT,IT<:Int}

    for mcmcit in 1:max_iter
        param = reshape(mcmc.param, :, 1)
        # test predictions (param is 1 x N_parameters)
        gp_pred, gp_predvar = predict(gp, param)

        if ndims(gp_predvar[1]) != 0
            mcmc_sample!(mcmc, vec(gp_pred), diag(gp_predvar[1]))
        else
            mcmc_sample!(mcmc, vec(gp_pred), vec(gp_predvar))
        end

    end
end

end # module MarkovChainMonteCarlo
