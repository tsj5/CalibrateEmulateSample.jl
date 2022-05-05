# refactor-Sample
module MarkovChainMonteCarlo

using ..Emulators
using ..ParameterDistributions

import Distributions: sample # Reexport sample()
using Distributions
using DocStringExtensions
using LinearAlgebra
using Printf
using Random
using Statistics

using MCMCChains
import AbstractMCMC: sample # Reexport sample()
using AbstractMCMC
import AdvancedMH

export
    EmulatorPosteriorModel,
    PriorProposalMHSampler,
    MCMCProtocol,
    EmulatorRWSampling,
    MCMCWrapper,
    accept_ratio,
    optimize_stepsize,
    get_posterior,
    sample

# ------------------------------------------------------------------------------------------
# Output space transformations between original and SVD-decorrelated coordinates.
# Redundant with what's in Emulators.jl, but need to reimplement since we don't have
# access to obs_noise_cov

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transform samples from the original (correlated) coordinate system to the SVD-decorrelated
coordinate system used by Emulator.
"""
function to_decorrelated(data::AbstractMatrix{FT}, em::Emulator{FT}) where {FT<:AbstractFloat}
    if em.standardize_outputs && em.standardize_outputs_factors !== nothing 
        # standardize() data by scale factors, if they were given
        data = data ./ em.standardize_outputs_factors
    end
    decomp = em.decomposition
    if decomp !== nothing
        # Use SVD decomposition of obs noise cov, if given, to transform data to 
        # decorrelated coordinates.
        inv_sqrt_singvals = Diagonal(1.0 ./ sqrt.(decomp.S)) 
        return inv_sqrt_singvals * decomp.Vt * data
    else
        return data
    end
end
function to_decorrelated(data::AbstractVector{FT}, em::Emulator{FT}) where {FT<:AbstractFloat}
    # method for single sample
    out_data = to_decorrelated(reshape(data, :, 1), em)
    return vec(out_data)
end

# ------------------------------------------------------------------------------------------
# Use emulated model in sampler

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factory which constructs `AdvancedMH.DensityModel` objects given a set of prior 
distributions on the model parameters and an [`Emulator`](@ref), which encodes the 
log-likelihood of the data given parameters. Together this yields the log density we're 
attempting to sample from with the MCMC, which is the role of the `DensityModel` class in 
the `AbstractMCMC` interface.
"""
function EmulatorPosteriorModel(
    prior::ParameterDistribution,
    em::Emulator{FT}, 
    obs_sample::AbstractVector{FT}
) where {FT <: AbstractFloat}
    # recall predict() written to return multiple `N_samples`: expects input to be a Matrix
    # with `N_samples` columns. Returned g is likewise a Matrix, and g_cov is a Vector of 
    # `N_samples` covariance matrices. For MH, N_samples is always 1, so we have to 
    # reshape()/re-cast input/output; simpler to do here than add a predict() method.
    return AdvancedMH.DensityModel(
        function (θ) 
            # θ: model params we evaluate at; in original coords.
            # transform_to_real = false means g, g_cov, obs_sample are in decorrelated coords.
            g, g_cov = Emulators.predict(em, reshape(θ,:,1), transform_to_real=false)
            return logpdf(MvNormal(obs_sample, g_cov[1]), vec(g)) + get_logpdf(prior, θ)
       end
    )
end

# ------------------------------------------------------------------------------------------
# Record MH accept/reject decision in MCMCState object

"""
$(DocStringExtensions.TYPEDEF)

Extend the basic `AdvancedMH.Transition` (which encodes the current state of the MC during
sampling) with a boolean flag to record whether this state is new (arising from accepting a
MH proposal) or old (from rejecting a proposal).

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct MCMCState{T, L<:Real} <: AdvancedMH.AbstractTransition
    "Sampled value of the parameters at the current state of the MCMC chain."
    params :: T
    "Log probability of `params`, as computed by the model using the prior."
    log_density :: L
    "Whether this state resulted from accepting a new MC proposal."
    accepted :: Bool
end

# Boilerplate from AdvancedMH:
# Store the new draw and its log density.
MCMCState(model::AdvancedMH.DensityModel, params, accepted=true) =
    MCMCState(params, logdensity(model, params), accepted)

# Calculate the log density of the model given some parameterization.
AdvancedMH.logdensity(model::AdvancedMH.DensityModel, t::MCMCState) = t.log_density

# AdvancedMH.transition() is only called to create a new proposal, so create a MCMCState
# with accepted = true since that object will only be used if proposal is accepted.
function AdvancedMH.transition(
    sampler::AdvancedMH.MHSampler, 
    model::AdvancedMH.DensityModel, 
    params, 
    log_density::Real
)
    return MCMCState(params, log_density, true)
end

# method extending AdvancedMH.propose() to variable/explicitly given stepsize
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::AdvancedMH.MHSampler,
    model::AdvancedMH.DensityModel,
    current_state::MCMCState;
    stepsize::FT = 1.0
) where {FT<:AbstractFloat}
    return current_state.params + stepsize * rand(rng, sampler.proposal)
end

# Copy a MCMCState and set accepted = false
reject_transition(t::MCMCState) = MCMCState(t.params, t.log_density, false)

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.MHSampler,
    current_state::MCMCState;
    stepsize::FT = 1.0,
    kwargs...
) where {FT<:AbstractFloat}
    # Generate a new proposal.
    new_params = AdvancedMH.propose(rng, sampler, model, current_state; stepsize = stepsize)

    # Calculate the log acceptance probability and the log density of the candidate.
    new_log_density = AdvancedMH.logdensity(model, new_params)
    log_α = new_log_density - AdvancedMH.logdensity(model, current_state) +
        AdvancedMH.logratio_proposal_density(sampler, current_state, new_params)

    # Decide whether to return the previous params or the new one.
    new_state = if -Random.randexp(rng) < log_α
        # accept
        AdvancedMH.transition(sampler, model, new_params, new_log_density)
    else
        # reject
        reject_transition(current_state)
    end
    # Return a 2-tuple consisting of the next sample and the the next state.
    # In this case (MH obeying detailed balance) they are identical.
    return new_state, new_state
end

# ------------------------------------------------------------------------------------------
# Extend the record-keeping methods defined in AdvancedMH to include the 
# MCMCState.accepted field added above.

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:MCMCState},
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.MHSampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial=0,
    thinning=1,
    param_names=missing,
    kwargs...
)
    # Turn all the transitions into a vector-of-vectors.
    vals = [vcat(t.params, t.log_density, t.accepted) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_density, :accepted]

    # Bundle everything up and return a MCChains.Chains struct.
    return MCMCChains.Chains(
        vals, vcat(param_names, internal_names), 
        (parameters = param_names, internals = internal_names,);
        start=discard_initial + 1, thin=thinning,
    )
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Vector{<:MCMCState}},
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.Ensemble,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial=0,
    thinning=1,
    param_names=missing,
    kwargs...
)
    # Preallocate return array
    # NOTE: requires constant dimensionality.
    n_params = length(ts[1][1].params)
    nsamples = length(ts)
    # add 2 parameters for :log_density, :accepted
    vals = Array{Float64, 3}(undef, nsamples, n_params + 2, sampler.n_walkers)

    for n in 1:nsamples
        for i in 1:sampler.n_walkers
            walker = ts[n][i]
            for j in 1:n_params
                vals[n, j, i] = walker.params[j]
            end
            vals[n, n_params + 1, i] = walker.log_density
            vals[n, n_params + 2, i] = walker.accepted
        end
    end

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1][1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_density, :accepted]

    # Bundle everything up and return a MCChains.Chains struct.
    return MCMCChains.Chains(
        vals, vcat(param_names, internal_names), 
        (parameters = param_names, internals = internal_names);
        start=discard_initial + 1, thin=thinning,
    )
end

# ------------------------------------------------------------------------------------------
# Top-level object to contain model and sampler (but not state)

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructor for a Metropolis-Hastings sampler that generates proposals for new parameters 
based on the covariance of the `prior` object.
"""
function PriorProposalMHSampler(prior::ParameterDistribution)
    Σ = ParameterDistributions.cov(prior)
    return AdvancedMH.MetropolisHastings(
        AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(Σ)[1]), Σ))
    )
end

"""
$(DocStringExtensions.TYPEDEF)

Type used to dispatch different methods of the [`MCMCWrapper`](@ref) constructor, 
corresponding to different choices of DensityModel and Sampler.
"""
abstract type MCMCProtocol end

"""
$(DocStringExtensions.TYPEDEF)
    
[`MCMCProtocol`](@ref) which uses [`EmulatorPosteriorModel`](@ref) for the DensityModel (here, 
emulated likelihood \\* prior) and [`PriorProposalMHSampler`](@ref) for the sampler 
(generator of proposals for Metropolis-Hastings).
"""
struct EmulatorRWSampling <: MCMCProtocol end

"""
$(DocStringExtensions.TYPEDEF)

Top-level object to hold the prior, DensityModel and Sampler objects, as well as 
arguments to be passed to the sampling function.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct MCMCWrapper
    "`EnsembleKalmanProcess.ParameterDistribution` object describing the prior distribution on parameter values."
    prior::ParameterDistribution
    "`AdvancedMH.DensityModel` object, used to evaluate the posterior density being sampled from."
    log_posterior_map::AbstractMCMC.AbstractModel
    "Object describing a MCMC sampling algorithm and its settings."
    mh_proposal_sampler::AbstractMCMC.AbstractSampler
    "NamedTuple of other arguments to be passed to `AbstractMCMC.sample()`."
    sample_kwargs::NamedTuple
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructor for [`MCMCWrapper`](@ref) which performs the same standardization (SVD 
decorrelation) that was applied in the Emulator. It creates and wraps an instance of 
[`EmulatorPosteriorModel`](@ref), for sampling from the Emulator, and 
[`PriorProposalMHSampler`](@ref), for generating the MC proposals.

- `obs_sample`: A single sample from the observations. Can, e.g., be picked from an Obs 
  struct using get_obs_sample.
- `prior`: array of length `N_parameters` containing the parameters' prior distributions.
- `em`: [`Emulator`](@ref) to sample from. 
- `stepsize`: MCMC step size, applied as a scaling to the prior covariance.
- `init_params`: Starting point for MCMC sampling.
- `burnin`: Initial number of MCMC steps to discard (pre-convergence).
"""
function MCMCWrapper(
    ::EmulatorRWSampling,
    obs_sample::AbstractVector{FT},
    prior::ParameterDistribution,
    em::Emulator;
    init_params::AbstractVector{FT},
    burnin::IT=0,
    kwargs...
) where {FT<:AbstractFloat, IT<:Integer}
    obs_sample = to_decorrelated(obs_sample, em)
    log_posterior_map = EmulatorPosteriorModel(prior, em, obs_sample)
    mh_proposal_sampler = PriorProposalMHSampler(prior)
    sample_kwargs = (; # set defaults here
        :init_params => deepcopy(init_params),
        :param_names => get_name(prior),
        :discard_initial => burnin,
        :chain_type => MCMCChains.Chains
    )
    sample_kwargs = merge(sample_kwargs, kwargs) # override defaults with any explicit values
    return MCMCWrapper(prior, log_posterior_map, mh_proposal_sampler, sample_kwargs)
end

# Define new methods extending AbstractMCMC.sample() using the MCMCWrapper object.

# All cases where rng given
function sample(rng::Random.AbstractRNG, mcmc::MCMCWrapper, args...; kwargs...)
    # any explicit function kwargs override defaults in mcmc object
    kwargs = merge(mcmc.sample_kwargs, NamedTuple(kwargs))
    return AbstractMCMC.mcmcsample(
        rng, mcmc.log_posterior_map, mcmc.mh_proposal_sampler, args...; kwargs...
    )
end
# use default rng if none given
sample(mcmc::MCMCWrapper, args...; kwargs...) = sample(Random.GLOBAL_RNG, mcmc, args...; kwargs...)

# ------------------------------------------------------------------------------------------
# Search for a MCMC stepsize that yields a good MH acceptance rate

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fraction of MC proposals in `chain` which were accepted (according to Metropolis-Hastings.)
"""
function accept_ratio(chain::MCMCChains.Chains)
    if :accepted in names(chain, :internals)
        return mean(chain, :accepted)
    else
        throw("MH `accepted` not recorded in chain: $(names(chain, :internals)).")
    end
end

function _find_mcmc_step_log(mcmc::MCMCWrapper)
    str_ = @sprintf "%d starting params:" 0
    for p in zip(mcmc.sample_kwargs.param_names, mcmc.sample_kwargs.init_params)
        str_ *= @sprintf " %s: %.3g" p[1] p[2]
    end
    println(str_)
    flush(stdout)
end

function _find_mcmc_step_log(it, stepsize, acc_ratio, chain::MCMCChains.Chains)
    str_ = @sprintf "%d stepsize: %.3g acc rate: %.3g\n\tparams:" it stepsize acc_ratio
    for p in pairs(get(chain; section=:parameters)) # can't map() over Pairs
        str_ *= @sprintf " %s: %.3g" p.first last(p.second)
    end
    println(str_)
    flush(stdout)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Use heuristics to choose a stepsize for the [`PriorProposalMHSampler`](@ref) element of 
`mcmc`, namely that MC proposals should be accepted between 15% and 35% of the time.
"""
function optimize_stepsize(
    rng::Random.AbstractRNG, mcmc::MCMCWrapper; 
    init_stepsize = 1.0, N = 2000, max_iter = 20, sample_kwargs...
)
    doubled = false
    halved = false
    _find_mcmc_step_log(mcmc)
    for it = 1:max_iter
        stepsize = init_stepsize
        trial_chain = sample(mcmc, N; stepsize=stepsize, sample_kwargs...)
        acc_ratio = accept_ratio(trial_chain)
        _find_mcmc_step_log(it, stepsize, acc_ratio, trial_chain)
        if doubled && halved
            stepsize = 0.75 * stepsize
            doubled = false
            halved = false
        elseif acc_ratio < 0.15
            stepsize = 0.5 * stepsize
            halved = true
        elseif acc_ratio > 0.35
            stepsize = 2.0 * stepsize
            doubled = true
        else
            @printf "Set sampler to new stepsize: %.3g\n" stepsize
            return stepsize
        end
    end
    throw("Failed to choose suitable stepsize in $(max_iter) iterations.")
end
# use default rng if none given
optimize_stepsize(mcmc::MCMCWrapper; kwargs...) = optimize_stepsize(Random.GLOBAL_RNG, mcmc; kwargs...)


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a `ParameterDistribution` object corresponding to the empirical distribution of the 
MC samples.

NB: multiple MCMC.Chains not implemented.
"""
function get_posterior(mcmc::MCMCWrapper, chain::MCMCChains.Chains)
    p_names = get_name(mcmc.prior)
    p_slices = batch(mcmc.prior)
    flat_constraints = get_all_constraints(mcmc.prior)
    # live in same space as prior
    p_constraints = [flat_constraints[slice] for slice in p_slices]
    
    # Cast data in chain to a ParameterDistribution object. Data layout in Chain is an
    # (N_samples x n_params x n_chains) AxisArray, so samples are in rows.
    p_chain = Array(Chains(chain, :parameters)) # discard internal/diagnostic data
    posterior_samples = [
        Samples(p_chain[:, slice, 1], params_are_columns=false) for slice in p_slices
    ]
    posterior_distribution = ParameterDistribution(posterior_samples, p_constraints, p_names)
    return posterior_distribution
end

end # module MarkovChainMonteCarlo