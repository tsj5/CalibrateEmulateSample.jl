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

# Reexport sample()
using AbstractMCMC: sample
export sample

export 
    GPDensityModel,
    VariableStepProposal,
    VariableStepMHSampler,
    MCMCWrapper,
    get_stepsize,
    set_stepsize!,
    accept_ratio,
    find_mcmc_step!,
    get_posterior,
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
# Record MH accept/reject decision in MHTransition object

"""
    MHTransition

Extend the basic `AdvancedMH.Transition` (which encodes the current state of the MC during
sampling) with a boolean flag to record whether this state is new (arising from accepting a
MH proposal) or old (from rejecting a proposal).
"""
struct MHTransition{T, L<:Real} <: AdvancedMH.AbstractTransition
    params :: T
    lp :: L
    accept :: Bool
end

# Boilerplate from AdvancedMH:
# Store the new draw and its log density.
function MHTransition(model::AdvancedMH.DensityModel, params, accept=true)
    return MHTransition(params, logdensity(model, params), accept)
end
# Calculate the log density of the model given some parameterization.
AdvancedMH.logdensity(model::AdvancedMH.DensityModel, t::MHTransition) = t.lp

# AdvancedMH.transition() is only called to create a new proposal, so create a MHTransition
# with accept = true since that object will only be used if proposal is accepted.
function AdvancedMH.transition(
    sampler::AdvancedMH.MHSampler, 
    model::AdvancedMH.DensityModel, 
    params, logdensity::Real
)
    return MHTransition(params, logdensity, true)
end

# tell propose() what to do with `MHTransition`s
function AdvancedMH.propose(
    rng::Random.AbstractRNG,
    sampler::AdvancedMH.MHSampler,
    model::AdvancedMH.DensityModel,
    transition_prev::MHTransition,
)
    return AdvancedMH.propose(rng, sampler.proposal, model, transition_prev.params)
end

# Copy a MHTransition and set accept = false
function reject_transition(t::MHTransition)
    return MHTransition(t.params, t.lp, false)
end

# Define the other sampling steps.
# Return a 2-tuple consisting of the next sample and the the next state.
# In this case they are identical, and either a new proposal (if accepted)
# or the previous proposal (if not accepted).
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::AdvancedMH.DensityModel,
    sampler::AdvancedMH.MHSampler,
    transition_prev::MHTransition;
    kwargs...
)
    # Generate a new proposal.
    candidate = AdvancedMH.propose(rng, sampler, model, transition_prev)

    # Calculate the log acceptance probability and the log density of the candidate.
    logdensity_candidate = AdvancedMH.logdensity(model, candidate)
    logα = logdensity_candidate - AdvancedMH.logdensity(model, transition_prev) +
        AdvancedMH.logratio_proposal_density(sampler, transition_prev, candidate)

    # Decide whether to return the previous params or the new one.
    transition = if -Random.randexp(rng) < logα
        # accept
        AdvancedMH.transition(sampler, model, candidate, logdensity_candidate)
    else
        # reject
        reject_transition(transition_prev)
    end
    return transition, transition
end

# ------------------------------------------------------------------------------------------
# Extend the record-keeping methods defined in AdvancedMH to include the 
# MHTransition.accept field added above.

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
    ts::Vector{<:MHTransition},
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
    vals = [vcat(t.params, t.lp, t.accept) for t in ts]

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_p, :accept]

    # Bundle everything up and return a MCChains.Chains struct.
    return MCMCChains.Chains(
        vals, vcat(param_names, internal_names), 
        (parameters = param_names, internals = internal_names,);
        start=discard_initial + 1, thin=thinning,
    )
end

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Vector{<:MHTransition}},
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
    # add 2 parameters for log_p, accept
    vals = Array{Float64, 3}(undef, nsamples, n_params + 2, sampler.n_walkers)

    for n in 1:nsamples
        for i in 1:sampler.n_walkers
            walker = ts[n][i]
            for j in 1:n_params
                vals[n, j, i] = walker.params[j]
            end
            vals[n, n_params + 1, i] = walker.lp
            vals[n, n_params + 2, i] = walker.accept
        end
    end

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = [Symbol(:param_, i) for i in 1:length(keys(ts[1][1].params))]
    else
        # Generate new array to be thread safe.
        param_names = Symbol.(param_names)
    end
    internal_names = [:log_p, :accept]

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
    MCMCWrapper

Top-level object to hold the `AdvancedMH.DensityModel` and Sampler objects, as well as 
arguments to be passed to the sampling function.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct MCMCWrapper
    "`AdvancedMH.DensityModel` object, used to generate log-likelihood of data|params."
    model::AbstractMCMC.AbstractModel
    "Object describing a MCMCWrapper sampling algorithm and its settings."
    sampler::AbstractMCMC.AbstractSampler
    "Number of steps to sample in the MC."
    N::Union{Int64, Nothing}
    "NamedTuple of other arguments to be passed to `AbstractMCMC.sample()`."
    sample_kwargs::NamedTuple
end

"""
    MCMCWrapper

Constructor for [`MCMCWrapper`](@ref) which performs obs standardization and takes keywords 
compatible with previous implementation.

- `obs_sample`: A single sample from the observations. Can, e.g., be picked from an Obs 
  struct using get_obs_sample.
- `obs_noise_cov`: Covariance of the observational noise.
- `prior`: array of length `N_parameters` containing the parameters' prior distributions.
- `step`: MCMCWrapper step size.
- `param_init`: Starting point for MCMCWrapper sampling.
- `max_iter`: Number of MCMCWrapper steps to take during sampling.
- `burnin`: Initial number of MCMCWrapper steps to discard (pre-convergence).
- `standardize`: Whether to use SVD to standardize observations.
- `norm_factor`: Optional factor by which to rescale observations.
- `svd`: Whether to use SVD to decorrelate observations.
- `truncate_svd`: Threshold for retaining singular values.
"""
function MCMCWrapper(
    obs_sample::Vector{FT},
    obs_noise_cov::Array{FT, 2},
    prior::ParameterDistribution;
    step::FT,
    param_init::Vector{FT},
    max_iter::Union{IT, Nothing}=nothing,
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
    return MCMCWrapper(
        model, sampler, max_iter, (;
        :init_params => deepcopy(param_init),
        :param_names => get_name(mcmc.prior),
        :discard_initial => burnin,
        :chain_type => MCMCChains.Chains
    ))
end

"""
    get_stepsize

Returns the current MH `stepsize`, assuming we're using [`VariableStepProposal`](@ref) to
generate proposals. Throws an error for Samplers/Proposals without a `stepsize` field.
"""
function get_stepsize(mcmc::MCMCWrapper)
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
function set_stepsize!(mcmc::MCMCWrapper, new_step)
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
# Define new methods extending AbstractMCMC.sample() using the MCMCWrapper object defined above.

# use default rng if none given
function AbstractMCMC.sample(mcmc::MCMCWrapper, args...; kwargs...)
    return AbstractMCMC.sample(Random.GLOBAL_RNG, mcmc, args...; kwargs...)
end

# vanilla case
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    mcmc::MCMCWrapper,
    N::Union{Integer, Nothing} = nothing;
    kwargs...
)
    if N === nothing
        N = mcmc.N
    end
    if N === nothing || N < 0
        throw("Misspecified number of steps: $(N)")
    end
    # explicit function kwargs override what's in mcmc
    kwargs = merge(mcmc.sample_kwargs, NamedTuple(kwargs))
    return AbstractMCMC.mcmcsample(rng, mcmc.model, mcmc.sampler, N; kwargs...)
end

# case where we pass an isdone() Function for early termination
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    isdone;
    kwargs...
)
    # explicit function kwargs override what's in mcmc
    kwargs = merge(mcmc.sample_kwargs, NamedTuple(kwargs))
    return AbstractMCMC.mcmcsample(rng, model, sampler, isdone; kwargs...)
end

# parallel case
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::AbstractModel,
    sampler::AbstractSampler,
    parallel::AbstractMCMCEnsemble,
    N::Integer,
    nchains::Integer;
    kwargs...
)
    # explicit function kwargs override what's in mcmc
    kwargs = merge(mcmc.sample_kwargs, NamedTuple(kwargs))
    return AbstractMCMC.mcmcsample(rng, mcmc.model, mcmc.sampler, parallel, N, nchains; kwargs...)
end

# ------------------------------------------------------------------------------------------
# still to be updated

function accept_ratio(chain::MCMCChains.Chains)
    if :accept in names(chain, :internals)
        return mean(chain, :accept)
    else
        throw("MH `accept` not recorded in chain: $(names(chain, :internals)).")
    end
end

function find_mcmc_step!(mcmc_test::MCMCWrapper, gp::GaussianProcess{FT}; max_iter=2000) where {FT}
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

function get_posterior(mcmc::MCMCWrapper)
    #Return a parameter distributions object
    parameter_slices = batch(mcmc.prior)
    posterior_samples = [Samples(mcmc.posterior[slice,mcmc.burnin+1:end]) for slice in parameter_slices]
    flattened_constraints = get_all_constraints(mcmc.prior)
    parameter_constraints = [flattened_constraints[slice] for slice in parameter_slices] #live in same space as prior
    parameter_names = get_name(mcmc.prior) #the same parameters as in prior
    posterior_distribution = ParameterDistribution(posterior_samples, parameter_constraints, parameter_names)
    return posterior_distribution
    
end

function sample_posterior!(mcmc::MCMCWrapper,
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
