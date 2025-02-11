# AbstractMCMC sampling API

```@meta
CurrentModule = CalibrateEmulateSample.MarkovChainMonteCarlo
```

The "sample" part of CES refers to exact sampling from the emulated posterior via [Markov chain Monte
Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC). Within this paradigm, we want to provide the
flexibility to use multiple sampling algorithms; the approach we take is to use the general-purpose
[AbstractMCMC.jl](https://turing.ml/dev/docs/for-developers/interface) API, provided by the
[Turing.jl](https://turing.ml/dev/) probabilistic programming framework.

This page provides a summary of AbstractMCMC which augments the existing documentation
(\[[1](https://turing.ml/dev/docs/for-developers/interface)\],
\[[2](https://turing.ml/dev/docs/for-developers/how_turing_implements_abstractmcmc)\]) and highlights how it's used by
the CES package in [MarkovChainMonteCarlo](@ref). It's not meant to be a complete description of the AbstractMCMC
package.

## Use in MarkovChainMonteCarlo

At present, Turing has limited support for derivative-free optimization, so we only use this abstract API and not Turing
itself. We also use two related dependencies, [AdvancedMH](https://github.com/TuringLang/AdvancedMH.jl) and
[MCMCChains](https://github.com/TuringLang/MCMCChains.jl). 

Julia's philosophy is to use small, composable packages rather than monoliths, but this can make it difficult to
remember where methods are defined! Below we describe the relevant parts of 

- The AbstractMCMC API,
- Extended to the case of Metropolis-Hastings (MH) sampling by AdvancedMH,
- Further extended for the needs of CES in [Markov chain Monte
  Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo).

## Classes and methods

### Sampler

A Sampler is AbstractMCMC's term for an implementation of a MCMC sampling algorithm, along with all its configuration
parameters. All samplers must inherit from `AbstractMCMC.AbstractSampler`. 

Currently CES only implements the Metropolis-Hastings (MH) algorithm. Because it's so straightforward, much of
AbstractMCMC isn't needed. We implement two variants of MH with two different Samplers: `RWMetropolisHastings` and
`pCNMetropolisHastings`, both of which inherit from the `AdvancedMH.MHSampler` base class. The public constructor for
both Samplers is [`MetropolisHastingsSampler`](@ref); the different Samplers are specified by passing a
[`MCMCProtocol`](@ref) object to this constructor.

The MH Sampler classes have only one field, `proposal`, which is the distribution used to generate new MH proposals via
stochastic offsets to the current parameter values. This is done by
[AdvancedMH.propose()](https://github.com/TuringLang/AdvancedMH.jl/blob/master/src/proposal.jl), which gets called for
each MCMC `step()` (below). The difference between our two Samplers is in how this proposal is generated:

- [`RWMHSampling`](@ref) does vanilla random-walk proposal generation with a constant, user-specified step size (this
  differs from the AdvancedMH implementation, which doesn't provide for a step size.)

- [`pCNMHSampling`](@ref) for preconditioned Crank-Nicholson proposals. Vanilla random walk sampling doesn't have a
  well-defined limit for high-dimensional parameter spaces; pCN replaces the random walk with an Ornstein–Uhlenbeck
  [AR(1)] process so that the Metropolis acceptance probability remains non-zero in this limit. See [Beskos et. al.
  (2008)](https://www.worldscientific.com/doi/abs/10.1142/S0219493708002378) and [Cotter et. al.
  (2013)](https://projecteuclid.org/journals/statistical-science/volume-28/issue-3/MCMC-Methods-for-Functions--Modifying-Old-Algorithms-to-Make/10.1214/13-STS421.full).

This is the only difference: generated proposals are then either accepted or rejected according to the same MH criterion
(in `step()`, below.)

### Models

In Turing, the Model is the distribution one performs inference on, which may involve observed and hidden variables and
parameters. For CES, we simply want to sample from the posterior, so our Model distribution is simply the emulated
likelihood (see [Emulators](@ref)) together with the prior. This is constructed by [`EmulatorPosteriorModel`](@ref).

### Sampling with the MCMC Wrapper object

At a [high level](https://turing.ml/dev/docs/using-turing/guide), a Sampler and Model is all that's needed to do MCMC
sampling. This is done by the [`sample`](https://github.com/TuringLang/AbstractMCMC.jl/blob/master/src/sample.jl) method
provided by AbstractMCMC (extending the method from BaseStats). 

To be more user-friendly, in CES we wrap the Sampler, Model and other necessary configuration into a
[`MCMCWrapper`](@ref) object. The constructor for this object ensures that all its components are created consistently,
and performs necessary bookkeeping, such as converting coordinates to the decorrelated basis. We extend [`sample`](@ref)
with methods to use this object (that simply unpack its fields and call the appropriate method from AbstractMCMC.)

### Chain

The [MCMCChain](https://beta.turing.ml/MCMCChains.jl/dev/) class is used to store the results of the MCMC sampling; the
package provides simple diagnostics for visualization and diagnosing chain convergence.


### Internals: Transitions

Implementing MCMC involves defining states and transitions of a Markov process (whose stationary distribution is what we
seek to sample from). AbstractMCMC's terminology is a bit confusing for the MH case; *states* of the chain are described
by `Transition` objects, which contain the current sample (and other information like its log-probability). 

AdvancedMH defines an `AbstractTransition` base class for use with its methods; we implement our own child class,
[`MCMCState`](@ref), in order to record statistics on the MH acceptance ratio.

### Internals: Markov steps

Markov *transitions* of the chain are defined by overloading AbstractMCMC's `step` method, which takes the Sampler and
current `Transition` and implements the Sampler's logic to returns an updated `Transition` representing the chain's new
state (actually, a pair of `Transitions`, for cases where the Sampler doesn't obey detailed balance; this isn't relevant
for us). 

For example, in Metropolis-Hastings sampling this is where we draw a proposal sample and accept or reject it according
to the MH criterion. AdvancedMH implements this
[here](https://github.com/TuringLang/AdvancedMH.jl/blob/ba86e49a3ebd1ee94d0becc3211738e3be6fd538/src/mh-core.jl#L90-L113);
we re-implement this method because 1) we need to record whether a proposal was accepted or rejected, and 2) our calls
to `propose()` are stepsize-dependent.

