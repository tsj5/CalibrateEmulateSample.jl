module Emulators

using ..DataStorage
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions

export 
    Emulator,
    optimize_hyperparameters!,
    predict,
    to_decorrelated,
    from_decorrelated

"""
    MachineLearningTool

Type to dispatch different emulators:

 - GaussianProcess <: MachineLearningTool
"""
abstract type MachineLearningTool end
# defaults in error, all MachineLearningTools require these functions.
function throw_define_mlt()
    throw(ErrorException("Unknown MachineLearningTool defined, please use a known implementation"))
end
function build_models!(mlt,iopairs) throw_define_mlt() end
function optimize_hyperparameters!(mlt) throw_define_mlt() end
function predict(mlt,new_inputs) throw_define_mlt() end

# include the different <: ML models
include("GaussianProcess.jl") #for GaussianProcess
# include("RandomFeature.jl")
# include("NeuralNetwork.jl")
# etc.


# We will define the different emulator types after the general statements

"""
    Emulator

Structure used to represent a general emulator.

# Fields

$(TYPEDFIELDS)
"""
struct Emulator{FT<:AbstractFloat}
    "Machine learning tool, defined as a struct of type MachineLearningTool"
    machine_learning_tool::MachineLearningTool
    "Normalized, standardized pairs of input/output data"
    training_pairs::PairedDataContainer{FT}
    "Mean of input; 1 × input_dim"
    input_mean::Array{FT, 2}
    "If given, normalize inputs by inverse square root of the input covariance matrix; input_dim × input_dim"
    sqrt_inv_input_cov::Union{Array{FT, 2}, Nothing}
    "If given, standardize outputs (`outputs/ standardize_outputs_factor`)"
    standardize_outputs_factors::Union{Array{FT,1}, Nothing}
    "The singular value decomposition of obs_noise_cov, such that obs_noise_cov = decomposition.U * Diagonal(decomposition.S) * decomposition.Vt. NB: the svd may be reduced in dimensions"
    decomposition::Union{SVD, Nothing}
end 

# Constructor for the Emulator Object
function Emulator(
    machine_learning_tool::MachineLearningTool,
    input_output_pairs::PairedDataContainer{FT};
    obs_noise_cov::Union{Nothing, Array{FT, 2}} = nothing,
    normalize_inputs::Bool = true,
    standardize_outputs_factors::Union{Array{FT,1}, Nothing} = nothing,
    truncate_svd::FT = 1.0
    ) where {FT<:AbstractFloat}

    # For Consistency checks
    input_dim, output_dim = size(input_output_pairs, 1)
    if obs_noise_cov !== nothing
        err2 = "obs_noise_cov must be of size ($output_dim, $output_dim), got $(size(obs_noise_cov))"
        size(obs_noise_cov) == (output_dim, output_dim) || throw(ArgumentError(err2))
    end

    input_mean = reshape(mean(get_inputs(input_output_pairs), dims=2), :, 1) #column vector
    
    # [1.] Normalize the inputs? 
    if normalize_inputs
        # Normalize (NB the inputs have to be of) size [input_dim × N_samples] to pass to GPE())
        sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(input_output_pairs), dims=2))))
        training_inputs = normalize(
            get_inputs(input_output_pairs),
            input_mean,
            sqrt_inv_input_cov
        )
    else
        sqrt_inv_input_cov = nothing
        training_inputs = get_inputs(input_output_pairs)
    end

    # [2.], [3.] Standardize and decorrelate the outputs
    decomposition = svd_transform_cov(
        obs_noise_cov; 
        factors = standardize_outputs_factors, truncate_svd = truncate_svd
    )
    training_outputs = to_decorrelated(
        get_outputs(input_output_pairs);
        factors = standardize_outputs_factors, decomp = decomposition
    )

    # [4.] build an emulator
    training_pairs = PairedDataContainer(training_inputs, training_outputs)
    build_models!(machine_learning_tool, training_pairs)
    
    return Emulator{FT}(
        machine_learning_tool,
        training_pairs,
        input_mean,
        sqrt_inv_input_cov,
        standardize_outputs_factors,
        decomposition
    )
end    

"""
    function optimize_hyperparameters!(emulator::Emulator)

optimize the hyperparameters in the machine learning tool
"""
function optimize_hyperparameters!(emulator::Emulator{FT}) where {FT}
    optimize_hyperparameters!(emulator.machine_learning_tool)
end


"""
    function predict(emulator::Emulator, new_input; transform_to_real=false) 

Makes a prediction using the emulator on a new input. Default is to predict in the 
decorrelated space.
"""
function predict(
    emulator::Emulator{FT}, new_input::Vector{FT}; transform_to_real=false
) where {FT<:AbstractFloat}
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(emulator.training_pairs, 1)
    size(new_input, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    # [1.] normalize
    normalized_new_input = normalize(emulator, reshape(new_input, :, 1))

    # [2.]  predict. Note: ds = decorrelated, standard
    ds_output, ds_output_var = predict(emulator.machine_learning_tool, normalized_new_input)

    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    if transform_to_real 
        #transform back to real coords - cov becomes dense
        return from_decorrelated(vec(ds_output), ds_output_var, emulator)
    else
        # remain in decorrelated, standardized coordinates (cov remains diagonal)
        return vec(ds_output), Diagonal(ds_output_var)
    end
end

"""
    function predict(emulator::Emulator, new_inputs; transform_to_real=false) 

makes a prediction using the emulator on multiple new inputs (each new inputs given as data 
columns). Default is to predict in the decorrelated space.
"""
function predict(
    emulator::Emulator{FT}, new_inputs::Array{FT, 2}; transform_to_real=false
) where {FT<:AbstractFloat}
    # Check if the size of new_inputs is consistent with the GP model's input
    # dimension. 
    input_dim, output_dim = size(emulator.training_pairs, 1)
    N_samples = size(new_inputs, 2)
    size(new_inputs, 1) == input_dim || throw(ArgumentError("GP object and input observations do not have consistent dimensions"))

    # [1.] normalize
    normalized_new_inputs = normalize(emulator,new_inputs)

    # [2.]  predict. Note: ds = decorrelated, standard
    ds_outputs, ds_output_vars = predict(emulator.machine_learning_tool, normalized_new_inputs)

    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    if transform_to_real 
        #transform back to real coords - cov becomes dense
        s_outputs, s_output_covs = from_decorrelated(ds_outputs, ds_output_vars, emulator)
    else
        # remain in decorrelated, standardized coordinates (cov remains diagonal)
        # Convert to vector of matrices to match the format  
        # when transform_to_real=true
        s_outputs = ds_outputs
        s_output_covs = vec([Diagonal(ds_output_vars[:, j]) for j in 1:N_samples])
    end
    if output_dim == 1
        s_output_covs = [s_output_covs[i][1] for i in 1:N_samples]
    end
    return s_outputs, s_output_covs
end

# ------------------------------------------------------------------------------------------
# Normalization and Standardization/ Decorrelation
"""
    function normalize(emulator::Emulator, inputs)

normalize the input data, with a normalizing function
"""
function normalize(emulator::Emulator{FT}, inputs) where {FT<:AbstractFloat}
    if emulator.sqrt_inv_input_cov !== nothing
        return normalize(inputs, emulator.input_mean, emulator.sqrt_inv_input_cov)
    else
        return inputs
    end
end

"""
    function normalize(inputs, input_mean, sqrt_inv_input_cov)

normalize with the empirical Gaussian distribution of points
"""
function normalize(inputs, input_mean, sqrt_inv_input_cov)
    training_inputs = sqrt_inv_input_cov * (inputs .- input_mean)
    return training_inputs 
end

"""
    svd_transform_cov(obs_noise_cov; factors, truncate_svd)

Apply a singular value decomposition (SVD) to the covariance of observational noise.
  - `obs_noise_cov` - Covariance of observational noise.
  - `factors` - Optional vector of standardize_outputs_factors to scale by, length output_dim.
  - `truncate_svd` - Project onto this fraction of the largest principal components. Defaults
    to 1.0 (no truncation).

Returns the (possibly truncated) SVD decomposition, of type LinearAlgebra.SVD, of the
scaled covariance matrix.
  
Note: If F::SVD is the factorization object, U, S, V and Vt can be obtained via 
F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt. The singular values 
in S are sorted in descending order.
"""
function svd_transform_cov(
    obs_noise_cov::Array{FT, 2}; 
    factors::Union{Array{FT,1}, Nothing} = nothing,
    truncate_svd::FT = 1.0
) where {FT<:AbstractFloat}
    if factors !== nothing
        # standardize() cov by scale factors, if they were given
        obs_noise_cov = obs_noise_cov ./ (factors .* factors')
    end

    decomp = svd(obs_noise_cov)
	if truncate_svd < 1.0
        # Truncate the SVD as a form of regularization
        # Find cutoff
        S_cumsum = cumsum(decomp.S) / sum(decomp.S)
        ind = findall(x -> (x > truncate_svd), S_cumsum)
        k = ind[1]
        n = size(obs_noise_cov)[1]
        println("SVD truncated at k: ", k, "/", n)
	    return SVD(decomp.U[:, 1:k], decomp.S[1:k], decomp.Vt[1:k, :])
	else
	    return decomp
    end
end

function svd_transform_cov(::Nothing; kwargs...)
    # method for no-op case: no obs_noise_cov
    return nothing
end

function to_decorrelated(
    data::Array{FT, 2};
    factors::Union{Array{FT,1}, Nothing} = nothing,
    decomp::Union{SVD, Nothing} = nothing
) where {FT<:AbstractFloat}
    if factors !== nothing
        # standardize() data by scale factors, if they were given
        data = data ./ factors
    end
    if decomp !== nothing
        # Use SVD decomposition of obs noise cov, if given,  to transform data to 
        # decorrelated coordinates.
        sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(decomp.S)) 
        return sqrt_singular_values_inv * decomp.Vt * data
    else
        return data
    end
end


to_decorrelated(data::Array{FT, 2}, em::Emulator{FT}) where {FT<:AbstractFloat} = to_decorrelated(
    data; factors=em.standardize_outputs_factors, decomp=em.decomposition
)

function to_decorrelated(data::Vector{FT}, args...; kwargs...) where {FT<:AbstractFloat}
    out_data = to_decorrelated(reshape(data, :, 1), args...; kwargs...)
    return vec(out_data)
end


function _from_decorrelated_precompute(
    factors::Union{Array{FT,1}, Nothing}, decomposition::Union{SVD, Nothing}
) where {FT<:AbstractFloat}
    # Returns NamedTuple of stuff we compute once per call to from_decorrelated().
    if factors !== nothing
        cov_factors = (factors .* factors')
    else
        cov_factors = nothing
    end
    if decomposition !== nothing
        sqrt_singvals = Diagonal(sqrt.(decomposition.S))
        return (
            factors = factors,
            cov_factors = cov_factors,
            V_sqrtD = decomposition.V * sqrt_singvals,
            sqrtD_Vt = sqrt_singvals * decomposition.Vt
        )
    else
        return (
            factors = factors,
            cov_factors = cov_factors,
            V_sqrtD = nothing,
            sqrtD_Vt = nothing
        )
    end
end

function _from_decorrelated_μ(
    μ::Union{Vector{FT}, Matrix{FT}}, tup::NamedTuple
) where {FT<:AbstractFloat}
    # Subroutine for transforming mean(s) from decorrelated to original coords.
    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    if tup.V_sqrtD !== nothing
        # We created meanvGP = D_inv * Vt * mean_v so meanv = V * D * meanvGP.
        μ = tup.V_sqrtD * μ
    end
    # [4.] unstandardize
    if tup.factors !== nothing
        μ = μ .* tup.factors
    end
    return μ
end

function _from_decorrelated_σ2(σ2::Vector{FT}, tup::NamedTuple) where {FT<:AbstractFloat}
    # Subroutine for transforming a single vector of variances from decorrelated to 
    # original coords.
    # [3.] transform back to real coordinates or remain in decorrelated coordinates
    σ2 = Diagonal(σ2)
    if tup.V_sqrtD !== nothing
        # transform back to real coords - cov becomes dense
        σ2 = tup.V_sqrtD * σ2 * tup.sqrtD_Vt
    end
    # [4.] unstandardize
    if tup.cov_factors !== nothing
        σ2 = σ2 .* tup.cov_factors
    end
    # cast to appropriate form
    if tup.V_sqrtD !== nothing
        return Symmetric(σ2)
    else
        return Diagonal(σ2)
    end
end

"""
    from_decorrelated(μ::Vector{FT}, σ2::Vector{FT}, em::Emulator)

Transform the mean and covariance back to the original (correlated) coordinate system
  - `μ` - predicted mean; length output_dim.
  - `σ2` - predicted variance as vector, length output_dim. 
  - `em` - [`Emulator`](@ref) object.

Return the mean and covariance, transformed back to the original coordinate system.
"""
function from_decorrelated(μ::Vector{FT}, σ2::Vector{FT}, em::Emulator{FT}) where {FT<:AbstractFloat}
    tup = _from_decorrelated_precompute(em.standardize_outputs_factors, em.decomposition)
    return _from_decorrelated_μ(μ, tup), _from_decorrelated_σ2(σ2, tup)
end

"""
    from_decorrelated(μs::Array{FT, 2}, σ2s::Array{FT, 2}, em::Emulator)

Transform the mean and covariance back to the original (correlated) coordinate system
  - `μs` - predicted means; output_dim × N_samples.
  - `σ2s` - predicted variances; output_dim × N_samples. 
  - `em` - [`Emulator`](@ref) object.

Returns the transformed means (output_dim × N_predicted_points) and covariances. 
Note that transforming the variance back to the original coordinate system
results in non-zero off-diagonal elements, so instead of just returning the 
elements on the main diagonal (i.e., the variances), we return the full 
covariance at each point, as a Vector of length N_samples, where 
each element is a matrix of size output_dim × output_dim.
"""
function from_decorrelated(μs::Array{FT, 2}, σ2s::Array{FT, 2}, em::Emulator{FT}) where {FT<:AbstractFloat}
    @assert size(μs) == size(σ2s)
    output_dim, N_samples = size(σ2s)

    tup = _from_decorrelated_precompute(em.standardize_outputs_factors, em.decomposition)
    xform_μs = _from_decorrelated_μ(μs, tup)

    if em.decomposition !== nothing
        # transform back to real coords - cov becomes dense
        xform_σ2s = Vector{Symmetric{FT, Matrix{FT}}}(undef, N_samples)
    else 
        xform_σ2s = Vector{Diagonal{FT, Vector{FT}}}(undef, N_samples)
    end
    for j in 1:N_samples
        xform_σ2s[j] = _from_decorrelated_σ2(σ2s[:,j], tup)
    end
    return xform_μs, xform_σ2s
end

end
