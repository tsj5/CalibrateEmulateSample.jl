# Import modules
using Random
using Test
using Statistics 
using Distributions
using LinearAlgebra

using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataStorage

@testset "Emulators" begin

   
    #build some quick data + noise
    m=50
    d=6
    x = rand(3, m) #R^3
    y = rand(d, m) #R^5
    
    # "noise"
    μ = zeros(d) 
    Σ = rand(d,d) 
    Σ = Σ'*Σ 
    noise_samples = rand(MvNormal(μ, Σ), m)
    y += noise_samples
    
    iopairs = PairedDataContainer(x,y,data_are_columns=true)
    @test get_inputs(iopairs) == x
    @test get_outputs(iopairs) == y

    
    
    # [1.] test SVD 
    test_SVD = svd(Σ)
    decomposition = Emulators.svd_transform_cov(Σ, truncate_svd=1.0)
    @test_throws MethodError Emulators.svd_transform_cov(Σ[:,1], truncate_svd=1.0)
    @test test_SVD.V[:,:] == decomposition.V #(use [:,:] to make it an array)
    @test test_SVD.Vt == decomposition.Vt
    @test test_SVD.S == decomposition.S

    # 2D y version
    transformed_y = Emulators.to_decorrelated(y; factors=nothing, decomp=decomposition)
    @test size(transformed_y) == size(y)

    # 1D y version
    transformed_y = Emulators.to_decorrelated(y[:, 1]; factors=nothing, decomp=decomposition)
    @test size(transformed_y) == size(y[:, 1])
    
    # Reverse SVD (1D)
    tup = Emulators._from_decorrelated_precompute(nothing, decomposition)
    y_new = Emulators._from_decorrelated_μ(transformed_y, tup)
    y_cov_new = Emulators._from_decorrelated_σ2(ones(d), tup)
    @test size(y_new)[1] == size(y[:, 1])[1]
    @test y_new ≈ y[:,1]
    @test y_cov_new ≈ Σ
    
    # Truncation
    trunc_decomposition = Emulators.svd_transform_cov(Σ, truncate_svd=0.9)
    transformed_y = Emulators.to_decorrelated(y[:, 1]; factors=nothing, decomp=trunc_decomposition)
    trunc_size = size(trunc_decomposition.S)[1]
    @test test_SVD.S[1:trunc_size] == trunc_decomposition.S
    @test size(transformed_y)[1] == trunc_size
    
    # [2.] test Normalization
    input_mean = reshape(mean(get_inputs(iopairs), dims=2), :, 1) #column vector
    sqrt_inv_input_cov = sqrt(inv(Symmetric(cov(get_inputs(iopairs), dims=2))))

    norm_inputs = Emulators.normalize(get_inputs(iopairs),input_mean,sqrt_inv_input_cov)
    @test norm_inputs == sqrt_inv_input_cov * (get_inputs(iopairs) .- input_mean)

    # [3.] test Standardization
    norm_factors = 10.0
    norm_factors = fill(norm_factors, size(y[:,1])) # must be size of output dim
    s_y = Emulators.to_decorrelated(get_outputs(iopairs); factors=norm_factors, decomp=nothing)
    @test s_y ≈ get_outputs(iopairs) ./ norm_factors
    
    # [4.] test emulator preserves the structures

    #build an unknown type
    struct MLTester <: Emulators.MachineLearningTool end
    
    mlt = MLTester()
    
    @test_throws ErrorException emulator = Emulator(
        mlt,
        iopairs,
        obs_noise_cov=Σ,
        normalize_inputs=true,
        standardize_outputs_factors=nothing,
        truncate_svd=1.0)

    #build a known type, with defaults
    gp = GaussianProcess(GPJL())

    emulator = Emulator(
        gp,
        iopairs,
        obs_noise_cov=Σ,
        normalize_inputs=false,
        standardize_outputs_factors=nothing,
        truncate_svd=1.0)
    
    # compare SVD/norm/stand with stored emulator version
    test_decomp = emulator.decomposition
    @test test_decomp.V == decomposition.V #(use [:,:] to make it an array)
    @test test_decomp.Vt == decomposition.Vt
    @test test_decomp.S == decomposition.S

    emulator2 = Emulator(
        gp,
        iopairs,
        obs_noise_cov=Σ,
        normalize_inputs=true,
        standardize_outputs_factors=nothing,
        truncate_svd=1.0)
    train_inputs = get_inputs(emulator2.training_pairs)
    @test norm_inputs ≈ train_inputs

    train_inputs2 = Emulators.normalize(emulator2,get_inputs(iopairs))
    @test norm_inputs ≈ train_inputs

    # reverse standardise
    emulator3 = Emulator(
        gp,
        iopairs,
        obs_noise_cov=Σ,
        normalize_inputs=false,
        standardize_outputs_factors=norm_factors,        
        truncate_svd=1.0)

    #standardized and decorrelated (sd) data
    sd_train_outputs = get_outputs(emulator3.training_pairs)
    sqrt_singular_values_inv = Diagonal(1.0 ./ sqrt.(emulator3.decomposition.S)) 
    decorrelated_s_y = sqrt_singular_values_inv * emulator3.decomposition.Vt * s_y
    @test decorrelated_s_y ≈ sd_train_outputs

    # test 1d transform & inverse
    test_y = rand(d)
    transformed_y = Emulators.to_decorrelated(test_y, emulator3)
    y_new, y_cov_new = Emulators.from_decorrelated(transformed_y, ones(d), emulator3)
    @test size(y_new) == size(test_y)
    @test y_new ≈ test_y
    @test y_cov_new ≈ Σ

    # truncation
    emulator4 = Emulator(
        gp,
        iopairs,
        obs_noise_cov=Σ,
        normalize_inputs=false,
        standardize_outputs_factors=nothing,
        truncate_svd=0.9)
    trunc_size = size(emulator4.decomposition.S)[1]
    @test test_SVD.S[1:trunc_size] == emulator4.decomposition.S

    # test 2d transform & inverse
    N_samp = 10
    test_y = rand(d, N_samp)
    transformed_y = Emulators.to_decorrelated(test_y, emulator3)
    y_new, y_cov_new = Emulators.from_decorrelated(transformed_y, ones(d, N_samp), emulator3)
    @test size(y_new) == size(test_y)
    @test size(y_cov_new) == (N_samp,)
    @test y_new ≈ test_y
    @test y_cov_new[1] ≈ Σ

end
