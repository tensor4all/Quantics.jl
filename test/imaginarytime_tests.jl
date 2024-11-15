@testitem "imaginarytime_tests.jl/imaginarytime" begin
    using Test
    using Quantics
    using ITensors: siteinds, Index
    using ITensors.SiteTypes: op
    import ITensors
    import SparseIR: Fermionic, Bosonic, FermionicFreq, valueim

    function _test_data_imaginarytime(nbit, β)
        ω = 0.5
        N = 2^nbit
        halfN = 2^(nbit - 1)

        # Tau
        gtau(τ) = -exp(-ω * τ) / (1 + exp(-ω * β))
        @assert gtau(0.0) + gtau(β) ≈ -1
        τs = collect(LinRange(0.0, β, N + 1))[1:(end - 1)]
        gtau_smpl = Vector{ComplexF64}(gtau.(τs))

        # Matsubra
        giv(v::FermionicFreq) = 1 / (valueim(v, β) - ω)
        vs = FermionicFreq.(2 .* collect((-halfN):(halfN - 1)) .+ 1)
        giv_smpl = giv.(vs)

        return gtau_smpl, giv_smpl
    end

    @testset "decompose" begin
        β = 2.0
        nbit = 10
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sites = siteinds("Qubit", nbit)
        gtau_mps = Quantics.decompose_gtau(gtau_smpl, sites; cutoff=1e-20)

        gtau_smpl_reconst = vec(Array(reduce(*, gtau_mps), reverse(sites)...))

        @test gtau_smpl_reconst ≈ gtau_smpl
    end

    @testset "ImaginaryTimeFT.to_wn" begin
        ITensors.set_warn_order(100)
        β = 1.5
        nbit = 6
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sitesτ = [Index(2, "Qubit,τ=$n") for n in 1:nbit]
        sitesiω = [Index(2, "Qubit,iω=$n") for n in 1:nbit]
        gtau_mps = Quantics.decompose_gtau(gtau_smpl, sitesτ; cutoff=1e-20)
        giv_mps = Quantics.to_wn(Fermionic(), gtau_mps, β; cutoff=1e-20, tag="τ",
            sitesdst=sitesiω)

        giv = vec(Array(reduce(*, giv_mps), reverse(sitesiω)...))

        @test maximum(abs, giv - giv_smpl) < 2e-2
    end

    @testset "ImaginaryTimeFT.to_tau" begin
        ITensors.set_warn_order(100)
        β = 1.5
        nbit = 8
        nτ = 2^nbit

        gtau_smpl, giv_smpl = _test_data_imaginarytime(nbit, β)

        sitesτ = [Index(2, "Qubit,τ=$n") for n in 1:nbit]
        sitesiω = [Index(2, "Qubit,iω=$n") for n in 1:nbit]
        giv_mps = Quantics.decompose_giv(giv_smpl, sitesiω; cutoff=1e-20)

        gtau_mps = Quantics.to_tau(Fermionic(), giv_mps, β; cutoff=1e-20, tag="iω",
            sitesdst=sitesτ)

        gtau = vec(Array(reduce(*, gtau_mps), reverse(sitesτ)...))

        # There is ocillation around tau = 0, beta.
        @test maximum(abs, (gtau - gtau_smpl)[trunc(Int, 0.2 * nτ):trunc(Int, 0.8 * nτ)]) <
              1e-2
    end
end

@testitem "imaginarytime_tests.jl/poletomps" begin
    using Test
    using Quantics
    import ITensors: siteinds, Index
    import ITensors
    import SparseIR: Fermionic, Bosonic, FermionicFreq, valueim

    @testset "poletomps" begin
        nqubit = 10
        sites = siteinds("Qubit", nqubit)
        β = 10.0
        ω = 1.2
        gtau = Quantics.poletomps(sites, β, ω)
        gtauvec = vec(Array(reduce(*, gtau), reverse(sites)))
        gtauf(τ) = -exp(-τ * ω) / (1 + exp(-β * ω))
        gtauref = gtauf.(LinRange(0, β, 2^nqubit + 1)[1:(end - 1)])
        @test maximum(abs, gtauref .- gtauvec) < 1e-14
    end

    @testset "poletomps_negative_pole" begin
        nqubit = 16
        sites = siteinds("Qubit", nqubit)
        β = 1000.0
        ω = -10.0
        gtau = Quantics.poletomps(Fermionic(), sites, β, ω)
        gtauvec = vec(Array(reduce(*, gtau), reverse(sites)))
        gtauf(τ) = -exp((β - τ) * ω)
        gtauref = gtauf.(LinRange(0, β, 2^nqubit + 1)[1:(end - 1)])
        @test maximum(abs, gtauref .- gtauvec) < 1e-14
    end
end
