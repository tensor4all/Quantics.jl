@testitem "util.jl" begin
    using Test
    import PartitionedMPSs: PartitionedMPSs, SubDomainMPS, PartitionedMPS, project,
                            isprojectedat
    import Quantics
    using ITensors
    using ITensors.SiteTypes: siteinds
    using ITensorMPS: randomMPS, randomMPO, random_mps, MPO, MPS

    include("_util.jl")

    @testset "_replace_mpo_siteinds!" begin
        nbit = 3
        sites = siteinds("Qubit", nbit)
        M = MPO(ComplexF64, sites, ["Y" for n in 1:nbit])

        sites2 = [Index(2, "n=$n") for n in 1:nbit]
        Quantics._replace_mpo_siteinds!(M, sites, sites2)

        @test all([!hasind(M[n], sites[n]) for n in 1:nbit])
        @test all([!hasind(M[n], sites[n]') for n in 1:nbit])
        @test all([hasind(M[n], sites2[n]) for n in 1:nbit])
        @test all([hasind(M[n], sites2[n]') for n in 1:nbit])
    end

    #==
    @testset "combinesiteinds" begin
        # [s1, (s2,s3), (s4,s5), s6]
        nbit = 6
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:2]
        M = randomMPS(sites; linkdims=2)

        Mc = Quantics.combinesiteinds(M, csites; targetsites=sites[2:5])

        @test length(Mc) == 4
        @test all(dim.(siteinds(Mc)) .== [2, 4, 4, 2])
    end

    @testset "splitsiteind (deprecated)" for nbit in [4, 6]
        sites = siteinds("Qubit", nbit)
        csites = [Index(4, "csite=$s") for s in 1:(nbit ÷ 2)]
        D = 3
        mps = randomMPS(csites; linkdims=D)
        mps_split = Quantics.splitsiteind(mps, sites)
        @test vec(Array(reduce(*, mps_split), sites)) ≈ vec(Array(reduce(*, mps), csites))

        mps_reconst = Quantics.combinesiteinds(mps_split, csites)
        @test vec(Array(reduce(*, mps_reconst), csites)) ≈
              vec(Array(reduce(*, mps), csites))
    end
    ==#

    @testset "unfuse_siteinds" for nsites in [2, 4], R in [2, 3]
        sites = [Index(2^R, "csite=$s") for s in 1:nsites]

        bonddim = 3
        mps = randomMPS(sites; linkdims=bonddim)

        newsites = [[Index(2, "n=$n,m=$m") for m in 1:R] for n in 1:nsites]

        mps_split = Quantics.unfuse_siteinds(mps, sites, newsites)

        newsites_flatten = collect(Iterators.flatten(newsites))
        @test newsites_flatten == siteinds(mps_split)
        @test vec(Array(reduce(*, mps_split), newsites_flatten)) ≈
              vec(Array(reduce(*, mps), sites))
    end

    @testset "split_tensor" begin
        nsite = 6
        sites = [Index(2, "Qubit, site=$n") for n in 1:nsite]
        tensor = randomITensor(sites)
        tensors = Quantics.split_tensor(tensor, [sites[1:2], sites[3:4], sites[5:6]])
        @test tensor ≈ reduce(*, tensors)
    end

    @testset "split_tensor2" begin
        nsite = 8
        sites = [Index(2, "Qubit, site=$n") for n in 1:nsite]
        tensor = randomITensor(sites)
        tensors = Quantics.split_tensor(tensor, [sites[1:3], sites[4:5], sites[6:8]])
        @test length(inds(tensors[1])) == 4
        @test length(inds(tensors[2])) == 4
        @test length(inds(tensors[3])) == 4
        @test tensor ≈ reduce(*, tensors)
    end

    @testset "matchsiteinds_mps" begin
        N = 2
        physdim = 2

        sites = [Index(physdim, "n=$n") for n in 1:(2N)]
        sites_sub = sites[1:2:end]
        M = randomMPS(sites_sub) + randomMPS(sites_sub)

        M_ext = Quantics.matchsiteinds(M, sites)

        tensor = Array(reduce(*, M), sites_sub)
        tensor_reconst = zeros(Float64, fill(physdim, 2N)...)
        tensor_reconst .= reshape(tensor, size(tensor)..., fill(1, N)...)

        tensor2 = Array(reduce(*, M_ext), sites_sub, sites[2:2:end])
        @test tensor2 ≈ tensor_reconst
    end

    @testset "matchsiteinds_mpo" begin
        N = 2
        physdim = 2

        sites = [Index(physdim, "n=$n") for n in 1:(2N)]
        sites_A = sites[1:2:end]
        sites_B = sites[2:2:end]
        M = randomMPO(sites_A) + randomMPO(sites_A)

        M_ext = Quantics.matchsiteinds(M, sites)

        tensor_ref = reduce(*, M) * reduce(*, [delta(s, s') for s in sites_B])
        tensor_reconst = reduce(*, M_ext)
        @test tensor_ref ≈ tensor_reconst
    end

    @testset "matchsiteinds_mpo2" begin
        N = 2
        physdim = 2

        sites = [Index(physdim, "n=$n") for n in 1:(3N)]
        sites_A = sites[1:3:end]
        sites_B = sites[2:3:end]
        sites_C = sites[3:3:end]
        sites_BC = vcat(sites_B, sites_C)
        M = randomMPO(sites_A) + randomMPO(sites_A)

        M_ext = Quantics.matchsiteinds(M, sites)

        tensor_ref = reduce(*, M) * reduce(*, [delta(s, s') for s in sites_BC])
        tensor_reconst = reduce(*, M_ext)
        @test tensor_ref ≈ tensor_reconst
    end

    @testset "combinsite" begin
        nrepeat = 3
        N = 3 * nrepeat
        sites = siteinds("Qubit", N)
        M = MPO(randomMPS(sites))
        sites1 = sites[1:3:end]
        sites2 = sites[2:3:end]
        sites3 = sites[3:3:end]
        for n in 1:nrepeat
            M = Quantics.combinesites(M, sites1[n], sites2[n])
        end
        flag = true
        for n in 1:nrepeat
            flag = flag && hasinds(M[2 * n - 1], sites1[n], sites2[n])
            flag = flag && hasind(M[2 * n], sites3[n])
        end
        @test flag
    end

    @testset "_directprod" begin
        sites1 = siteinds("Qubit", 2)
        sites2 = siteinds("Qubit", 2)
        M1 = randomMPS(sites1)
        M2 = randomMPS(sites2)
        M12 = Quantics._directprod(M1, M2)

        M1_reconst = Array(reduce(*, M1), sites1)
        M2_reconst = Array(reduce(*, M2), sites2)
        M12_reconst = Array(reduce(*, M12), vcat(sites1, sites2))

        M12_ref = reshape(reshape(M1_reconst, 2^2, 1) * reshape(M2_reconst, 1, 2^2), 2, 2,
            2, 2)

        @test M12_reconst ≈ M12_ref
    end

    @testset "rearrange_siteinds (xy)" begin
        L = 5
        sitesx = [Index(2, "x=$n") for n in 1:L]
        sitesy = [Index(2, "y=$n") for n in 1:L]

        sitesxy = collect(Iterators.flatten(zip(sitesx, sitesy)))

        Ψ = random_mps(sitesxy)

        sitesxy_fused = [[x, y] for (x, y) in zip(sitesx, sitesy)]

        Ψ_fused = Quantics.rearrange_siteinds(Ψ, sitesxy_fused)

        @test prod(Ψ) ≈ prod(Ψ_fused) # We reconstruct a full tensor, do not use it for large L

        sitesxy_fused_ = siteinds(MPO(collect(Ψ_fused)))

        for (x, y) in zip(sitesxy_fused, sitesxy_fused_)
            @test Set(x) == Set(y)
        end

        Ψ_reconst = Quantics.rearrange_siteinds(Ψ_fused, [[x] for x in sitesxy])

        @test Ψ ≈ Ψ_reconst
    end

    @testset "rearrange_siteinds (xyz)" begin
        L = 4
        sitesx = [Index(2, "x=$n") for n in 1:L]
        sitesy = [Index(2, "y=$n") for n in 1:L]
        sitesz = [Index(2, "z=$n") for n in 1:L]

        sitesxyz = collect(Iterators.flatten(zip(sitesx, sitesy, sitesz)))

        Ψ = random_mps(sitesxyz)

        sitesxyz_fused = Vector{Index{Int}}[]
        for i in 1:L
            push!(sitesxyz_fused, [sitesx[i], sitesy[i]])
            push!(sitesxyz_fused, [sitesz[i]])
        end

        Ψ_fused = Quantics.rearrange_siteinds(Ψ, sitesxyz_fused)

        @test prod(Ψ) ≈ prod(Ψ_fused)

        sitesxyz_fused_ = siteinds(MPO(collect(Ψ_fused)))

        for (x, y) in zip(sitesxyz_fused, sitesxyz_fused_)
            @test Set(x) == Set(y)
        end

        Ψ_reconst = Quantics.rearrange_siteinds(Ψ_fused, [[x] for x in sitesxyz])

        @test Ψ ≈ Ψ_reconst
    end

    @testset "makesitediagonal" begin
        L = 2

        sitesx = [Index(2, "x=$n") for n in 1:L]

        Ψ = random_mps(sitesx)

        M = Quantics.makesitediagonal(Ψ, "x")

        Ψ_recost = Array(prod(Ψ), sitesx...)
        M_recost = Array(prod(M), prime.(sitesx)..., sitesx...)

        for i in 1:2, i2 in 1:2, j in 1:2, j2 in 1:2
            if i != i2 || j != j2
                @test M_recost[i, j, i2, j2] ≈ 0.0
            else
                @test M_recost[i, j, i2, j2] ≈ Ψ_recost[i, j]
            end
        end
    end

    @testset "SubDomainMPS" begin
        @testset "rearrange_siteinds" begin
            N = 3
            sitesx = [Index(2, "x=$n") for n in 1:N]
            sitesy = [Index(2, "y=$n") for n in 1:N]
            sitesz = [Index(2, "z=$n") for n in 1:N]
            sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

            Ψ = MPS(collect(_random_mpo(sites)))

            prjΨ = SubDomainMPS(Ψ)
            prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

            sitesxy = collect(collect.(zip(sitesx, sitesy)))
            sites_rearranged = Vector{Index{Int}}[]
            for i in 1:N
                push!(sites_rearranged, sitesxy[i])
                push!(sites_rearranged, [sitesz[i]])
            end
            prjΨ1_rearranged = Quantics.rearrange_siteinds(prjΨ1, sites_rearranged)

            @test reduce(*, MPS(prjΨ1)) ≈ reduce(*, MPS(prjΨ1_rearranged))
            @test PartitionedMPSs.siteinds(prjΨ1_rearranged) == sites_rearranged
        end

        @testset "makesitediagonal and extractdiagonal" begin
            N = 3
            sitesx = [Index(2, "x=$n") for n in 1:N]
            sitesy = [Index(2, "y=$n") for n in 1:N]
            sitesz = [Index(2, "z=$n") for n in 1:N]

            sitesxy_vec = [[x, y] for (x, y) in zip(sitesx, sitesy)]
            sitesz_vec = [[z] for z in sitesz]
            sites = [x for pair in zip(sitesxy_vec, sitesz_vec) for x in pair]

            Ψ = MPS(collect(_random_mpo(sites)))

            prjΨ = SubDomainMPS(Ψ)
            prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

            prjΨ1_diagonalz = Quantics.makesitediagonal(prjΨ1, "y")
            sites_diagonalz = Iterators.flatten(siteinds(prjΨ1_diagonalz))

            psi_diag = prod(prjΨ1_diagonalz.data)
            psi = prod(prjΨ1.data)

            @test Quantics.extractdiagonal(prjΨ1_diagonalz, "y") ≈ prjΨ1

            for indval in eachindval(sites_diagonalz...)
                ind = first.(indval)
                val = last.(indval)

                index_dict = Dict{Index{Int},Vector{Int}}()
                for (i, el) in enumerate(ind)
                    baseind = noprime(el)
                    if haskey(index_dict, baseind)
                        push!(index_dict[baseind], i)
                    else
                        index_dict[baseind] = [i]
                    end
                end
                repeated_indices = [is for is in values(index_dict) if length(is) > 1]

                isdiagonalelement = all(allequal(val[i] for i in is)
                for is in repeated_indices)

                if isdiagonalelement
                    nondiaginds = unique(noprime(i) => v for (i, v) in indval)
                    @test psi_diag[indval...] == psi[nondiaginds...]
                else
                    @test iszero(psi_diag[indval...])
                end
            end
        end
    end
end
