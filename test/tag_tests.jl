@testitem "tag_tests.jl/tag" begin
    using Test
    import Quantics
    using ITensors

    @testset "findallsites_by_tag" for tag in ["x", "y"]
        nbit = 4
        sites = [Index(2, "Qubit,$(tag)=$x") for x in 1:nbit]
        @test Quantics.findallsites_by_tag(sites; tag=tag) == [1, 2, 3, 4]
        @test isempty(Quantics.findallsites_by_tag(sites; tag="notfound"))

        invalid_tag = "$(tag)="
        @test_throws "Invalid tag: $(tag)=" Quantics.findallsites_by_tag(sites,
            tag=invalid_tag)

        invalid_sites = [Index(2, "Qubit,$(tag)=1"), Index(2, "Qubit,$(tag)=1")]
        @test_throws "with $(tag)=1!" Quantics.findallsites_by_tag(invalid_sites, tag=tag)
        @test_throws "Invalid tag: $(tag)=" Quantics.findallsites_by_tag(invalid_sites,
            tag="$(tag)=")
    end

    @testset "findallsiteinds_by_tag" for tag in ["x", "y"]
        nbit = 4
        sites = [Index(2, "Qubit,$(tag)=$x") for x in 1:nbit]
        @test Quantics.findallsiteinds_by_tag(sites; tag=tag) == sites
        @test isempty(Quantics.findallsiteinds_by_tag(sites; tag="notfound"))

        invalid_tag = "$(tag)="
        @test_throws "Invalid tag: $(tag)=" Quantics.findallsiteinds_by_tag(sites,
            tag=invalid_tag)

        invalid_sites = [Index(2, "Qubit,$(tag)=1"), Index(2, "Qubit,$(tag)=1")]
        @test_throws "with $(tag)=1!" Quantics.findallsiteinds_by_tag(invalid_sites,
            tag=tag)
        @test_throws "Invalid tag: $(tag)=" Quantics.findallsiteinds_by_tag(invalid_sites,
            tag="$(tag)=")
    end
end
