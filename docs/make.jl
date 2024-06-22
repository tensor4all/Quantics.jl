using Quantics
using Documenter

DocMeta.setdocmeta!(Quantics, :DocTestSetup, :(using Quantics); recursive=true)

makedocs(;
    modules=[Quantics],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="Quantics.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/Quantics.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md"
    ])

deploydocs(;
    repo="github.com/tensor4all/Quantics.jl.git",
    devbranch="main"
)
