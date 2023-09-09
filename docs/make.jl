using Quantics
using Documenter

DocMeta.setdocmeta!(Quantics, :DocTestSetup, :(using Quantics); recursive=true)

makedocs(;
    modules=[Quantics],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://gitlab.com/tensors4fields/Quantics.jl/blob/{commit}{path}#{line}",
    sitename="Quantics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[]),
    pages=[
        "Home" => "index.md"
    ])
