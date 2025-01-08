@doc raw"""
  ``a  x + b  y``, where ``a = 0, \pm 1`` and ``b = 0, \pm 1`` (``a + b \neq -2``).

```
         out
          |
       --------
 cin --|  T   |-- cout
       --------
        |    |
        x    y
```

  ``T_{x, y, \mathrm{out}, \mathrm{cin}, \mathrm{cout}} = 1`` if ``a  x + b  y + \mathrm{cin} = \mathrm{cout}``, ``=0`` otherwise (`out` is the output bit).
"""
function _binaryop_tensor(a::Int, b::Int, site_x::Index{T}, site_y::Index{T},
        site_out::Index{T},
        cin_on::Bool, cout_on::Bool, bc::Int) where {T}
    abs(a) <= 1 || error("a must be either 0, 1, -1")
    abs(b) <= 1 || error("b must be either 0, 1, -1")
    abs(bc) == 1 || error("bc must be either 1, -1")
    a + b != -2 || error("a = -1 and b = -1 not supported")

    cins = cin_on ? [-1, 0, 1] : [0]
    cinsize = length(cins)
    coutsize = cout_on ? 3 : 1
    tensor = zeros(Float64, (cinsize, coutsize, 2, 2, 2))
    for (idx_cin, cin) in enumerate(cins), y in 0:1, x in 0:1
        res = a * x + b * y + cin
        if res >= 0
            cout = _getbit(abs(res), 1)
        else
            cout = -1
        end
        if cout_on
            tensor[idx_cin, cout + 2, x + 1, y + 1, (abs(res) & 1) + 1] = 1
        else
            tensor[idx_cin, 1, x + 1, y + 1, (abs(res) & 1) + 1] = (cout == 0 ? 1 : bc)
        end
    end
    link_in = Index(cinsize, "link_in")
    link_out = Index(coutsize, "link_out")
    return ITensor(tensor, [link_in, link_out, site_x, site_y, site_out]), link_in, link_out
end

"""
Create a tensor acting on a vector of sites.
"""
function binaryop_tensor_multisite(sites::Vector{Index{T}},
        coeffs::Vector{Tuple{Int,Int}},
        pos_sites_in::Vector{Tuple{Int,Int}},
        cin_on::Bool,
        cout_on::Bool,
        bc::Vector{Int}) where {T<:Number}

    # Check
    sites = noprime.(sites)
    nsites = length(sites)
    length(coeffs) == nsites || error("Length of coeffs does not match that of coeffs")
    length(pos_sites_in) == nsites ||
        error("Length of pos_sites_in does not match that of coeffs")

    sites_in = [Index(2, "site_dummy,n=$n") for n in eachindex(sites)]
    links_in = Index{T}[]
    links_out = Index{T}[]

    # First, we need to know the number of dummny indices for each site.
    ndumnyinds = zeros(Int, nsites)
    for n in 1:nsites
        for s in pos_sites_in[n]
            ndumnyinds[s] += 1
        end
    end

    res = ITensor(1)

    for n in 1:nsites
        res *= dense(delta(sites[n],
            [setprime(sites_in[n], plev) for plev in 1:ndumnyinds[n]]))
    end

    currentdummyinds = ones(Int, nsites)
    for n in 1:nsites
        sites_ab = setprime(sites_in[pos_sites_in[n][1]],
            currentdummyinds[pos_sites_in[n][1]]),
        setprime(sites_in[pos_sites_in[n][2]],
            currentdummyinds[pos_sites_in[n][2]])
        for i in 1:2
            currentdummyinds[pos_sites_in[n][i]] += 1
        end
        t, lin, lout = _binaryop_tensor(coeffs[n]..., sites_ab..., sites[n]',
            cin_on, cout_on, bc[n])
        push!(links_in, lin)
        push!(links_out, lout)
        res *= t
    end

    linkin = Index(prod(dim.(links_in)), "linkin")
    linkout = Index(prod(dim.(links_out)), "linkout")
    res = permute(res, [links_in..., links_out..., prime.(sites)..., sites...])
    # Here, we swap sites and prime(sites)!
    res = ITensor(ITensors.data(res), [linkin, linkout, sites..., prime.(sites)...])
    return res
end

"""
Construct an MPO representing a selector associated with binary operations.

We describe the functionality for length(coeffs) = 2 (nsites_bop).
In this case, site indices are split into a list of chuncks of nsites_bop sites.

Binary operations are applied to each chunck and the direction of carry is forward (rev_carrydirec=true)
or backward (rev_carrydirec=false).

Assumed rev_carrydirec = true, we consider a two-variable g(x, y), which is quantized as
x = (x_1 ... x_R)_2, y = (y_1 ... y_R)_2.

We now define a new function by binary operations as
f(x, y) = g(a * x + b * y + s1, c * x + d * y + s2),
where a, b, c, d = +/- 1, 0, and s1, s1 are arbitrary integers.

`bc` is a vector of boundary conditions for each arguments of `g` (not of `f`).
"""
function affinetransform(M::MPS,
        tags::AbstractVector{String},
        coeffs_dic::AbstractVector{Dict{String,Int}},
        shift::AbstractVector{Int},
        bc::AbstractVector{Int};
        kwargs...)
    transformer = affinetransformmpo(siteinds(M), tags, coeffs_dic, shift, bc)
    return _apply(transformer, M; kwargs...)
end

function affinetransformmpo(sites::AbstractVector{Index{T}},
        tags::AbstractVector{String},
        coeffs_dic::AbstractVector{Dict{String,Int}},
        shift::AbstractVector{Int},
        bc::AbstractVector{Int})::MPO where {T}
    # f(x, y) = g(a * x + b * y + s1, c * x + d * y + s2)
    #         = h(a * x + b * y,      c * x + d * y),
    # where h(x, y) = g(x + s1, y + s2).
    # The transformation is executed in this order: g -> h -> f.

    mpos = MPO[]

    # Number of variables involved in transformation
    ntransvars = length(tags)

    2 <= ntransvars ||
        error("Number of variables for transformation must be greater than or equal to 2.")

    sites_for_tag = []
    for tag in tags
        push!(sites_for_tag, findallsites_by_tag(sites; tag=tag))
        if length(sites_for_tag[end]) == 0
            error("Tag $tag is not found.")
        end
    end

    length(unique(map(length, sites_for_tag))) == 1 ||
        error("Number of sites for each tag must be equal.")

    length(shift) == ntransvars || error("Length of shift must be equal to that of tags.")

    # If shift is required
    if !all(shift .== 0)
        for i in 1:ntransvars
            push!(mpos, shiftaxismpo(sites, shift[i]; tag=tags[i], bc=bc[i]))
        end
    end

    # Followed by a rotation
    push!(mpos, affinetransformmpo(sites, tags, coeffs_dic, bc))

    # Contract MPOs
    res = mpos[1]
    for n in 2:length(mpos)
        res = _apply(mpos[n], res; cutoff=1e-25, maxdim=typemax(Int))
    end

    return res
end

"""
Affine transform of a MPS with no shift
Significant bits are assumed to be aligned from left to right for all tags.
"""
function affinetransform(M::MPS,
        tags::AbstractVector{String},
        coeffs_dic::AbstractVector{Dict{String,Int}},
        bc::AbstractVector{Int};
        kwargs...)
    transformer = affinetransformmpo(siteinds(M), tags, coeffs_dic, bc)
    return _apply(transformer, M; kwargs...)
end

"""
Generate an MPO representing an affine transform of a MPS with no shift
Significant bits are assumed to be aligned from left to right for all tags.
"""
function affinetransformmpo(sites::AbstractVector{Index{T}},
        tags::AbstractVector{String},
        coeffs_dic::AbstractVector{Dict{String,Int}},
        bc::AbstractVector{Int})::MPO where {T}
    mpos = MPO[]

    # f(x, y) = g(a * x + b * y + s1, c * x + d * y + s2)
    #         = h(a * x + b * y,      c * x + d * y),
    # where h(x, y) = g(x + s1, y + s2).
    # The transformation is taken place in this order: g -> h -> f.

    # Number of variables involved in transformation
    ntransvars = length(tags)

    2 <= ntransvars ||
        error("Number of variables for transformation must be greater than or equal to 2.")

    sites_for_tag = []
    for tag in tags
        push!(sites_for_tag, findallsites_by_tag(sites; tag=tag))
        if length(sites_for_tag[end]) == 0
            error("Tag $tag is not found.")
        end
    end

    length(unique(map(length, sites_for_tag))) == 1 ||
        error("Number of sites for each tag must be equal.")

    tags_to_pos = Dict(tag => i for (i, tag) in enumerate(tags))

    all([length(c) == 2 for c in coeffs_dic]) ||
        error("Length of each element in coeffs_dic must be 2")

    coeffs = Tuple{Int,Int}[]
    pos_sites_in = Tuple{Int,Int}[]
    for inewval in 1:ntransvars
        length(coeffs_dic[inewval]) == 2 ||
            error("Length of each element in coeffs_dic must be 2: $(coeffs_dic[inewval])")
        pos_sites_in_ = [tags_to_pos[t] for (t, c) in coeffs_dic[inewval]]
        length(unique(pos_sites_in_)) == 2 ||
            error("Each element of pos_sites_in must contain two different values: $(pos_sites_in_)")
        all(pos_sites_in_ .>= 0) || error("Invalid tag: $(coeffs_dic[inewval])")

        push!(pos_sites_in, Tuple(pos_sites_in_))
        push!(coeffs, Tuple([c for (t, c) in coeffs_dic[inewval]]))
    end

    length(tags) == ntransvars || error("Length of tags does not match that of coeffs")
    length(pos_sites_in) == ntransvars ||
        error("Length of pos_sites_in does not match that of coeffs")

    # Check if the order of significant bits is consistent among all tags
    rev_carrydirecs = Bool[]
    pos_for_tags = []
    sites_for_tags = []
    for i in 1:ntransvars
        push!(sites_for_tags, findallsiteinds_by_tag(sites; tag=tags[i]))
        pos_for_tag = findallsites_by_tag(sites; tag=tags[i])
        push!(rev_carrydirecs, isascendingorder(pos_for_tag))
        push!(pos_for_tags, pos_for_tag)
    end

    valid_rev_carrydirecs = all(rev_carrydirecs .== true) || all(rev_carrydirecs .== false)
    valid_rev_carrydirecs ||
        error("The order of significant bits must be consistent among all tags!")

    #all(rev_carrydirecs .== true) ||
    #error("Significant bits are aligned from left to right for all tags!")

    length(unique([length(s) for s in sites_for_tags])) == 1 ||
        error("The number of sites for each tag must be the same! $([length(s) for s in sites_for_tags])")

    rev_carrydirec = all(rev_carrydirecs .== true) # If true, significant bits are at the left end.

    if !rev_carrydirec
        transformer_ = affinetransformmpo(
            reverse(sites), reverse(tags), reverse(coeffs_dic), reverse(bc))
        return MPO([transformer_[n] for n in reverse(1:length(transformer_))])
    end

    # First check transformations with -1 and -1; e.g., (a, b) = (-1, -1)
    # These transformations are not supported in the backend.
    # To support this case, we need to flip the sign of coeffs as follows:
    #  f(x, y) = h(x + y, c * x + d * y) = g(- x -y, c * x + d * y),
    # where h(x, y) = g(-x, y).
    # The transformation is taken place in this order: g -> h -> f.
    sign_flips = [coeffs[n][1] == -1 && coeffs[n][2] == -1 for n in eachindex(coeffs)]

    for v in 1:ntransvars
        if sign_flips[v]
            push!(mpos, bc[v] * reverseaxismpo(sites; tag=tags[v], bc=bc[v]))
        end
    end

    # Apply binary operations (nomore (-1, -1) coefficients)
    coeffs_positive = [(sign_flips[n] ? abs.(coeffs[n]) : coeffs[n])
                       for n in eachindex(coeffs)]
    sites_mpo = collect(Iterators.flatten(Iterators.zip(sites_for_tags...)))
    transformer = _binaryop_mpo(sites_mpo, coeffs_positive, pos_sites_in;
        rev_carrydirec=true, bc=bc)
    transformer = matchsiteinds(transformer, sites)

    push!(mpos, transformer)

    # Contract MPOs
    res = mpos[1]
    for n in 2:length(mpos)
        res = _apply(mpos[n], res; cutoff=1e-25, maxdim=typemax(Int))
    end

    return res
end

"""
Construct an MPO representing a selector associated with binary operations.

We describe the functionality for length(coeffs) = 2 (nsites_bop).
In this case, site indices are split into a list of chuncks of nsites_bop sites.

Binary operations are applied to each chunck and the direction of carry is forward (rev_carrydirec=true)
or backward (rev_carrydirec=false).

Assumed rev_carrydirec = true, we consider a two-variable g(x, y), which is quantized as
x = (x_1 ... x_R)_2, y = (y_1 ... y_R)_2.

We now define a new function by binary operations as
f(x, y) = g(a * x + b * y, c * x + d * y),
where a, b, c, d = +/- 1, 0, and s1, s1 are arbitrary integers.

The transform from `g` to `f` can be represented as an MPO:
f(x_1, y_1, ..., x_R, y_R) = M(x_1, y_1, ...; x'_1, y'_1, ...) f(x'_1, y'_1, ..., x'_R, y'_R).

The MPO `M` acts a selector: The MPO selects values from `f` to form `g`.

For rev_carrydirec = false, the returned MPO represents
f(x_R, y_R, ..., x_1, y_1) = M(x_R, y_R, ...; x'_R, y'_R, ...) f(x'_R, y'_R, ..., x'_1, y'_1).

`bc` is a vector of boundary conditions for each arguments of `g` (not of `f`).
"""
function _binaryop_mpo(sites::Vector{Index{T}},
        coeffs::Vector{Tuple{Int,Int}},
        pos_sites_in::Vector{Tuple{Int,Int}};
        rev_carrydirec=false,
        bc::Union{Nothing,Vector{Int}}=nothing) where {T<:Number}
    # Number of variables involved in transformation
    nsites_bop = length(coeffs)

    if bc === nothing
        bc = ones(Int64, nsites_bop) # Default: periodic boundary condition
    end

    # First check transformations with -1 and -1; e.g., (a, b) = (-1, -1)
    # These transformations are not supported in _binaryop_mpo_backend.
    # To support this case, we need to flip the sign of coeffs as follows:
    #  f(x, y) = h(x + y, c * x + d * y) = g(-x-y, c * x + d * y),
    # where h(x, y) = g(-x, y).
    # The transformation is taken place in this order: g -> h -> f.
    sign_flips = [coeffs[n][1] == -1 && coeffs[n][2] == -1 for n in 1:length(coeffs)]
    coeffs_ = [(sign_flips[i] ? abs.(coeffs[i]) : coeffs[i]) for i in eachindex(coeffs)]

    # For g->h
    M = _binaryop_mpo_backend(sites, coeffs_, pos_sites_in; rev_carrydirec=rev_carrydirec,
        bc=bc)

    # For h->f
    for i in 1:nsites_bop
        if !sign_flips[i]
            continue
        end
        M_ = bc[i] *
             flipop(sites[i:nsites_bop:end]; rev_carrydirec=rev_carrydirec, bc=bc[i])
        M = _apply(M, matchsiteinds(M_, sites); cutoff=1e-25)
    end

    return M
end

# Limitation: a = -1 and b = -1 not supported. The same applies to (c, d).
function _binaryop_mpo_backend(sites::Vector{Index{T}},
        coeffs::Vector{Tuple{Int,Int}},
        pos_sites_in::Vector{Tuple{Int,Int}};
        rev_carrydirec=false,
        bc::Union{Nothing,Vector{Int}}=nothing) where {T<:Number}
    nsites = length(sites)
    nsites_bop = length(coeffs)
    ncsites = nsites ÷ nsites_bop
    length(pos_sites_in) == nsites_bop ||
        error("Length mismatch between coeffs and pos_sites_in")
    if bc === nothing
        bc = ones(Int64, nsites_bop) # Default: periodic boundary condition
    end

    links = [Index(3^nsites_bop, "link=$n") for n in 0:ncsites]
    links[1] = Index(1, "link=0")
    links[end] = Index(1, "link=$ncsites")

    tensors = ITensor[]
    sites2d = reshape(sites, nsites_bop, ncsites)
    for n in 1:ncsites
        sites_ = sites2d[:, n]
        cin_on = rev_carrydirec ? (n != ncsites) : (n != 1)
        cout_on = rev_carrydirec ? (n != 1) : (n != ncsites)
        tensor = binaryop_tensor_multisite(sites_,
            coeffs,
            pos_sites_in,
            cin_on,
            cout_on,
            bc)
        lleft, lright = links[n], links[n + 1]
        if rev_carrydirec
            replaceind!(tensor, firstind(tensor, "linkout") => lleft)
            replaceind!(tensor, firstind(tensor, "linkin") => lright)
        else
            replaceind!(tensor, firstind(tensor, "linkin") => lleft)
            replaceind!(tensor, firstind(tensor, "linkout") => lright)
        end

        inds_list = [[lleft, sites_[1]', sites_[1]]]
        for m in 2:(nsites_bop - 1)
            push!(inds_list, [sites_[m]', sites_[m]])
        end
        push!(inds_list, [lright, sites_[nsites_bop]', sites_[nsites_bop]])

        tensors = vcat(tensors, split_tensor(tensor, inds_list))
    end

    _removeedges!(tensors, sites)

    M = truncate(MPO(tensors); cutoff=1e-25)
    cleanup_linkinds!(M)

    return M
end

"""
For given function `g(x)` and shift `s`, construct an MPO representing `f(x) = g(x + s)`.
x: 0, ..., 2^R - 1
0 <= s <= 2^R - 1

We assume that left site indices correspond to significant digits
"""
function _shift_mpo(sites::Vector{Index{T}}, shift::Int; bc::Int=1) where {T<:Number}
    R = length(sites)
    0 <= shift <= 2^R - 1 || error("Invalid shift")

    ys = Quantics.tobin(shift, R)

    links = Index{T}[]
    tensors = ITensor[]

    for n in 1:R
        cin_on = n != R
        cout_on = n != 1
        sitey = Index(2, "Qubit, y")
        t, link_in, link_out = Quantics._binaryop_tensor(1, 1, sites[n]', sitey, sites[n],
            cin_on, cout_on, bc)
        t *= onehot(sitey => ys[n] + 1)
        if n < R
            push!(links, Index(dim(link_in), "link=$n"))
            replaceind!(t, link_in => links[end])
        end
        if n > 1
            replaceind!(t, link_out => links[n - 1])
        end
        if n == 1
            t *= onehot(link_out => 1)
        elseif n == R
            t *= onehot(link_in => 1)
        end

        push!(tensors, t)
    end

    MPO(tensors)
end
