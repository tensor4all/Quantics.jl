"""
a * x + b * y, where a = 0/+/-1 and b = 0/+1/-1.
"""
function _binaryop_tensor(a::Int, b::Int, site_x::Index{T}, site_y::Index{T},
                          site_out::Index{T},
                          cin_on::Bool, cout_on::Bool, bc::Int) where {T}
    abs(a) <= 1 || error("a must be either 0, 1, -1")
    abs(b) <= 1 || error("b must be either 0, 1, -1")
    abs(bc) == 1 || error("bc must be either 1, -1")

    cins = cin_on ? [-1, 0, 1] : [0]
    cinsize = length(cins)
    coutsize = cout_on ? 3 : 1
    tensor = Tensor(Float64, cinsize, coutsize, 2, 2, 2)
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

function binaryop_tensor_multisite(sites::Vector{Index{T}},
                                   coeffs::Vector{Tuple{Int,Int}},
                                   pos_sites_in::Vector{Tuple{Int,Int}},
                                   cin_on,
                                   cout_on,
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

    res = ITensor(1)
    for n in 1:nsites
        res *= delta(sites[n], [setprime(sites_in[n], plev) for plev in 1:nsites])
    end
    for n in 1:nsites
        sites_ab = sites_in[pos_sites_in[n][1]], sites_in[pos_sites_in[n][2]]
        sites_ab = setprime(sites_ab, n)
        t, lin, lout = _binaryop_tensor(coeffs[n]..., sites_ab..., sites[n]', cin_on,
                                        cout_on, bc[n])
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
f(x, y) = g(a * x + b * y, c * x + d * y),
where a, b, c, d = +/- 1, 0.

The transform from `g` to `f` can be represented as an MPO:
f(x_1, y_1, ..., x_R, y_R) = M(x_1, y_1, ...; x'_1, y'_1, ...) f(x'_1, y'_1, ..., x'_R, y'_R).

The MPO `M` acts a selector: The MPO selects values from `f` to form `g`.

For rev_carrydirec = false, the returned MPO represents
f(x_R, y_R, ..., x_1, y_1) = M(x_R, y_R, ...; x'_R, y'_R, ...) f(x'_R, y'_R, ..., x'_1, y'_1).
"""
function binaryop_mpo(sites::Vector{Index{T}},
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

    removeedges!(tensors, sites)

    M = truncate(MPO(tensors); cutoff=1e-25)
    cleanup_linkinds!(M)

    return M
end