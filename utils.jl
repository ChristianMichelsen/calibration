using StatsBase
using LinearAlgebra: diag, LAPACK
using Parameters
using UnPack
using ColorSchemes
using FFTW

#%%

const c_light = 299792458 # u"m/s"
const ν_rep = 14e9 # Hz
const ν_offset = 6.19e9 # Hz


function get_data(filename)
    f = FITS(filename)
    df = DataFrame(f[2])
    close(f)
    return df
end

#%%


function get_not_nan_idx(x)
    idx_left = argmax(.!isnan.(x))
    idx_right = length(x) - argmax(.!isnan.(reverse(x))) + 1
    return idx_left, idx_right
end



#%%


function make_image(df)

    orders = 1:nrow(df)

    img = zeros(nrow(df), length(df[1, :spectrum]))
    for (i, order) in enumerate(orders)
        img[i, :] = df[order, :spectrum]
    end
    return img
end

#%%


function StatsBase.cov2cor(X::Matrix)
    return cov2cor(X, sqrt.(diag(X)))
end



#%%


function get_sections(spectrum; w = 7)
    peak_ids = findmaxima(spectrum, w)[1]
    half_width = floor.(Int, diff(peak_ids) / 2)
    center_edges = peak_ids[1:end-1] .+ half_width
    left_edge = peak_ids[1] - half_width[1]
    if left_edge < 1
        left_edge = 1
    end
    right_edge = peak_ids[end] + half_width[end]
    if right_edge > length(spectrum)
        right_edge = length(spectrum)
    end
    sections = [left_edge, center_edges..., right_edge]
    return sections
end


#%%


@with_kw struct Data{T,S,R,U}
    x::T
    y::S
    σ::R
    λ_approx::U
end


@with_kw mutable struct FitObject
    name::String
    data::Data
    func::Function
    p0::Vector{<:Number}
    fitresult::Union{Missing,LsqFit.LsqFitResult} = missing
end


function fit!(fit::FitObject)
    @unpack x, y, σ, λ_approx = fit.data
    w = 1 ./ σ .^ 2
    # lb = [0, minimum(x), 0.01, 1.0, 0.0, -10]
    # ub = [10, maximum(x), 10.00, 10.00, 10.0, 10]
    # fitresult = curve_fit(fit.func, x, y, w, fit.p0, autodiff = :forwarddiff, lower=lb, upper=ub)
    fitresult = curve_fit(fit.func, x, y, w, fit.p0, autodiff = :forwarddiff)
    fit.fitresult = fitresult
end


function residuals(fit::FitObject)
    return fit.fitresult.resid
end

function coefficients(fit::FitObject)
    return coef(fit.fitresult)
end


function degrees_of_freedom(fit::FitObject)
    return dof(fit.fitresult)
end

function get_standard_deviations(fit::FitObject)
    return stderror(fit.fitresult)
end

function get_λ_true(fit::FitObject)
    λ_approx_μ = get_λ_approx_μ_SI(fit.data.x, fit.data.λ_approx, fit)
    return get_λ_true_from_λ_approx(λ_approx_μ)
end



function is_singular(J::Matrix{Float64})
    return LAPACK.getrf!(J' * J)[3] != 0
end


function is_singular(fit::FitObject)
    J = fit.fitresult.jacobian
    return is_singular(J)
end


function is_converged(fit::FitObject)
    if !fit.fitresult.converged
        return false
    end

    if is_singular(fit)
        return false
    end

    covar = covariance(fit)
    vars = diag(covar)
    vratio = minimum(vars) / maximum(vars)

    is_0 = isapprox(vratio, 0.0, atol = 0, rtol = Base.rtoldefault(vratio, 0.0, 0))
    is_negative = vratio < 0.0
    if is_0 || is_negative
        return false
    end

    return true
end


function is_valid(fit::FitObject)

    coefs = coefficients(fit)
    A, μ, σ, P, c, b = coefs

    if !(0.1 < A < 5)
        return false
    end

    if !(minimum(x) < μ < maximum(x))
        return false
    end

    if !(1.1 < σ < 3)
        return false
    end

    if !(0.75 < P < 2)
        return false
    end

    if !(0 < c < 5)
        return false
    end

    if !(-1 < b < 1)
        return false
    end

    return true
end


# sum(fit.resid .^ 2)
# coef(fit)
# dof(fit)
# # We can estimate errors on the fit parameters,
# # to get standard error of each parameter:
# sigma = stderror(fit)
# # to get margin of error and confidence interval of each parameter at 5% significance level:
# # margin_of_error = margin_error(fit, 0.05)
# confidence_inter = confidence_interval(fit, 0.05)


function covariance(fit::FitObject)
    return estimate_covar(fit.fitresult)
end


function correlation(fit::FitObject)
    C = estimate_covar(fit.fitresult)
    return cov2cor(C)
end

function correlation(X::Matrix)
    return cov2cor(X)
end


#%%

function compute_χ²(fit::FitObject)
    return sum(residuals(fit) .^ 2)
end


function print_χ²(fit::FitObject, digits = 1)
    χ² = round(compute_χ²(fit), digits = digits)
    println("χ² $(fit.name): $(χ²)")
end

#%%


function get_λ_approx_μ(x, λ_approx, fit)
    μ_fit = coefficients(fit)[2]
    idx = argmin(abs.(x .- μ_fit))
    λ_tmp = λ_approx[idx]
    return λ_tmp
end


function get_λ_approx_μ_SI(x, λ_approx, fit)
    return get_λ_approx_μ(x, λ_approx, fit) * 1e-10
end


function find_n(λ)
    n = (c_light / λ - ν_offset) / ν_rep
    n_int = round(Int, n)
    # TODO Insert check
    return n_int
end

function λ2ν(λ)
    c_light / λ
end

function ν2λ(ν)
    return c_light / ν
end

function νₙ(n)
    ν = ν_rep * n + ν_offset
    return ν
end

function n2λ(n)
    ν = νₙ(n)
    λ = ν2λ(ν)
    return λ
end


function get_λ_true_from_λ_approx(λ_approx)
    n = find_n(λ_approx)
    λ_true = n2λ(n)
    return λ_true
end



function make_fits(x, spectrum, uncertainty, λ_approx, sections)

    chi2s = Float64[]
    coefs = Vector[]
    λ_true = Float64[]
    converged = Bool[]
    valid = Bool[]
    standard_deviations = Vector[]

    for (low, high) in zip(sections[1:end-1], sections[2:end])

        # @show low, high

        mask_section = low:high

        data = Data(
            x = x[mask_section],
            y = spectrum[mask_section],
            σ = uncertainty[mask_section],
            λ_approx = λ_approx[mask_section],
        )

        fit_supergauss_ext = FitObject(
            name = "SuperGauss Ext",
            data = data,
            func = SuperGauss_vec,
            p0 = [1.0, mean(data.x), 1.0, 1.0, 0, 0],
        )

        fit!(fit_supergauss_ext)

        push!(chi2s, compute_χ²(fit_supergauss_ext))
        push!(coefs, coefficients(fit_supergauss_ext))
        push!(λ_true, get_λ_true(fit_supergauss_ext))
        push!(converged, is_converged(fit_supergauss_ext))
        push!(valid, is_valid(fit_supergauss_ext))

        if converged[end] && valid[end]
            push!(standard_deviations, get_standard_deviations(fit_supergauss_ext))
        else
            push!(standard_deviations, fill(NaN, length(fit_supergauss_ext.p0)))
        end
    end

    df_fit = DataFrame(permutedims(hcat(coefs...)), [:A, :μ, :σ, :P, :c, :b])
    df_fit[!, "χ²"] = chi2s
    df_fit[!, "λ_true"] = λ_true
    df_fit[!, "converged"] = converged
    df_fit[!, "valid"] = valid

    df_errors = DataFrame(
        permutedims(hcat(standard_deviations...)),
        [:std_A, :std_μ, :std_σ, :std_P, :std_c, :std_b],
    )

    return hcat(df_fit, df_errors)
end



#%%

function get_df_coefficients(fit::FitObject, d_p0)
    df_coefficients = DataFrame(keys(d_p0) .=> coefficients(fit))
    push!(df_coefficients, get_standard_deviations(fit))
    push!(df_coefficients, abs.(coefficients(fit) ./ get_standard_deviations(fit)))
    return df_coefficients
end



#%%


# let y=spectrum
#     # pks, vals = findmaxima(y)
#     pks, vals = findmaxima(y, 5)
#     pks, proms = peakproms(pks, y)
#     pks, widths, leftedge, rightedge = peakwidths(pks, y, proms)
#     return widths
# end


#%%

function Gauss(x, A, μ, σ, c)
    z = (x - μ)^2 / (2 * σ^2)
    return A * exp(-z) + c
end
@. Gauss_vec(x, p) = Gauss(x, p...)


function SuperGaussSimple(x, A, μ, σ, P)
    z = (x - μ)^2 / (2 * σ^2)
    return A * exp(-z^P)
end

function SuperGauss(x, A, μ, σ, P, c, b = 0)
    z = (x - μ)^2 / (2 * σ^2)
    return A * exp(-z^P) + c + b * (x - μ)
end
@. SuperGauss_vec(x, p) = SuperGauss(x, p...)

#%%

if false


    section_id = 430
    # section_id = 9
    mask_section = sections[section_id]:sections[section_id+1]
    plot_order(x[mask_section], spectrum[mask_section], λ_approx[mask_section])

    data = Data(
        x = x[mask_section],
        y = spectrum[mask_section],
        σ = uncertainty[mask_section],
        λ_approx = λ_approx[mask_section],
    )


    fit_gauss = FitObject(
        name = "Gauss",
        data = data,
        func = Gauss_vec,
        p0 = [1.0, mean(data.x), 1.0, 0.0],
    )
    fit!(fit_gauss)
    print_χ²(fit_gauss)
    coefficients(fit_gauss)
    covariance(fit_gauss)
    correlation(fit_gauss)


    fit_supergauss = FitObject(
        name = "SuperGauss",
        data = data,
        func = SuperGauss_vec,
        p0 = [1.0, mean(data.x), 1.0, 1.0, 0],
    )
    fit!(fit_supergauss)
    print_χ²(fit_supergauss)
    compute_χ²(fit_supergauss)
    coefficients(fit_supergauss)


    fit_supergauss_ext = FitObject(
        name = "SuperGauss Ext",
        data = data,
        func = SuperGauss_vec,
        p0 = [1.0, mean(data.x), 1.0, 1.0, 0, 0],
    )
    fit!(fit_supergauss_ext)
    print_χ²(fit_supergauss_ext)
    compute_χ²(fit_supergauss_ext)
    coefficients(fit_supergauss_ext)
    get_λ_true(fit_supergauss_ext) * 1e10

    #%%

    f_fits = plot([fit_gauss, fit_supergauss, fit_supergauss_ext])
    f_residuals = plot_residuals([fit_gauss, fit_supergauss, fit_supergauss_ext])

end

#%%

# plot_fourier(spectrum, "Spectrum", 100)

# #%%

# f_Δμ = Figure(resolution = (1200, 400));
# ax_Δμ = Axis(f_Δμ[1, 1], title = "Δμ", xlabel = "Fit Index", ylabel = "Δμ")
# lines!(ax_Δμ, diff(df_fit[!, :μ]))
# f_Δμ

# #%%

# display(plot_fourier(df_fit[!, "A"], "A", 5))
# display(plot_fourier(df_fit[!, "c"], "c", 5))

# for name in names(df_fit)
#     signal = df_fit[!, name]
#     display(plot_fourier(signal, name, 5))
# end

# #%%

# sections = get_sections(spectrum);
# section_id = 252
# orders = 40
# mask_section = sections[section_id]:sections[section_id+orders];
# # plot_order(x[mask_section], spectrum[mask_section], λ_approx[mask_section])

# data = Data(x = x[mask_section], y = spectrum[mask_section], σ = uncertainty[mask_section]);


# function MultiOrderSuperGauss(
#     x,
#     #
#     A0,
#     # ΔA,
#     # Af,
#     # Aϕ,
#     #
#     μ0,
#     Δμ,
#     δμ,
#     #
#     σ0,
#     # Δσ,
#     #
#     P0,
#     # ΔP,
#     #
#     c0,
#     # Δc,
#     # cf,
#     # cϕ,
#     #
#     orders,
# )

#     x̄ = (x - μ0)


#     # c = c0
#     c = c0 #+ Δc * sin(2π * cf * x̄ + cϕ)

#     res = 0
#     for i = 1:orders
#         j = i - 1

#         # A = A0 + ΔA * sin(2π * Af * j + Aϕ)
#         A = A0
#         # μ = μ0 + Δμ * j
#         μ = μ0 + Δμ * j + δμ * j^2
#         # σ = σ0 + Δσ * j
#         σ = σ0
#         # P = P0 + ΔP * j
#         P = P0

#         res += SuperGaussSimple(x, A, μ, σ, P)
#         # c += Δc * sin(2π * cf * j + cϕ)
#         # c += Δc * sin(2π * cf * x̄ + cϕ)
#     end

#     res += c
#     return res
# end
# @. MultiOrderSuperGauss_vec(x, p) = MultiOrderSuperGauss(x, p..., orders)



# d_p0 = OrderedDict(
#     :A0 => 0.55,
#     # :ΔA => 0.0,
#     # :Af => 0.1,
#     # :Aϕ => 0.0,

#     :μ0 => 3073.0,
#     :Δμ => 13,
#     :δμ => 0.0005,
#     #
#     :σ0 => 1.7,
#     # :Δσ => 0.0005,
#     #
#     :P0 => 1.3,
#     # :ΔP => 0.0005,
#     #
#     :c0 => 0.12,
#     # :Δc => 0.01,
#     # :cf => 0.001,
#     # :cϕ => 0.0,
# );


# fit_gauss = FitObject(
#     name = "Multi Order Super Gauss",
#     data = data,
#     func = MultiOrderSuperGauss_vec,
#     p0 = collect(values(d_p0)),
# );
# fit!(fit_gauss);
# print_χ²(fit_gauss)
# plot(fit_gauss)
# df_coefficients = get_df_coefficients(fit_gauss, d_p0);

# df_coefficients[1, :]
# # select(df_coefficients, Between(:A0, :Aϕ))[1, :]
# # select(df_coefficients, Between(:c0,:cϕ))[1, :]
# # select(df_coefficients, Between(:c0,:cϕ))[1, :]

# # degrees_of_freedom(fit_gauss)


# #%%
