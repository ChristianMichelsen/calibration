using StatsBase
using LinearAlgebra: diag
using Parameters
using UnPack
using ColorSchemes
using FFTW



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

function plot_order(x, spectrum, λ_approx; remove_nans = false)

    f = Figure(resolution = (1200, 400))

    if remove_nans
        idx_left, idx_right = get_not_nan_idx(spectrum)
    else
        idx_left, idx_right = 1, length(spectrum)
    end


    ax1 = Axis(
        f[1, 1],
        xlabel = "Pixel Index",
        ylabel = "Spectrum Intensity",
        limits = (x[idx_left], x[idx_right], 0, nothing),
    )
    ax2 = Axis(
        f[1, 1],
        xlabel = "Approximative λ",
        limits = (λ_approx[idx_left], λ_approx[idx_right], 0, nothing),
        xticklabelcolor = :grey,
        xlabelcolor = :grey,
        xaxisposition = :top,
    )

    lines!(ax1, x, spectrum)
    lines!(ax2, λ_approx, spectrum, linewidth = 0)
    # errorbars!(ax1, x, spectrum, uncertainty)
    hidexdecorations!(ax2, label = false, ticklabels = false, ticks = false)

    return f
end

#%%


function plot_overlap(df)

    orders = 1:nrow(df)

    f = Figure(resolution = (1200, 400))
    ax = Axis(f[1, 1], xlabel = "Log Approximative λ", ylabel = "Order")
    for order in orders
        lines!(
            ax,
            log.(df[order, :wavelength]),
            fill(order, length(df[order, :wavelength])),
        )
    end
    return f
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

function plot_image(image)

    f = Figure(resolution = (1000, 300))
    ax = Axis(
        f[1, 1],
        xlabel = "Pixel ID",
        ylabel = "Order",
        # autolimitaspect = 25,
        limits = (0, size(image, 2), 0, size(image, 1)),
    )
    # ax.yreversed = true
    heatmap!(ax, image')
    return f

end


#%%


function StatsBase.cov2cor(X::Matrix)
    return cov2cor(X, sqrt.(diag(X)))
end



#%%


function get_sections(spectrum; w = 3)
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


# let x = x[mask], spectrum = spectrum[mask], λ_approx = λ_approx[mask]
#     pks, vals = findmaxima(spectrum, 2)
#     f = Figure(resolution = (1200, 400))
#     ax1 = Axis(
#         f[1, 1],
#         xlabel = "Pixel Index",
#         ylabel = "Spectrum Intensity",
#         limits = (nothing, nothing, 0, nothing),
#     )
#     lines!(ax1, x, spectrum)
#     scatter!(ax1, x[pks], spectrum[pks], color = :red)
#     f
# end


#%%


@with_kw struct Data{T,S,R}
    x::T
    y::S
    σ::R
end


@with_kw mutable struct FitObject
    name::String
    data::Data
    func::Function
    p0::Vector{<:Number}
    fitresult::Union{Missing,LsqFit.LsqFitResult} = missing
end


function fit!(fitobject::FitObject)
    @unpack x, y, σ = fitobject.data
    w = 1 ./ σ .^ 2
    fitresult = curve_fit(fitobject.func, x, y, w, fitobject.p0, autodiff = :forwarddiff)
    fitobject.fitresult = fitresult
end


function residuals(fitobject::FitObject)
    return fitobject.fitresult.resid
end

function coefficients(fitobject::FitObject)
    return coef(fitobject.fitresult)
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


function get_colors()
    names = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]
    colors = ColorSchemes.Set1_9
    d_colors = Dict(names .=> colors)
    return colors
end


#%%


function plot(fits::Vector{FitObject})

    data = fits[1].data
    xx = range(minimum(data.x), maximum(data.x), 1000)
    colors = get_colors()

    f = Figure(resolution = (1200, 400))
    ax1 = Axis(
        f[1, 1],
        xlabel = "Pixel Index",
        ylabel = "Spectrum Intensity",
        limits = (nothing, nothing, 0, nothing),
    )

    scatter!(ax1, data.x, data.y, label = "Data", color = :black, markersize = 6)
    errorbars!(ax1, data.x, data.y, data.σ, whiskerwidth = 2, color = :black) # same low and high error

    for (fit, color) in zip(fits, colors)
        lines!(ax1, xx, fit.func(xx, coefficients(fit)), label = fit.name, color = color)
    end

    axislegend(ax1, position = :lt)
    return f
end

function plot(fit::FitObject)
    return plot([fit])
end


#%%


function plot_residuals(fits::Vector{FitObject})

    data = fits[1].data
    x = fits[1].data.x
    colors = get_colors()

    f = Figure(resolution = (1200, 400))
    ax = Axis(
        f[1, 1],
        xlabel = "Pixel Index",
        ylabel = "Residuals",
        # limits = (nothing, nothing, 0, nothing),
    )

    for (fit, color) in zip(fits, colors)
        scatter!(ax, x, residuals(fit), label = fit.name, color = color, markersize = 5)
        errorbars!(ax, x, residuals(fit), ones(length(x)), whiskerwidth = 10, color = color)
    end

    axislegend(ax, position = :lt)
    return f
end

function plot_residuals(fit::FitObject)
    return plot_residuals([fit])
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

function make_fits(x, spectrum, uncertainty, sections)

    fits = FitObject[]
    chi2s = Float32[]
    coefs = Vector[]

    for (low, high) in zip(sections[1:end-1], sections[2:end])

        mask_section = low:high

        data = Data(
            x = x[mask_section],
            y = spectrum[mask_section],
            σ = uncertainty[mask_section],
        )

        fit_supergauss_ext = FitObject(
            name = "SuperGauss Ext",
            data = data,
            func = SuperGauss_vec,
            p0 = [1.0, mean(data.x), 1.0, 1.0, 0, 0],
        )

        fit!(fit_supergauss_ext)

        push!(fits, fit_supergauss_ext)
        push!(chi2s, compute_χ²(fit_supergauss_ext))
        push!(coefs, coefficients(fit_supergauss_ext))

    end

    return fits, chi2s, coefs
end



#%%

function plot_fit_coefficients(df_fit; include_points = false)

    f = Figure(resolution = (800, 1200))
    axes = [
        Axis(f[i, 1], limits = (0, nrow(df_fit), nothing, nothing)) for
        i = 1:length(names(df_fit))
    ]

    for (ax, name) in zip(axes, names(df_fit))
        ax.ylabel = name

        if include_points
            scatterlines!(ax, df_fit[!, name])
        else
            lines!(ax, df_fit[!, name])
        end

        if name != last(names(df_fit))
            hidexdecorations!(ax, grid = false)
        end
    end

    axes[end].xlabel = "Fit Index"
    return f

end


#%%

function plot_fourier(signal, name, ymax = nothing)

    Δt = 1

    # Fourier Transform of it
    F = fft(signal) |> fftshift
    freqs = fftfreq(length(signal), 1.0 / Δt) |> fftshift

    # plots
    f = Figure(resolution = (1200, 400))
    ax1 = Axis(f[1, 1], title = name)
    lines!(ax1, signal)
    ax2 = Axis(f[1, 2], title = "Spectrum of $name", limits = (-0.5, 0.5, 0, ymax))
    lines!(ax2, freqs, abs.(F))
    return f
end
