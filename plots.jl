
function plot_order(x, spectrum, λ_approx=missing; remove_nans = false)

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
    if !isa(λ_approx, Missing)
        ax2 = Axis(
            f[1, 1],
            xlabel = "Approximative λ",
            limits = (λ_approx[idx_left], λ_approx[idx_right], 0, nothing),
            xticklabelcolor = :grey,
            xlabelcolor = :grey,
            xaxisposition = :top,
        )
    end

    lines!(ax1, x, spectrum)
    if !isa(λ_approx, Missing)
    lines!(ax2, λ_approx, spectrum, linewidth = 0)
        # errorbars!(ax1, x, spectrum, uncertainty)
        hidexdecorations!(ax2, label = false, ticklabels = false, ticks = false)
    end
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

function plot_fit_coefficients(df_fit; include_points = false)

    df_plot = df_fit[!, Between(:A, :b)]

    f = Figure(resolution = (800, 1200))
    axes = [
        Axis(f[i, 1], limits = (0, nrow(df_plot), nothing, nothing)) for
        i = 1:length(names(df_plot))
    ]

    for (ax, name) in zip(axes, names(df_plot))
        ax.ylabel = name

        if include_points
            scatterlines!(ax, df_plot[!, name])
        else
            lines!(ax, df_plot[!, name])
        end

        if name != last(names(df_plot))
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



function get_f_calibration(df; x_multiplier = 1e10)

    x = df[!, :λ_true]
    y = df[!, :μ]
    σy = df[!, :σ]

    x *= x_multiplier

    f = Figure(resolution = (800, 600))
    ax = Axis(
        f[1, 1],
        xlabel = "λ true [Å]",
        ylabel = "Pixel Index",
        title = "Calibration Curves",
        limits = (minimum(x) * 0.99, maximum(x) * 1.01, 0, nothing),
    )
    errorbars!(ax, x * x_multiplier, y, σy, color = :grey) # same low and high error
    scatter!(ax, x, y, markersize = 2, color = :black)
    return f
end
