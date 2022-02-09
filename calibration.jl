using FITSIO
using DataFrames
using CairoMakie
using LsqFit
using Peaks

include("utils.jl")

savefig = false
make_images = false

if make_images

    dir = "../expres_tp/"
    filenames = filter(x -> endswith(x, ".fits"), readdir(dir, join = true))

    for filename in filenames
        println(filename)

        df = get_data(filename)

        f_overlap = plot_overlap(df)
        if savefig
            save("figures/overlap__$(basename(filename)).pdf", f_overlap)
        end

        image = make_image(df)
        f_image = plot_image(image)
        if savefig
            save("figures/image__$(basename(filename)).pdf", f_image, pt_per_unit = 1)
            save("figures/image__$(basename(filename)).png", f_image, px_per_unit = 20)
        end

    end
end



filename = "../expres_tp/LFC_200907.1063.fits"
df = get_data(filename)

order = 45
spectrum_all = df[order, :spectrum];
uncertainty_all = df[order, :uncertainty];
λ_approx_all = df[order, :wavelength];
x_all = range(1, length(spectrum_all));
plot_order(x_all, spectrum_all, λ_approx_all)

nan_left, nan_right = get_not_nan_idx(spectrum_all)
mask = nan_left:nan_right

spectrum = spectrum_all[mask];
uncertainty = uncertainty_all[mask];
λ_approx = λ_approx_all[mask];
x = x_all[mask];

plot_order(x, spectrum, λ_approx)


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

function SuperGauss(x, A, μ, σ, P, c, b = 0)
    z = (x - μ)^2 / (2 * σ^2)
    return A * exp(-z^P) + c + b * (x - μ)
end
@. SuperGauss_vec(x, p) = SuperGauss(x, p...)


sections = get_sections(spectrum);
section_id = 252
mask_section = sections[section_id]:sections[section_id+1];
plot_order(x[mask_section], spectrum[mask_section], λ_approx[mask_section])

data = Data(x = x[mask_section], y = spectrum[mask_section], σ = uncertainty[mask_section]);


fit_gauss = FitObject(
    name = "Gauss",
    data = data,
    func = Gauss_vec,
    p0 = [1.0, mean(data.x), 1.0, 0.0],
)
fit!(fit_gauss);
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
fit!(fit_supergauss);
print_χ²(fit_supergauss)
compute_χ²(fit_supergauss)
coefficients(fit_supergauss);


fit_supergauss_ext = FitObject(
    name = "SuperGauss Ext",
    data = data,
    func = SuperGauss_vec,
    p0 = [1.0, mean(data.x), 1.0, 1.0, 0, 0],
)
fit!(fit_supergauss_ext);
print_χ²(fit_supergauss_ext)
compute_χ²(fit_supergauss_ext)
coefficients(fit_supergauss_ext);

#%%

f_fits = plot([fit_gauss, fit_supergauss, fit_supergauss_ext])
f_residuals = plot_residuals([fit_gauss, fit_supergauss, fit_supergauss_ext])


#%%

fits, chi2s, coefs = make_fits(x, spectrum, uncertainty, sections);

df_fit = DataFrame(permutedims(hcat(coefs...)), [:A, :μ, :σ, :P, :c, :b]);
df_fit[!, "χ²"] = chi2s;
df_fit

#%%

plot_fit_coefficients(df_fit)

#%%

plot_fourier(spectrum, "Spectrum", 100)

#%%

f_Δμ = Figure(resolution = (1200, 400));
ax_Δμ = Axis(f_Δμ[1, 1], title = "Δμ", xlabel = "Fit Index", ylabel = "Δμ")
lines!(ax_Δμ, diff(df_fit[!, :μ]))
f_Δμ

#%%

display(plot_fourier(df_fit[!, "A"], "A", 5))
display(plot_fourier(df_fit[!, "c"], "c", 5))

for name in names(df_fit)
    signal = df_fit[!, name]
    display(plot_fourier(signal, name, 5))
end



# # # convert from wavelength to frequency - GHz  #
# # frequency_lfc = (c / (wlfc * 1e-10)) * 1e-9

# c = 299792458
# function λ2ν_units(λ)
#     (c / (λ * 1e-10)) * 1e-9
# end

# function λ2ν(λ)
#     c / λ
# end

# function ν2λ(ν)
#     c / ν
# end


# λ2ν(6340 * 1e-10)
# ν2λ(νₙ(100)) * 1e10

#%%

# function νₙ(n)
#     νᵣ = 14e9
#     νₒ = 6.19e9
#     return νᵣ * n + νₒ
# end

# νₙ(1)
