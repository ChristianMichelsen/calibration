using FITSIO
using DataFrames
using CairoMakie
using LsqFit
using Peaks
using OrderedCollections


include("utils.jl")
include("plots.jl")

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

order = 10

function get_df_fits(df, orders, spectrum_min = 0.2)

    df_fits = DataFrame[]

    for order = orders[1]:orders[end]

        spectrum_all = df[order, :spectrum]
        uncertainty_all = df[order, :uncertainty]
        λ_approx_all = df[order, :wavelength]
        x_all = range(1, length(spectrum_all))
        # plot_order(x_all, spectrum_all, λ_approx_all)

        nan_left, nan_right = get_not_nan_idx(spectrum_all)
        mask = nan_left:nan_right

        spectrum = spectrum_all[mask]
        uncertainty = uncertainty_all[mask]
        λ_approx = λ_approx_all[mask]
        x = x_all[mask]
        # plot_order(x, spectrum, λ_approx)

        if !any(spectrum_min .< spectrum)
            continue
        end

        #%%

        # println(order)
        sections = get_sections(spectrum, w = 7)
        df_fit = make_fits(x, spectrum, uncertainty, λ_approx, sections)
        # N_not_converged = sum(.!df_fit.converged)
        # N_not_valid = sum(.!df_fit.valid)
        # N_not_converged + N_not_valid
        # sum(df_fit.converged .&& df_fit.valid)
        # println(df_fit)

        df_fit[!, :order] .= order

        push!(df_fits, df_fit)
    end

    return vcat(df_fits...)
end


function get_df_fits(df)
    orders = [1, nrow(df)]
    return get_df_fits(df, orders)
end



df_fits = get_df_fits(df);

mask_good = df_fits.converged .&& df_fits.valid;
df_fits_good = df_fits[mask_good, :];

#%%

f_fit_coefficients_good = plot_fit_coefficients(df_fits_good)

if savefig
    save(
        "figures/fit_coefficients_good__$(basename(filename)).pdf",
        f_fit_coefficients_good,
        pt_per_unit = 1,
    )
end
#%%

f_calibration = get_f_calibration(df_fits_good)
f_calibration

if savefig
    save("figures/calibration__$(basename(filename)).pdf", f_calibration, pt_per_unit = 1)
end