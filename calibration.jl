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
    get_images(filenames, savefig)
end



filename = "../expres_tp/LFC_200907.1063.fits"
df = get_data(filename)
#%%


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