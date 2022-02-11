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

xxx = xxx

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

coefficient_names = names(df_fits_good[!, Between(:A, :b)])
for name in coefficient_names
    f = Figure(resolution = (1000, 300))
    ax = Axis(f[1, 1], title = name) #, limits = (0, nrow(df_fits_good), nothing, nothing))
    x = df_fits_good[!, :μ]
    y = df_fits_good[!, :order]
    c = df_fits_good[!, name]
    scatter!(ax, x, y, color = c, markersize = 4)
    f
    if savefig
        save(
            "figures/fit_coefficients_good_image_$(basename(filename))__$name.png",
            f,
            px_per_unit = 10,
        )
    end
end


#%%

f_calibration = get_f_calibration(df_fits_good)
f_calibration

if savefig
    save("figures/calibration__$(basename(filename)).pdf", f_calibration, pt_per_unit = 1)
end


#%%

using Arrow
Arrow.write("./data/df_fits.arrow", df_fits);
df2 = DataFrame(Arrow.Table("./data/df_fits.arrow"))

using Parquet
write_parquet("./data/df_fits.parquet", df_fits);
df3 = identity.(DataFrame(read_parquet("./data/df_fits.parquet")))

#%%

# f = Figure()
# ax = Axis(f[1, 1])
# x = df_fits_good[!, :λ_true_peak]
# y = df_fits_good[!, :λ_approx_peak]
# scatter!(ax, x, y, markersize = 4)
# f