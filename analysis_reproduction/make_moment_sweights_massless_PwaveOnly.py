import os
import glob
import gc

import numpy as np
import pandas as pd
import zfit

from sweights.experimental import Cows


input_pattern = (
    "./results_cow_leon_PwaveOnly_fixedFirst60k30k_SB2to1_bkg_eff_Ig_mass_degree4_floatAngles_hesse/"
    "coverage_toys/*.h5"
)

output_dir = (
    "/work/submit/xiaot425/IAP2026/"
    "results_moments_leon_massless_PwaveOnly_fixedFirst60k30k_mass_fitD4_2to1/"
    "coverage_toys"
)

os.makedirs(output_dir, exist_ok=True)


def file_number(path):
    return int(
        os.path.splitext(
            os.path.basename(path)
        )[0]
    )


def zfit_pdf_to_callable_1d_for_cows(
    zpdf,
    obs_space,
):
    def wrapped(x):
        arr = np.asarray(
            x,
            dtype=float,
        )

        if arr.ndim == 1:
            values = arr

        elif arr.ndim == 2:
            if arr.shape[0] == 1:
                values = arr[0]

            elif arr.shape[1] == 1:
                values = arr[:, 0]

            else:
                raise ValueError(
                    f"Unexpected shape {arr.shape}"
                )

        else:
            raise ValueError(
                f"Unexpected ndim {arr.ndim}"
            )

        values = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        vals = zpdf.pdf(
            values,
            norm=obs_space,
        ).numpy()

        return np.asarray(
            vals,
            dtype=float,
        ).reshape(-1)

    return wrapped


def make_background_mass_series_basis_1d(
    base_pdf,
    max_degree=4,
):
    mass_min = 5.170
    mass_max = 5.500

    grid = np.linspace(
        mass_min,
        mass_max,
        200000,
    )

    base_values = np.asarray(
        base_pdf(grid),
        dtype=float,
    ).reshape(-1)

    u_grid = (
        (grid - mass_min)
        / (mass_max - mass_min)
    )

    basis_functions = []
    basis_labels = []

    for degree in range(
        max_degree + 1
    ):
        factor_grid = (
            u_grid ** degree
        )

        integral = np.trapezoid(
            base_values * factor_grid,
            grid,
        )

        if (
            not np.isfinite(integral)
            or integral <= 0.0
        ):
            raise RuntimeError(
                "Bad normalization for "
                f"background mass degree "
                f"{degree}: {integral}"
            )

        def make_basis(
            deg,
            norm_value,
            basis_label,
        ):
            def basis(x):
                arr = np.asarray(
                    x,
                    dtype=float,
                )

                if arr.ndim == 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]

                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]

                arr = np.asarray(
                    arr,
                    dtype=float,
                ).reshape(-1)

                u_x = (
                    (arr - mass_min)
                    / (mass_max - mass_min)
                )

                base_values_x = np.asarray(
                    base_pdf(arr),
                    dtype=float,
                ).reshape(-1)

                values = (
                    base_values_x
                    * u_x ** deg
                    / norm_value
                )

                return np.asarray(
                    values,
                    dtype=float,
                ).reshape(-1)

            basis.__name__ = basis_label

            return basis

        label = (
            f"bkg_nominal_times_"
            f"mass_degree{degree}"
        )

        basis_functions.append(
            make_basis(
                degree,
                integral,
                label,
            )
        )

        basis_labels.append(
            label
        )

    return (
        basis_functions,
        basis_labels,
    )


input_files = sorted(
    glob.glob(input_pattern),
    key=file_number,
)

print(
    "Found",
    len(input_files),
    "coverage toys",
)

if len(input_files) == 0:
    raise RuntimeError(
        "No coverage toy files found."
    )


for j, infile in enumerate(input_files):

    toy_number = file_number(
        infile
    )

    print()
    print("=" * 80)
    print(
        f"Processing toy {toy_number} "
        f"({j + 1}/{len(input_files)})"
    )
    print("=" * 80)

    outfile = os.path.join(
        output_dir,
        f"{toy_number}.h5",
    )

    df = pd.read_hdf(
        infile
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Check toy composition
    # --------------------------------------------------

    n_signal_true = int(
        np.sum(
            df["is_signal"] == 1
        )
    )

    n_background_true = int(
        np.sum(
            df["is_signal"] == 0
        )
    )

    n_total_true = (
        n_signal_true
        + n_background_true
    )

    signal_fraction = (
        n_signal_true
        / n_total_true
        if n_total_true > 0
        else np.nan
    )

    print()
    print(
        "Toy composition:"
    )
    print(
        "  signal     =",
        n_signal_true,
    )
    print(
        "  background =",
        n_background_true,
    )
    print(
        "  total      =",
        n_total_true,
    )
    print(
        "  signal fraction =",
        signal_fraction,
    )

    if not np.isclose(
        signal_fraction,
        2.0 / 3.0,
        atol=0.01,
    ):
        print(
            "WARNING: this toy is not "
            "approximately 2/3 signal "
            "and 1/3 background."
        )

    # --------------------------------------------------
    # B-mass observable
    # --------------------------------------------------

    mass = zfit.Space(
        "B_mass",
        limits=(
            5.170,
            5.500,
        ),
    )

    # --------------------------------------------------
    # Signal mass model
    # --------------------------------------------------

    mu_sig = zfit.Parameter(
        f"mu_sig_{toy_number}",
        5.2803419191,
        5.26,
        5.30,
    )

    sigma_sig_1 = zfit.Parameter(
        f"sigma_sig_1_{toy_number}",
        0.0128153324,
        0.006,
        0.025,
    )

    sigma_sig_2 = zfit.Parameter(
        f"sigma_sig_2_{toy_number}",
        0.0185816699,
        0.006,
        0.050,
    )

    frac_cb1 = zfit.Parameter(
        f"frac_cb1_{toy_number}",
        0.4757138896,
        0.0,
        1.0,
    )

    alphal_1 = zfit.Parameter(
        f"alphal_1_{toy_number}",
        1.6218988271,
    )

    nl_1 = zfit.Parameter(
        f"nl_1_{toy_number}",
        1.4728249951,
    )

    alphar_1 = zfit.Parameter(
        f"alphar_1_{toy_number}",
        8.8805061092,
    )

    nr_1 = zfit.Parameter(
        f"nr_1_{toy_number}",
        3.4777885102,
    )

    alphal_2 = zfit.Parameter(
        f"alphal_2_{toy_number}",
        4.0559392732,
    )

    nl_2 = zfit.Parameter(
        f"nl_2_{toy_number}",
        0.2852616055,
    )

    alphar_2 = zfit.Parameter(
        f"alphar_2_{toy_number}",
        2.2846128649,
    )

    nr_2 = zfit.Parameter(
        f"nr_2_{toy_number}",
        2.5076116760,
    )

    for p in [
        alphal_1,
        nl_1,
        alphar_1,
        nr_1,
        alphal_2,
        nl_2,
        alphar_2,
        nr_2,
    ]:
        p.floating = False

    fitpdf_mass_cb1 = (
        zfit.pdf.DoubleCB(
            obs=mass,
            mu=mu_sig,
            sigma=sigma_sig_1,
            alphal=alphal_1,
            nl=nl_1,
            alphar=alphar_1,
            nr=nr_1,
        )
    )

    fitpdf_mass_cb2 = (
        zfit.pdf.DoubleCB(
            obs=mass,
            mu=mu_sig,
            sigma=sigma_sig_2,
            alphal=alphal_2,
            nl=nl_2,
            alphar=alphar_2,
            nr=nr_2,
        )
    )

    fitpdf_mass = (
        zfit.pdf.SumPDF(
            [
                fitpdf_mass_cb1,
                fitpdf_mass_cb2,
            ],
            fracs=frac_cb1,
        )
    )

    # --------------------------------------------------
    # Background nominal mass model
    # --------------------------------------------------

    lambda_bkg_1 = (
        zfit.Parameter(
            f"lambda_bkg_1_"
            f"{toy_number}",
            -6.0,
            -30.0,
            10.0,
        )
    )

    lambda_bkg_2 = (
        zfit.Parameter(
            f"lambda_bkg_2_"
            f"{toy_number}",
            -1.0,
            -20.0,
            10.0,
        )
    )

    frac_bkg_exp1 = (
        zfit.Parameter(
            f"frac_bkg_exp1_"
            f"{toy_number}",
            0.7,
            0.0,
            1.0,
        )
    )

    fitpdf_bkg_mass_1 = (
        zfit.pdf.Exponential(
            obs=mass,
            lambda_=lambda_bkg_1,
        )
    )

    fitpdf_bkg_mass_2 = (
        zfit.pdf.Exponential(
            obs=mass,
            lambda_=lambda_bkg_2,
        )
    )

    fitpdf_bkg_mass = (
        zfit.pdf.SumPDF(
            [
                fitpdf_bkg_mass_1,
                fitpdf_bkg_mass_2,
            ],
            fracs=frac_bkg_exp1,
        )
    )

    # --------------------------------------------------
    # Degree-4 background mass expansion in the 1D fit
    # --------------------------------------------------

    bkg_poly_coeffs = [1.0]

    for degree in range(1, 5):
        bkg_poly_coeffs.append(
            zfit.Parameter(
                f"bkg_mass_D4_coeff{degree}_{toy_number}",
                1.0,
                0.01,
                20.0,
            )
        )

    fitpdf_bkg_mass_poly = (
        zfit.pdf.Bernstein(
            obs=mass,
            coeffs=bkg_poly_coeffs,
        )
    )

    frac_bkg_nominal = (
        zfit.Parameter(
            f"frac_bkg_nominal_{toy_number}",
            0.8,
            0.0,
            1.0,
        )
    )

    fitpdf_bkg_mass_expanded = (
        zfit.pdf.SumPDF(
            [
                fitpdf_bkg_mass,
                fitpdf_bkg_mass_poly,
            ],
            fracs=frac_bkg_nominal,
        )
    )
    
    # --------------------------------------------------
    # Extended yields
    # --------------------------------------------------

    nsig_start = max(
        float(
            np.sum(
                df["is_signal"] == 1
            )
        ),
        1.0,
    )

    nbkg_start = max(
        float(
            np.sum(
                df["is_signal"] == 0
            )
        ),
        1.0,
    )

    Nsig = zfit.Parameter(
        f"Nsig_{toy_number}",
        nsig_start,
        0.0,
        200000.0,
    )

    Nbkg = zfit.Parameter(
        f"Nbkg_{toy_number}",
        nbkg_start,
        0.0,
        200000.0,
    )

    signal_mass_pdf = (
        fitpdf_mass.create_extended(
            Nsig,
        )
    )

    background_mass_pdf = (
        fitpdf_bkg_mass_expanded.create_extended(
            Nbkg,
        )
    )

    mass_model = (
        zfit.pdf.SumPDF(
            [
                signal_mass_pdf,
                background_mass_pdf,
            ]
        )
    )

    # --------------------------------------------------
    # Unweighted 1D B-mass fit
    # --------------------------------------------------

    mass_data = (
        zfit.Data.from_pandas(
            df[["B_mass"]],
            obs=mass,
        )
    )

    mass_loss = (
        zfit.loss.ExtendedUnbinnedNLL(
            model=mass_model,
            data=mass_data,
        )
    )

    result = (
        zfit.minimize.Minuit()
        .minimize(
            mass_loss,
        )
    )

    result.update_params()

    print()
    print(
        "1D B-mass fit result:"
    )
    print(
        result
    )

    if not result.valid:
        print(
            f"WARNING: toy "
            f"{toy_number} "
            "mass fit is not valid "
            "-- skipping"
        )

        del df
        del mass_data
        del mass_loss
        del result

        try:
            zfit.run.clear_graph_cache()
        except Exception:
            pass

        gc.collect()

        continue

    # --------------------------------------------------
    # Build fitted 1D mass PDFs for COWs
    # --------------------------------------------------

    mass_values = (
        df["B_mass"]
        .to_numpy(
            dtype=float,
        )
    )

    signal_mass_callable = (
        zfit_pdf_to_callable_1d_for_cows(
            fitpdf_mass,
            mass,
        )
    )

    background_mass_callable = (
        zfit_pdf_to_callable_1d_for_cows(
            fitpdf_bkg_mass_expanded,
            mass,
        )
    )

    # --------------------------------------------------
    # Degree-4 background mass expansion
    #
    # g_b,r(m) =
    #     g_b,0(m) * u(m)^r
    #
    # r = 0, 1, 2, 3, 4
    # --------------------------------------------------

    (
        background_mass_basis,
        background_mass_labels,
    ) = (
        make_background_mass_series_basis_1d(
            base_pdf=(
                background_mass_callable
            ),
            max_degree=4,
        )
    )

    print()
    print(
        "1D background mass basis:"
    )

    for label in (
        background_mass_labels
    ):
        print(
            " ",
            label,
        )

    # --------------------------------------------------
    # Mixture normalization I = g
    # --------------------------------------------------

    fitted_Nsig = float(
        Nsig.value()
    )

    fitted_Nbkg = float(
        Nbkg.value()
    )

    fitted_total = (
        fitted_Nsig
        + fitted_Nbkg
    )

    if (
        not np.isfinite(
            fitted_total
        )
        or fitted_total <= 0.0
    ):
        raise RuntimeError(
            "Invalid fitted total yield."
        )

    def mass_mixture_norm(x):
        return (
            fitted_Nsig
            / fitted_total
            * signal_mass_callable(x)
            +
            fitted_Nbkg
            / fitted_total
            * background_mass_callable(x)
        )

    # --------------------------------------------------
    # 1D mass-expansion COW
    # --------------------------------------------------

    mass_cow = Cows(
        sample=mass_values,
        sample_pdf=(
            mass_mixture_norm
        ),
        spdf=[
            signal_mass_callable,
        ],
        bpdf=(
            background_mass_basis
        ),
        norm=(
            mass_mixture_norm
        ),
        range=(
            5.170,
            5.500,
        ),
        summation=True,
    )

    mass_condition_number = (
        np.linalg.cond(
            mass_cow._am
        )
    )

    print()
    print(
        "1D mass-expansion COW "
        "condition number =",
        mass_condition_number,
    )

    # --------------------------------------------------
    # Signal COW weight
    # --------------------------------------------------

    signal_mass_weight = (
        mass_cow[0](
            mass_values
        )
    )

    df["signal_sweight"] = (
        np.asarray(
            signal_mass_weight,
            dtype=float,
        )
    )

    print()
    print(
        "Nsig fit =",
        float(
            Nsig.value()
        ),
    )

    print(
        "Nbkg fit =",
        float(
            Nbkg.value()
        ),
    )

    print(
        "sum signal_sweight =",
        df[
            "signal_sweight"
        ].sum(),
    )

    print(
        "signal_sweight min =",
        df[
            "signal_sweight"
        ].min(),
    )

    print(
        "signal_sweight max =",
        df[
            "signal_sweight"
        ].max(),
    )

    print(
        "signal_sweight "
        "negative fraction =",
        np.mean(
            df[
                "signal_sweight"
            ].to_numpy()
            < 0.0
        ),
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_hdf(
        outfile,
        key="data",
        mode="w",
    )

    print()
    print(
        "Saved:"
    )

    print(
        outfile
    )

    # --------------------------------------------------
    # Clean up
    # --------------------------------------------------

    del df
    del mass_data
    del mass_loss
    del result
    del mass_cow

    try:
        zfit.run.clear_graph_cache()
    except Exception:
        pass

    gc.collect()


print()
print(
    "Finished."
)