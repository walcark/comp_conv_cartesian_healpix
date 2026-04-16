"""Test script for the atmospheric correction (unif -> surface)."""

import matplotlib.pyplot as plt
import torch

from cartvsheal import disk_image, king_psf, radial_profile, unif_to_surface


def main():
    RES = 0.120
    N = 1000

    # Observed uniform-scene reflectance: bright disk on dark background
    rho_unif = disk_image(res=RES, n=N, radius=20) * 0.25 + 0.05

    psf_king = king_psf(res=RES, n=N, sigma=0.5, gamma=1.5)

    # Realistic 5S atmospheric parameters (scalar, spectrally representative)
    sph_alb = torch.tensor(0.12)  # spherical albedo
    tdir_up = torch.tensor(0.72)  # direct upward transmittance
    tdif_up = torch.tensor(0.13)  # diffuse upward transmittance

    rho_s = unif_to_surface(
        rho_unif,
        sph_alb=sph_alb,
        tdir_up=tdir_up,
        tdif_up=tdif_up,
        psf_kernel=psf_king,
    )

    extent = [-N * RES / 2, N * RES / 2, -N * RES / 2, N * RES / 2]

    # --- 2D side-by-side ---
    fig2d, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    vmin = min(rho_unif.min().item(), rho_s.min().item())
    vmax = max(rho_unif.max().item(), rho_s.max().item())

    im1 = ax1.imshow(
        rho_unif.numpy(),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(r"$\rho_{unif}$ (input)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig2d.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(
        rho_s.numpy(),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(r"$\rho_s$ (corrected)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig2d.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    diff = (rho_s - rho_unif).numpy()
    absmax = max(abs(diff.min()), abs(diff.max()))
    im3 = ax3.imshow(
        diff,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-absmax,
        vmax=absmax,
    )
    ax3.set_title(r"$\rho_s - \rho_{unif}$ (correction)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig2d.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig2d.tight_layout()

    # --- Radial profiles ---
    r_unif, v_unif = radial_profile(rho_unif, RES)
    r_s, v_s = radial_profile(rho_s, RES)

    fig_rad, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_unif, v_unif, label=r"$\rho_{unif}$")
    ax.plot(r_s, v_s, label=r"$\rho_s$ (corrected)", linestyle="--")
    ax.set_xlabel("r")
    ax.set_ylabel("azimuthal mean")
    ax.set_title("Radial profiles — atmospheric correction")
    ax.legend()
    fig_rad.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
