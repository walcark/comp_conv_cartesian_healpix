"""Simple scripts to test the FFT-based convolution using PyTorch."""

import matplotlib.pyplot as plt

from cartvsheal import fft_convolve_2D_torch, disk_image, king_psf, radial_profile


def main():
    RES = 0.120
    N = 1000

    test_disk = disk_image(res=RES, n=N, radius=20)

    psf_king = king_psf(res=RES, n=N, sigma=0.2, gamma=1.0)

    conv_disk = fft_convolve_2D_torch(
        test_disk,
        psf_king,
        padding="reflect",
        conv_type="same",
    )

    extent = [-N * RES / 2, N * RES / 2, -N * RES / 2, N * RES / 2]

    # --- 2D side-by-side ---
    fig2d, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(test_disk.numpy(), origin="lower", extent=extent, cmap="gray")
    ax1.set_title("test_disk (original)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig2d.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(conv_disk.numpy(), origin="lower", extent=extent, cmap="gray")
    ax2.set_title(r"conv_disk  (King PSF, $\sigma$=0.1, $\gamma$=1.0)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig2d.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig2d.tight_layout()

    # --- Radial profiles ---
    r_orig, v_orig = radial_profile(test_disk, RES)
    r_conv, v_conv = radial_profile(conv_disk, RES)

    fig_rad, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_orig, v_orig, label="test_disk")
    ax.plot(
        r_conv,
        v_conv,
        label=r"conv_disk (King $\sigma$=0.1, $\gamma$=1.0)",
        linestyle="--",
    )
    ax.set_xlabel("r")
    ax.set_ylabel("azimuthal mean")
    ax.set_title("Radial profiles")
    ax.legend()
    fig_rad.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
