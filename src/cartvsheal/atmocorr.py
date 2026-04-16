"""Atmospheric correction helpers (pure PyTorch, no nn.Module).

Implements the unif → surface reflectance step from the 5S radiative
transfer model, as used in adjeff.modules.models.unif2surface.
"""

import torch

from .cartesian_fft_convolve import fft_convolve_2D_torch


def unif_to_surface(
    rho_unif: torch.Tensor,
    sph_alb: torch.Tensor,
    tdir_up: torch.Tensor,
    tdif_up: torch.Tensor,
    psf_kernel: torch.Tensor,
    *,
    padding: str = "reflect",
) -> torch.Tensor:
    """Estimate surface reflectance from uniform-scene reflectance (5S model).

    Convolves *rho_unif* with *psf_kernel* to obtain the environmental
    reflectance *rho_env*, then applies the 5S adjacency-effect correction:

    .. code-block:: text

        frac    = (1 - rho_env · sph_alb) / (1 - rho_unif · sph_alb)
        rho_s   = (rho_unif · (tdir_up + tdif_up) · frac
                   - rho_env · tdif_up) / tdir_up

    Parameters
    ----------
    rho_unif : torch.Tensor
        Uniform-scene reflectance, shape (H, W).
    sph_alb : torch.Tensor
        Spherical albedo of the atmosphere, broadcastable to (H, W).
    tdir_up : torch.Tensor
        Direct upward transmittance, broadcastable to (H, W).
    tdif_up : torch.Tensor
        Diffuse upward transmittance, broadcastable to (H, W).
    psf_kernel : torch.Tensor
        Normalised 2-D PSF kernel, shape (K, K).
    padding : {"reflect", "replicate", "constant"}, optional
        Padding mode passed to :func:`fft_convolve_2D_torch` (default
        ``"reflect"``).

    Returns
    -------
    torch.Tensor
        Estimated surface reflectance *rho_s*, shape (H, W).
    """
    rho_env = fft_convolve_2D_torch(
        rho_unif,
        psf_kernel,
        padding=padding,
        conv_type="same",
    )

    frac = (1.0 - rho_env * sph_alb) / (1.0 - rho_unif * sph_alb)
    rho_s = (rho_unif * (tdir_up + tdif_up) * frac - rho_env * tdif_up) / tdir_up

    return rho_s
