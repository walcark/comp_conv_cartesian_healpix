"""Analytical PSF kernels as pure PyTorch functions.

Each function returns a normalised 2-D kernel tensor on a square grid of
size (n, n) with pixel resolution *res*.  No nn.Module, no trainable
parameters — intended for testing and visualisation.
"""

import torch


def _grid_r2(res: float, n: int) -> torch.Tensor:
    """Return the squared radial distance grid for an (n x n) image."""
    x = torch.linspace(-n * res / 2, n * res / 2, n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return xx**2 + yy**2


def gauss_psf(res: float, n: int, sigma: float) -> torch.Tensor:
    """Return a normalised Gaussian PSF kernel.

    Profile: ``exp(-r² / (2 σ²))``.

    Parameters
    ----------
    res : float
        Pixel resolution (spatial units per pixel).
    n : int
        Grid size (n × n pixels).
    sigma : float
        Standard deviation in the same spatial units as *res*.

    Returns
    -------
    torch.Tensor
        Normalised 2-D kernel of shape (n, n).
    """
    r2 = _grid_r2(res, n)
    kernel = torch.exp(-r2 / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gauss_general_psf(res: float, n: int, sigma: float, n_exp: float) -> torch.Tensor:
    """Return a normalised Generalised Gaussian PSF kernel.

    Profile: ``exp(-(r / σ)^n_exp)``.  Setting ``n_exp=2`` recovers the
    standard Gaussian; ``n_exp=1`` gives a Laplacian profile.

    Parameters
    ----------
    res : float
        Pixel resolution (spatial units per pixel).
    n : int
        Grid size (n × n pixels).
    sigma : float
        Scale radius in the same spatial units as *res*.
    n_exp : float
        Shape exponent controlling the tail steepness.

    Returns
    -------
    torch.Tensor
        Normalised 2-D kernel of shape (n, n).
    """
    r2 = _grid_r2(res, n)
    r = torch.sqrt(r2)
    kernel = torch.exp(-((r / sigma) ** n_exp))
    return kernel / kernel.sum()


def king_psf(res: float, n: int, sigma: float, gamma: float) -> torch.Tensor:
    """Return a normalised King profile PSF kernel.

    Profile: ``(1 + r² / (2 σ² γ))^{-γ}``.

    Parameters
    ----------
    res : float
        Pixel resolution (spatial units per pixel).
    n : int
        Grid size (n × n pixels).
    sigma : float
        Core width in the same spatial units as *res*.
    gamma : float
        Power-law index controlling the tail steepness.

    Returns
    -------
    torch.Tensor
        Normalised 2-D kernel of shape (n, n).
    """
    r2 = _grid_r2(res, n)
    kernel = (1.0 + r2 / (2.0 * sigma**2 * gamma)) ** (-gamma)
    return kernel / kernel.sum()
