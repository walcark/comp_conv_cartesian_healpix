"""Utilitary functions for the project."""

import numpy as np
import torch


def disk_image(res: float, n: int, radius: float) -> torch.Tensor:
    """Return a simple image with a disk (1.0 inside, 0.0 outside)."""
    x = torch.linspace(-n * res / 2, n * res / 2, n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return torch.where(torch.sqrt(xx**2 + yy**2) <= radius, 1.0, 0.0)


def radial_profile(image: torch.Tensor, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthal mean radial profile of a 2D image.

    Bins pixels by radial distance from the centre using natural bin width
    (one pixel wide), matching the approach in adjeff.utils.radial.

    Parameters
    ----------
    image : torch.Tensor
        2D tensor of shape (n, n).
    res : float
        Pixel resolution (spatial units per pixel).

    Returns
    -------
    r_centers : np.ndarray
        Bin centre radii.
    mean_vals : np.ndarray
        Azimuthal mean value per bin.
    """
    n = image.shape[0]
    x = torch.linspace(-n * res / 2, n * res / 2, n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2).ravel()
    vv = image.float().ravel()

    # natural bin count: bin width ~ 1 pixel (mirrors natural_npix in adjeff)
    npix = max(int((n - 1) / (2**0.5)) - 1, 2)
    bins = torch.linspace(0.0, rr.max(), npix + 1)
    inds = (torch.bucketize(rr, bins, right=False) - 1).clamp(0, npix - 1)

    counts = torch.bincount(inds, minlength=npix).float()
    sum_vals = torch.bincount(inds, weights=vv, minlength=npix)

    r_centers = (0.5 * (bins[:-1] + bins[1:])).clone()
    r_centers[0] = 0.0

    mean_vals = torch.full((npix,), float("nan"))
    mask = counts > 0
    mean_vals[mask] = sum_vals[mask] / counts[mask]

    return r_centers.numpy(), mean_vals.numpy()
