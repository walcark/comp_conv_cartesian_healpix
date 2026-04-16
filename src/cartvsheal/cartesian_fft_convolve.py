from typing import Literal, cast

import torch


def fft_convolve_2D_torch(
    in1: torch.Tensor,
    in2: torch.Tensor,
    *,
    padding: Literal["constant", "reflect", "replicate"],
    const_padding_values: float = 0.0,
    conv_type: str = "valid",
) -> torch.Tensor:
    """Compute a **true linear 2D FFT convolution** between two torch tensors.

    Unlike a standard FFT-based convolution, which is circular and can produce
    wrap around artifacts at the tensor edges, the linear convolution is
    computed by:

    1) Extending the input to avoid wrap-around,
    2) Zero-padding the kernel to match the extended input,
    3) Multiplying in the Fourier domain (rFFT -> iFFT),
    4) Cropping the result according to `conv_type`.

    This method is GPU-efficient and minimizes temporary allocations.

    Parameters
    ----------
    in1 : torch.Tensor
        2D input tensor of shape (N, N).
    in2 : torch.Tensor
        2D convolution kernel of shape (K, K).
    padding : {"constant", "reflect", "replicate"}
        How to pad the input before FFT. Only `"constant"` uses
        `const_padding_values`.
    const_padding_values : float, optional
        Constant value used when `padding="constant"` (default 0.0).
    conv_type : {"valid", "same"}, optional
        Determines output size: `"valid"`: only positions where the
        kernel fully overlaps input (N-K+1 × N-K+1), and `"same"`
        output has the same shape as input (N × N).

    Returns
    -------
    torch.Tensor
        The convolved 2D tensor, with shape determined by `mode_out`.

    """
    n = in1.shape[0]  # input size
    k = in2.shape[0]  # kernel size
    ext = n + k - 1  # Full linear extension

    # Simply pad the input with the constant value in constant
    # mode, else cast to 3D for `reflect` and `replicate`.
    pad = (0, k - 1, 0, k - 1)
    if padding == "constant":
        in1_ext = torch.nn.functional.pad(
            in1,
            pad,
            mode="constant",
            value=const_padding_values,
        )
    else:
        in1_ext = in1.unsqueeze(0).unsqueeze(0)
        in1_ext = torch.nn.functional.pad(
            in1_ext,
            pad,
            mode=padding,
        )
        in1_ext = in1_ext.squeeze(0).squeeze(0)

    # Ensure width even for rfft
    ext_fft = ext + (ext % 2)
    add_col = ext_fft != ext
    if add_col:
        in1_ext = torch.nn.functional.pad(
            in1_ext,
            (0, 1, 0, 0),
            mode="constant",
            value=0.0,
        )

    # Kernel padded — use F.pad so the gradient flows through in2.
    # In-place assignment into a fresh tensor severs the autograd graph.
    in2_ext = torch.nn.functional.pad(in2, (0, ext_fft - k, 0, ext - k))

    # FFT-based linear conv
    in1_fft = torch.fft.rfftn(in1_ext, s=(ext, ext_fft), dim=(0, 1))
    in2_fft = torch.fft.rfftn(in2_ext, s=(ext, ext_fft), dim=(0, 1))
    Y = torch.fft.irfftn(in1_fft * in2_fft, s=(ext, ext_fft), dim=(0, 1))
    if add_col:
        Y = Y[:, :ext]

    # Crop according to `conv_type`
    if conv_type == "valid":
        out_n = n - k + 1
        top = k - 1
    elif conv_type == "same":
        out_n = n
        top = (ext - n) // 2
    else:
        raise ValueError("Mode should either be 'same' or 'valid'.")

    out = Y[top : top + out_n, top : top + out_n].contiguous()
    return cast(torch.Tensor, out)
