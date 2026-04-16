# Comparison of the convolution -- cartesian grid vs HEALPix grid

## Requirements

- [pixi](https://pixi.sh) — conda/PyPI environment manager

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Setup

```bash
git clone https://github.com/walcark/comp_conv_cartesian_healpix.git
cd comp_conv_cartesian_healpix
```

Pixi installs dependencies automatically on first run, but you can also install explicitly:

```bash
pixi install -e cpu   # CPU environment
pixi install -e gpu   # GPU environment (requires CUDA 12.6)
```

## Running a script

Prefix every command with `pixi run -e cpu` or `pixi run -e gpu` depending on your machine:

```bash
# FFT cartesian convolution test
pixi run -e cpu python scripts/test_fftconv_cartesian.py

# Atmospheric correction test
pixi run -e cpu python scripts/test_atmocorr.py
```

On a GPU machine:

```bash
pixi run -e gpu python scripts/test_fftconv_cartesian.py
```
