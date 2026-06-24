# QSGS Method for Generating 2D Porous Media

A Python dataset generator for creating square, binary, two-dimensional porous-media microstructures. It combines a **Quartet Structure Generation Set (QSGS)-style seed-and-growth process** with **Gaussian smoothing**, porosity adjustment, and pore-connectivity analysis.

The generated images can be used as synthetic inputs for porous-media research, image analysis, machine learning, numerical simulation, and workflow testing.

> 中文简介：本项目使用 QSGS 风格的随机成核与生长方法生成二维多孔介质结构，并通过高斯模糊和动态阈值改善边界平滑性、控制目标孔隙率。程序还会检测孔隙空间是否左右贯通，并将结果写入 PNG 文件名。

## Features

- Generates binary 2D porous-media microstructures.
- Supports configurable square image sizes and target porosities.
- Provides three seed-probability modes for different structure scales.
- Uses an 8-neighbor growth process to form solid clusters.
- Applies Gaussian smoothing and threshold-based rebinarization.
- Adjusts the solid-pixel count toward the requested porosity.
- Checks whether the pore phase connects the left and right boundaries.
- Saves connectivity information in each output filename.
- Displays generation progress and estimated remaining time.
- Writes detailed information to a log file.

## Generation Workflow

1. **Random seeding** — the grid starts as pore space, and solid seeds are placed according to the selected probability.
2. **QSGS-style growth** — solid clusters grow through randomly selected cells on their 8-neighbor boundaries until the target solid fraction is reached.
3. **Smoothing and rebinarization** — Gaussian blur produces smoother boundaries, and a dynamic threshold converts the result back into a binary image while targeting the requested porosity.
4. **Connectivity classification** — connected pore regions are identified. A sample is marked `conn` when a pore component touches both the left and right boundaries; otherwise, it is marked `block`.

```text
Random seeds
     ↓
QSGS-style solid growth
     ↓
Gaussian smoothing
     ↓
Dynamic thresholding
     ↓
Porosity calculation
     ↓
Left-to-right pore-connectivity check
     ↓
Binary PNG + log output
```

## Binary Image Convention

| Phase | Array value | PNG value | Color |
|---|---:|---:|---|
| Pore space | `0` | `0` | Black |
| Solid phase | `1` | `255` | White |

Porosity is calculated as:

```text
porosity = number of pore pixels / total number of pixels
```

## Requirements

- Python 3
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [SciPy](https://scipy.org/)

Install the dependencies:

```bash
python -m pip install numpy opencv-python scipy
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick Start

```bash
git clone https://github.com/renjiawen555/QSGSmethod-to-generate-porous-media.git
cd QSGSmethod-to-generate-porous-media
python -m pip install numpy opencv-python scipy
python generate_qsgs_dataset_v2.py
```

Generated images are written to:

```text
data/raw_images/
```

Detailed logs are written to:

```text
qsgs_generation.log
```

> **Important:** The current default configuration generates approximately **99,981 images**. This can require substantial CPU time and disk space. For an initial test, reduce `images_per_combination` or narrow the porosity and mode settings in `main()`.

## Default Configuration

| Parameter | Default | Description |
|---|---:|---|
| Image size | `128 × 128` | Width and height of every output image |
| Porosity range | `0.40` to `0.80` | Target pore fraction |
| Porosity step | `0.05` | Difference between adjacent porosity levels |
| Gaussian sigma | `1.0` | Boundary-smoothing strength |
| Modes | `A`, `B`, `C` | Three initial seed probabilities |
| Target dataset size | Approximately `100,000` | Distributed across all porosity-mode combinations |
| Output format | PNG | 8-bit binary image |

The default porosity values are:

```text
0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
```

## Structure Modes

| Mode | Seed probability | Intended morphology |
|---|---:|---|
| `A` | `0.01` | More seeds and relatively finer clusters |
| `B` | `0.005` | Intermediate structure scale |
| `C` | `0.0005` | Fewer seeds and larger clusters |

These descriptions are relative. The exact structure remains stochastic and also depends on porosity, image size, and Gaussian smoothing.

## Customization

The main configuration is located in `main()` inside `generate_qsgs_dataset_v2.py`:

```python
image_size = 128
porosities = np.arange(0.40, 0.85, 0.05)

modes = {
    "A": 0.01,
    "B": 0.005,
    "C": 0.0005,
}
```

### Small test run

Before generating the full dataset, replace the calculated image count with a small fixed value:

```python
images_per_combination = 5
```

### Change image size

```python
image_size = 256
```

The default workflow produces square images. The underlying `generate_qsgs(width, height, ...)` function accepts separate width and height values if rectangular images are needed in a custom workflow.

### Change porosity levels

```python
porosities = np.array([0.40, 0.50, 0.60, 0.70])
```

### Change smoothing strength

Modify the `sigma` value passed to `smooth_adjust_by_threshold`:

```python
image_array = smooth_adjust_by_threshold(
    image_array,
    porosity,
    sigma=1.0,
)
```

- Smaller `sigma`: retains more local detail and sharper boundaries.
- Larger `sigma`: produces smoother, rounder structures but may remove small features.

## Output Filename Format

Files follow this pattern:

```text
<porosity>_<mode>_<index>_<connectivity>.png
```

Example:

```text
0.60_B_371_conn.png
```

| Field | Meaning |
|---|---|
| `0.60` | Requested target porosity |
| `B` | Structure mode |
| `371` | Sequence number used by the current script |
| `conn` | Pore phase connects the left and right boundaries |
| `block` | No left-to-right pore connection was detected |

Example output structure:

```text
QSGSmethod-to-generate-porous-media/
├── generate_qsgs_dataset_v2.py
├── qsgs_generation.log
└── data/
    └── raw_images/
        ├── 0.40_A_371_block.png
        ├── 0.40_A_372_conn.png
        └── ...
```

## Reproducibility

Generation is stochastic because NumPy random sampling is used for seeding, growth, and threshold-tie adjustment. To obtain repeatable results, set a random seed at the beginning of `main()`:

```python
np.random.seed(42)
```

Without a fixed seed, repeated runs produce different microstructures. Existing files with identical names may be overwritten, so change the output directory or filename range when preserving previous results.

## Notes and Limitations

- The connectivity test checks only **left-to-right pore percolation**.
- Solid growth uses an 8-neighbor neighborhood, while the connectivity test uses SciPy's default connected-component structure.
- Connectivity is recorded as metadata; non-percolating samples are still saved.
- Porosity is represented on a discrete pixel grid, so achievable values depend on image resolution.
- Gaussian smoothing can change small-scale topology even when global porosity remains close to the target.
- Runtime grows with image dimensions and dataset size.
- Configuration is currently stored directly in source code; command-line arguments are not yet implemented.

## Possible Improvements

Contributions are welcome. Useful future improvements include:

- Command-line arguments for size, porosity, mode, count, seed, and output path.
- A `requirements.txt` or packaged installation configuration.
- Multi-directional connectivity checks.
- CSV or JSON metadata export.
- Automated tests for porosity accuracy and connectivity classification.
- Parallel generation for large datasets.
- Preview figures and statistical summaries.

## Project Structure

```text
.
├── README.md
└── generate_qsgs_dataset_v2.py
```

## Contributing

Bug reports, documentation improvements, algorithm discussions, and pull requests are welcome. When reporting a generation issue, please include:

- Python version and operating system
- Parameter values
- Relevant log output
- A small reproducible example
- Example generated images, when applicable

## Disclaimer

This project generates synthetic porous-media images for research and educational use. Users should independently validate whether the generated structures, phase convention, porosity accuracy, and connectivity definition are appropriate for their scientific or engineering application.

