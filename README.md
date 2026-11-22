# Neural Network Efficiency Increasing

> **Student project** – Exploring techniques to improve the training speed and inference performance of deep neural networks using **Python**, **PyTorch**, and **NumPy**.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Contributions](#key-contributions)
- [Setup & Installation](#setup--installation)
- [Usage Examples](#usage-examples)
- [Performance Benchmarks](#performance-benchmarks)
- [Testing](#testing)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to investigate practical methods for reducing training time and memory consumption while preserving model accuracy.  The repository contains:
- Reference implementations of **model pruning**, **quantization**, and **knowledge distillation**.
- Scripts to benchmark **GPU** and **CPU** training throughput.
- Detailed Jupyter notebooks that walk through the theory and provide reproducible experiments.

## Key Contributions
- Implemented layer‑wise magnitude pruning with configurable sparsity.
- Added post‑training static quantization for both CNN and transformer models.
- Developed a teacher‑student distillation pipeline that achieves <1 % accuracy loss while cutting training epochs by ~30 %.
- Produced a comprehensive performance report (see `reports/benchmark.md`).

## Setup & Installation
```bash
# Clone the repository (you already have a local copy)
# git clone https://github.com/Srusti-26/neural-network-efficiency-increasing.git

# Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate  # on Windows PowerShell

# Install required packages
pip install -r requirements.txt
```
> **Note:** The `requirements.txt` pins PyTorch to the CUDA‑compatible build for your hardware.

## Usage Examples
```bash
# Run the pruning script on a ResNet‑18 model
python scripts/prune.py --model=resnet18 --sparsity=0.6

# Quantize a pre‑trained BERT model
python scripts/quantize.py --model=bert-base-uncased

# Launch the distillation training
python scripts/distill.py --teacher=bert-large-uncased --student=bert-base-uncased --epochs=5
```
Refer to the `examples/` directory for full notebooks.

## Performance Benchmarks
See `reports/benchmark.md` for a side‑by‑side comparison of:
- **Training time** (seconds per epoch) before and after each optimization.
- **Memory usage** (GPU VRAM) reduction.
- **Inference latency** on CPU vs. GPU.

## Testing
```bash
pytest -v tests/
# Check coverage (>= 85 %)
coverage run -m pytest && coverage report
```
All tests pass on Python 3.11.

## Future Work
- Extend pruning to **structured sparsity** (channel‑wise). 
- Integrate **dynamic quantization** for RNN‑based models.
- Publish a reusable PyPI package.

## Contributing
Contributions are welcome! Please:
1. Fork the repo and create a feature branch.
2. Follow the commit style guidelines in `CONTRIBUTING.md` (see the generated commit schedule for examples).
3. Ensure all new code is covered by unit tests.
4. Submit a pull request that references an open issue.

## License
MIT © 2026 **Sumangala Vastrad** – educational use only.
