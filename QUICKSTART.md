# Quick Start Guide

Get up and running with the trading system in minutes.

## Prerequisites

- Python 3.11+
- Required data files in `data/`:
  - `TSLA_1min.csv`
  - `SPY_1min.csv`
  - `VIX_History.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Essential Commands

### Training

Start the interactive training process:

```bash
python train.py
```

The script will guide you through model selection, configuration, and training setup.

### Label Inspector

Visualize and analyze trading labels:

```bash
python label_inspector.py
```

Interactive tool for inspecting label distributions, patterns, and quality.

### Dashboards

Choose from three dashboard options:

```bash
# Terminal-based dashboard (lightweight)
python dashboard.py

# Web-based dashboard (rich visualizations)
streamlit run streamlit_dashboard.py

# Interactive dashboard (advanced analysis)
python interactive_dashboard.py
```

## Quick Workflow

1. Verify your data files are in the `data/` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Start training: `python train.py`
4. Monitor results: `streamlit run streamlit_dashboard.py`
5. Inspect labels: `python label_inspector.py`

## Documentation

For detailed information:

- **Architecture**: `v7/docs/TECHNICAL_SPECIFICATION.md`
- **Training Guide**: `v7/training/README.md`
- **Features**: `v7/docs/FEATURE_SUMMARY.md`

## Need Help?

- Check the training README for detailed configuration options
- Review the technical specification for system architecture
- Inspect the feature summary for understanding model inputs
