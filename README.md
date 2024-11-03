# 🔬 Zeolite Data Visualization Tool

A [Streamlit](https://streamlit.io)-based web application for visualizing and analyzing zeolite structure data from the **Zeo-1** dataset. This tool provides comprehensive analysis of zeolite properties including energy distributions, force analysis, and stress tensor visualization.

## 📚 Dataset Citation 

```bibtex
@article{zeo1_2022,
  title={Zeo-1, a computational data set of zeolite structures},
  author={Komissarov, L. and Verstraelen, T.},
  journal={Scientific Data},
  volume={9},
  pages={91},
  year={2022},
  doi={10.1038/s41597-022-01160-5}
}
```

## ✨ Features

### 🔋 Energy Analysis
* Total energy distribution visualization
* Energy per atom analysis  
* Statistical summaries and distributions
* Interactive scatter plots and histograms

### 🔄 Force Analysis
* Force component distributions (x, y, z)
* Force magnitude analysis
* Comprehensive statistical metrics
* Multi-plot visualizations including histograms and box plots

### 💪 Stress Analysis
* Stress tensor visualization
* Principal stress analysis
* Von Mises stress calculations  
* Hydrostatic pressure analysis
* Maximum shear stress distributions

### 🔍 Structure Visualization
* 3D interactive molecular structure viewing
* Atom-specific information on hover
* Bond visualization
* Color-coded atomic species

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zeolite-visualization.git
   cd zeolite-visualization
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Required Dependencies

* `streamlit`
* `pandas`
* `plotly`
* `numpy`
* `scipy`
* `ase` (Atomic Simulation Environment)

## 🚀 Usage
Access the web interface at https://stablezeolitevisual.streamlit.app/ 

## 📄 Data Format

The application expects `.npz` files containing the following data for each zeolite:

* `numbers`: Atomic numbers
* `xyz`: Atomic positions `[frames, number_of_atoms, 3]`
* `lattice`: Unit cell parameters `[frames, 3, 3]`
* `energy`: Total energy per frame `[frames]`
* `gradients`: Force components `[frames, number_of_atoms, 3]`
* `stress`: Stress tensor `[frames, 3, 3]`
* `charges`: Atomic charges `[number_of_atoms]`


