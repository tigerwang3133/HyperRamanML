# Machine Learning-Enhanced Hyperspectral Raman Imaging for Label-Free Molecular Atlas of Alzheimer’s Brain

A comprehensive analysis pipeline for hyperspectral Raman microscopy data, focusing on brain tissue imaging and classification using machine learning techniques. [Paper link](https://www.google.com).

## Project Overview

This project processes hyperspectral Raman spectroscopy data from brain tissue samples to:
- Normalize and preprocess spectral data
- Remove noise and outliers using baseline correction and PCA
- Map spectral features to spatial coordinates
- Perform component analysis and cross-correlation with known molecular signatures
- Train SVM and Logistic Regression models for disease classification
- Visualize results with spatial heat maps and spectral plots

## Features

- **Baseline Correction**: Asymmetric Least Squares (ALS) smoothing algorithm
- **Data Preprocessing**: Outlier removal, normalization, peak extraction
- **Component Analysis**: Correlation with molecular component libraries
- **Machine Learning**: SVM and Logistic Regression for classification
- **Visualization**: Spatial mapping and spectral plotting
- **PCA Analysis**: Principal component decomposition

## Project Structure

```
HyperRamanML/
├── README.md
├── requirements.txt
├── brain mapping analysis.ipynb    # Main analysis pipeline
├── svm_feature_importance.ipynb    # Feature importance & model training
└── map/
    ├── wet s5/                     # Sample data directory
    │   ├── animal 5.csv           # Spectral data
    │   └── animal 5.png           # Microscopy image
    ├── components.xlsx            # Molecular component library
    ├── components_abeta_harvard.xlsx
    ├── feature importance.xlsx    # SVM/LR coefficients
    └── map_axis.xlsx              # Calibration axis
```

## Data Format

### CSV Spectral Data
The spectral data files should contain:
- Column 1: `X` - X coordinate
- Column 2: `Y` - Y coordinate  
- Columns 3+: Intensity values at each wavenumber

Example: `animal 5.csv`

### Component Libraries (Excel)
- Column 1: Component name
- Columns 2+: Spectral intensity values

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tigerwang3133/HyperRamanML.git
   cd HyperRamanML
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Analysis Pipeline

1. Open `brain mapping analysis.ipynb` in Jupyter Notebook
2. Update the `sample_num` variable and `samples` dictionary to point to your data:
   ```python
   sample_num = 5
   samples = {
       5: ['./map/wet s5/', 'animal 5_-87_96_752_-54_62_480', 'animal 5']
   }
   ```
3. Run cells sequentially to:
   - Load and preprocess spectral data
   - Remove outliers using PCA thresholding
   - Normalize spectra and extract peaks
   - Correlate with molecular components
   - Generate spatial heat maps
   - Apply ML classifiers (SVM, Logistic Regression)

### Feature Importance Analysis

Use `svm_feature_importance.ipynb` to:
- Train SVM and Logistic Regression models
- Calculate feature importance
- Generate cross-validation scores
- Export model coefficients

## Key Parameters

### Preprocessing
- **Baseline correction**: `lam=10000, p=0.001` (Asymmetric Least Squares)
- **Savitzky-Golay filter**: window_length=11, polyorder=3
- **Peak extraction**: Percentile threshold (typically 30-70%)

### Outlier Detection (PCA-based)
Thresholds vary by sample:
```python
pca1threshold = 100000  # Sample 5
pca2threshold = 40000
```

### Peak Indices
- Normalize peak: 450 (1660 cm⁻¹)
- Secondary peak: 110 (1093 cm⁻¹)
- Brain-specific peak: 450 (1660 cm⁻¹)

### ML Model Parameters
- **SVM**: Training on spectral features
- **Logistic Regression**: Probability-based classification
- **PCA**: 3 principal components for visualization

## Output

The analysis generates:
- Spatial heat maps showing feature/marker distribution
- Box plots comparing component distributions
- Overlapping scatter plots for multi-component visualization
- PCA component plots
- ML classification heat maps
- DataFrames with spatial coordinates and analysis results

## Dependencies

See `requirements.txt` for complete list. Main packages:
- numpy, scipy: Numerical computing
- pandas: Data manipulation
- matplotlib: Visualization
- scikit-learn: Machine learning
- openpyxl: Excel file handling
- Pillow: Image processing

## Important Notes

### Data Privacy
- Sample data is for demonstration purposes
- Replace with your own spectral data
- Ensure compliance with data privacy regulations for brain tissue samples

### Hardware Requirements
- For large datasets (>10,000 spectra): 8GB+ RAM recommended
- Processing time: varies by sample size and number of components

### Troubleshooting

**Missing file errors**: Ensure `map/` directory structure and file names match the code paths

**Memory issues**: Reduce PCA components or process larger samples in batches

**Module import errors**: Verify all packages in `requirements.txt` are installed

## Mathematical Background

### Baseline Correction (ALS)
Asymmetric Least Squares algorithm for removing spectral background noise while preserving peak information.

### Cosine Similarity
Used for component correlation: measures angle between spectral vectors in high-dimensional space.

### Feature Normalization
Spectra normalized by reference peak intensity to account for sample thickness variations.

## Citation

If using this code in research, please cite:
```
[citation]
```

## License

MIT

## Contact

For questions or issues, please contact: zw76@rice.edu

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

