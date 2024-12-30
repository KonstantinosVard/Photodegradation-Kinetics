# Photoinitiator Photodissociation Kinetics

This project aims to determine the kinetic constant $k$ of the photodissociation process of newly synthesized photoinitiators, alongside characterizing the absorption spectrum of the resulting products. 

- **Importance of the Kinetic Constant**: The kinetic constant is crucial for understanding the photoinitiative efficiency in polymerization, as the cleavage of the unstable bond represents the first chemical step in the photoinitiation process.

- **Comparison with Computational Predictions**: The study also compares experimental results with computational predictions obtained using M06-2X and B3LYP functionals with a 6-31G basis set.

For further details, refer to the master thesis (available in Greek): [https://olympias.lib.uoi.gr/jspui/handle/123456789/38609](https://olympias.lib.uoi.gr/jspui/handle/123456789/38609).

## Problem Statement

The major challenge was the overlap in the UV-vis absorbance spectrum between the photoinitiators and their dissociation products. This overlap made it difficult to isolate the true dissociation kinetic constant. 

Resolving this interference is critical for:
- **Accurate Modeling**: Properly modeling the photodissociation process.
- **Correlation with Computational Chemistry**: Ensuring alignment with theoretical predictions from computational studies.

## Assumptions

The model makes the following key assumptions:

1. **Two-Species System**: Only two species are involved in the photodissociation process—the photoinitiator and its primary product. This assumption is valid under the presence of isosbestic points, which are specific wavelengths where the total absorbance remains constant over time during the reaction.

2. **Consistent Spectra**: The spectra of the reactant and product are assumed to remain consistent in shape throughout the reaction. This ensures that changes in absorbance accurately reflect the interconversion between the two species. This assumption applies to the typical range of absorbances studied in UV-vis spectroscopy ($A ≈ 0.2 - 2$).

These assumptions simplify the analysis and enable the isolation and modeling of the reaction kinetics.


Those two assumptions simplify the analysis and allow for the isolation and modeling of the reaction kinetics.

## Repository Contents

### 1. `implementation.ipynb`

This Jupyter notebook contains the step-by-step implementation of the analysis. It covers:

- **Data Preprocessing and Visualization**: Includes preparation and visualization of experimental results.
- **Kinetic Parameter Optimization**: Uses regression techniques to determine the kinetic constants.
- **Comparison with Computational Results**: Aligns experimental findings with theoretical predictions.

### 2. `after_isosbestic.py`

This Python script defines the `afterisosbestic` class, which implements the modeling of photodissociation kinetics. Key features include:

- **Kinetic Modeling**: Models first-order kinetics for reagents and products.
- **Loss Minimization**: Utilizes Adam optimization to fit experimental data.
- **Metrics**: Computes $R^2$ and adjusted $R^2$ values.
- **Visualization**: Generates plots to visualize kinetic behaviors at specific wavelengths and overall spectral changes.


## Results and Future Work

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The kinetic constant of photoinitiator cleavage was successfully characterized,
as the model's results aligned with theoretical calculations
and additional experimental photodissociation studies, such as photodissociation analysis using NMR.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Future work involves conducting HPLC-UV experiments to separate the overlapping spectra and 
accurately determine the true kinetic constant of a given photoinitiator under specific conditions,
such as concentration and temperature. These experiments would also provide the real spectrum of the reaction products.
Although this approach was attempted during the master’s thesis, it was unsuccessful due to the similar polarity of the reagent and product,
which hindered their effective separation. Optimizing separation techniques or exploring alternative methods could overcome this challenge in future studies.
