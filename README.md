# VMC-EigenHydrogen
Variational Monte Carlo method to estimate ground state energy for the Hydrogen atom in 1D

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Using the CLI](#using-the-cli)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results & Visualization](#results--visualization)
- [License](#license)

## Overview
This project implements a **Variational Monte Carlo (VMC) approach** to estimate the ground-state energy of the **hydrogen atom**. The **Metropolis-Hastings algorithm** is used to sample the wavefunction, while **on-the-fly parameter optimization** is performed to refine the variational parameter \( alpha \). If you're new to VMC and want to understand the theory behind this method, check out:  
📖**[Computational Physics (2nd Edition) - Jos Thijssen](https://www.cambridge.org/)** (Cambridge University Press)

## Features
- ✅ Implements **Metropolis-Hastings sampling**  
- ✅ **Optimizes** the variational parameter using **gradient descent**  
- ✅ **Saves results** to CSV files and plots
- ✅ Provides **CLI** for customizable simulations  
- ✅ Includes **unit tests** with **high test coverage**  

## Installation
Ensure you have **Python 3.8+** installed. Then, clone the repository and install dependencies:
```bash
git clone https://github.com/eleonora-casoni/VMC-EigenHydrogen.git
cd VMC-EigenHydrogen
pip install -r requirements.txt 
```
## Usage
You can run the simulation using the default parameters or specify custom parameters.

**Run with Default Parameters**
To run the simulation with default parameters, simply execute:

```bash
python -m vmc_simulation.main 
```
**Run with Custom Parameters**
To specify custom values for the simulation parameters run:

```bash
python -m vmc_simulation.main --equilibration_steps 2000 --numsteps 100 --numwalkers 3000 --alpha 1 --learning-rate 0.005 --output-dir my_results

```
### Available Arguments

| Argument              | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `--numwalkers`        | Number of walkers exploring the phase space.                    |
| `--numsteps`          | Number of Metropolis steps.                                     |
| `--equilibration_steps` | Number of thermalization steps before measurement.             |
| `--alpha`             | Initial value of the variational parameter α.                 |
| `--learning_rate`        | Controls the step size in optimization.                 |
| `--output`            | (Optional) Directory to save results (default: `results`).     |

## Output Directory

📂 By default, the simulation saves results in the `results/` folder, where you can find CSV files and plots generated using default parameters.

## Understanding The Parameters

All four parameters (**equilibration_steps**, **numsteps**, **numwalkers**, **learning_rate**) affect simulation performance:

*   **Too small:** Insufficient `equilibration_steps`, `numsteps`, or `numwalkers` will result in poor statistical accuracy and unreliable results. Insufficient `learning_rate` will make convergence too slow.

*   **Too large:** Excessive `equilibration_steps`, `numsteps`, or `numwalkers` will lead to extremely slow computation times or even freezing, hindering the simulation's efficiency. With excessive `learning_rate` the simulation may fail to converge.

⚠️  **Recommendation:** Carefully choose reasonable parameter values to strike a balance between simulation speed and the desired level of accuracy.

## Project structure

VMC-EigenHydrogen/  
├── 📂vmc_simulation/  
│   ├── __init__.py  
│   ├── main.py           # Runs the full VMC simulation  
│   ├── simulation.py     # Core simulation functions  
│   ├── plot.py           # Plotting & result-saving functions  
├── 📂tests/              # Unit tests for all components  
├── 📂results/            # Default folder for generated plots & CSVs  
├── requirements.txt      # Required dependencies  
├── README.md             # Documentation  
├── .gitignore            # Git ignore file  
├── LICENSE  


## Testing

Run the unit tests with:

```bash
pytest 
```
The test suite ensures:

*   Trial wavefunction behavior is correct
*   Metropolis-Hastings sampling is valid
*   Alpha optimization updates correctly
*   Edge cases (corner values, invalid type inputs) are handled properly

## Results & Visualization

Simulation outputs are automatically saved in `results/`:

*   `final_positions.csv` → Final walker positions
*   `alpha_evolution.csv` → Alpha parameter evolution
*   `energy_evolution.csv` → Energy values over time

Plots are also generated:

*   📊 Histogram of final positions
*   📈 Alpha evolution over iterations
*   📉 Energy evolution over iterations

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**, but remember to write **tests**!

**Author**
----------

**Eleonora Casoni**  
👩‍💻 [GitHub Profile](https://github.com/eleonora-casoni) 🐱

