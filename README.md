# VMC-EigenHydrogen
Variational Monte Carlo method to estimate ground state energy for the Hydrogen atom in 1D

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Available Arguments](#available-arguments)
  - [Understanding The Parameters](#understanding-the-parameters)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Results & Visualization](#results--visualization)
- [License](#license)

## Overview
This project implements a **Variational Monte Carlo (VMC) approach** to estimate the ground-state energy of the **hydrogen atom**. The **Metropolis-Hastings algorithm** is used to sample the wavefunction, while **on-the-fly parameter optimization** is performed to refine the variational parameter \( alpha \). If you're new to VMC and want to understand the theory behind this method, check out:  
📖[Computational Physics (2nd Edition) - Jos Thijssen](https://www.cambridge.org/) (Cambridge University Press)

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

**Run with Default Parameters:**
To run the simulation with default parameters, simply execute:

```bash
python -m vmc_simulation.main 
```
**Run with Custom Parameters:**
To specify custom values for the simulation parameters run:

```bash
python -m vmc_simulation.main --equilibration_steps 2000 --numsteps 100 --numwalkers 3000 --alpha 1 --learning-rate 0.005 --step-size 0.3 --output-dir my_results

```
**Run with configuration file:**
The [`config_files`](config_files/) folder contains configuration files. if you want to use them, run

```bash
python -m vmc_simulation.main --config filename.ini

```
You can even provide you own `.ini` configuration file, just save it in the [`config_files`](config_files/) folder. 

### Available Arguments

| Argument              | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `--numwalkers`        | Number of walkers exploring the phase space.                    |
| `--numsteps`          | Number of Metropolis steps.                                     |
| `--equilibration-steps` | Number of thermalization steps before measurement.             |
| `--alpha`             | Initial value of the variational parameter α.                 |
| `--learning-rate`        | Controls the step size in optimization.                 |
| `--step-size`        | Magnitude of random displacements in equilibration.                 |
| `--output`            | (Optional) Directory to save results (default: `results`).     |

## Output Directory

📂 By default, the simulation saves results in the `results/` folder, where you can find CSV files and plots generated using default parameters.

## Understanding The Parameters

All five parameters (**equilibration_steps**, **numsteps**, **numwalkers**, **learning_rate**, **step_size**) affect simulation performance:

**Too small:**    
*   Insufficient `equilibration_steps`, `numsteps`, or `numwalkers` will result in poor statistical accuracy and unreliable results. 
*   Insufficient `learning_rate` will make convergence too slow.
*   Insufficient `step_size` will cause walkers to explore too little, reducing sampling efficiency and preventing proper optimization

**Too large:**   
*   Excessive `equilibration_steps`, `numsteps`, or `numwalkers` will lead to extremely slow computation times or even freezing, hindering the simulation's efficiency. 
*   Excessive `learning_rate` may cause the simulation to fail to converge.
*   Excessive `step_size` will make walkers jump too far, potentially missing important regions of the probability distribution, reducing accuracy.


⚠️  **Recommendation:** Carefully choose reasonable parameter values to strike a balance between simulation speed and the desired level of accuracy.

## Project structure

VMC-EigenHydrogen/  
├── 📂vmc_simulation/  
│   ├── __init__.py  
│   ├── main.py           # Runs the full VMC simulation  
│   ├── simulation.py     # Core simulation functions  
│   ├── plot.py           # Plotting & result-saving functions  
│   ├── config_handler.py  # Handles config files    
├── 📂tests/              # Unit tests for all components  
├── 📂results/            # Default folder for generated plots & CSVs  
├── 📂config_files/       # stores configuration files    
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

### ℹ️ Accessing Help  
For a complete list of parameters, their descriptions and their default values run:  
```bash
python -m vmc_simulation.main --help
```

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! Feel free to open an **issue** or submit a **pull request**, but remember to write **tests**!

**Author**
----------

**Eleonora Casoni**  
👩‍💻 [GitHub Profile](https://github.com/eleonora-casoni) 🐱

