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
This project implements a **Variational Monte Carlo (VMC) approach** to estimate the ground-state energy of the **hydrogen atom**. The **Metropolis-Hastings algorithm** is used to sample the wavefunction, while **on-the-fly parameter optimization** is performed to refine the variational parameter \( alpha \). If you're new to Variational Monte Carlo (VMC) and want to understand the theory behind this method, check out:  
📖 **Computational Physics (Second Edition)**  
by **Jos Thijssen** – Cambridge University Press.

## Features
✅ Implements **Metropolis-Hastings sampling**  
✅ **Optimizes** the wavefunction parameter \( alpha \) using **gradient descent**  
✅ **Saves results** to CSV files and plots
✅ Provides **CLI** for customizable simulations  
✅ Includes **unit tests** with **high test coverage**  

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
By default, results (plots and CSV files) will be saved in the results folder.
```
Run with Custom Parameters
To specify custom values for the simulation parameters:

```bash
python -m vmc_simulation.main --equilibration_steps 2000 --numsteps 100 --numwalkers 3000 --alpha 1
```