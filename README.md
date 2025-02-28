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
This project implements a **Variational Monte Carlo (VMC) approach** to estimate the ground-state energy of the **hydrogen atom**. The **Metropolis-Hastings algorithm** is used to sample the wavefunction, while **on-the-fly parameter optimization** is performed to refine the variational parameter \( \alpha \).

## Features
✅ Implements **Metropolis-Hastings sampling**  
✅ **Optimizes** the wavefunction parameter \( \alpha \) using **gradient descent**  
✅ **Saves results** to CSV files and **plots** them
✅ Provides **CLI** for customizable simulations  
✅ Includes **unit tests** with **high test coverage**  

## Installation
Ensure you have **Python 3.8+** installed. Then, clone the repository and install dependencies:
```bash
git clone https://github.com/eleonora-casoni/VMC-EigenHydrogen.git
cd VMC-EigenHydrogen
pip install -r requirements.txt 
