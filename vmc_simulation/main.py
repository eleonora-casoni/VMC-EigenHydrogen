import argparse
from vmc_simulation.simulation import metropolis
from vmc_simulation.plot import (
    plot_position,
    plot_alpha_evolution,
    plot_energy_evolution,
    save_results_to_csv,
)


def main():
    """Parse command-line arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Run a Metropolis simulation for hydrogen atom ground state.")

    # Input options
    parser.add_argument("--numwalkers", type=int, default=4000, help="Number of Monte Carlo walkers (default: 5000)")
    parser.add_argument("--numsteps", type=int, default=120, help="Number of Monte Carlo steps (default: 120)")
    parser.add_argument("--equilibration_steps", type=int, default=3000, help="Equilibration steps (default: 3000)")
    parser.add_argument("--alpha", type=float, default=0.8, help="Initial variational parameter alpha (default: 0.8)")

    # Output options
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save outputs (default: ./results)")

    args = parser.parse_args()

    
    position_vec_fin, alpha_fin, alpha_buffer, E_buffer, dE_da_buffer, initial_pos = metropolis(
        args.equilibration_steps, args.numsteps, args.numwalkers, args.alpha
    )

    #expected values
    print('\n expected alpha value = 1')
    print('\n expected energy value = -1/2')

    #final results
    print(f"Final optimized alpha: {alpha_fin:.3f}")
    print(f"Mean final position: {position_vec_fin.mean():.3f}")
    print(f"variance of mean posiition:{position_vec_fin.var():.3f}")
    print(f"Mean local energy: {E_buffer.mean():.3f}")
    print(f"variance of mean local energy:{ E_buffer.var():.3f}")

    
    save_results_to_csv(position_vec_fin, alpha_buffer, E_buffer, args.output_dir)
    plot_position(position_vec_fin, save_path=f"{args.output_dir}/histogram_positions.png")
    plot_alpha_evolution(alpha_buffer, save_path=f"{args.output_dir}/alpha_evolution.png")
    plot_energy_evolution(E_buffer, save_path=f"{args.output_dir}/energy_evolution.png")
  

if __name__ == "__main__":
    main()
