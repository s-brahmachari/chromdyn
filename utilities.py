from topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics

class Tester:
    def __init__(self):
        pass

    def harmtrap(self):
        # List of trap stiffness values to iterate over
        kr_values = [0, 0.1, 1, 5]
        Np=1000
        # Initialize topology generator and generate polymer topology
        generator = TopologyGenerator()
        generator.gen_top([Np])  # Generate a chain of 1000 units

        # Store results: list of (kr, Rg)
        results = []

        # Iterate over trap stiffness values
        for kr in kr_values:
            print(f"\n[INFO] Running simulation with Harmonic Trap stiffness kr = {kr}\n" + "-"*60)
            
            # Initialize simulation object
            sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="CPU", output_dir=f"harmtrap_kr_{kr}")
            
            # Setup system with harmonic trap
            sim.system_setup(mode='harmtrap', kr=kr)
            
            # Setup simulation (platform, integrator, positions)
            sim.simulation_setup()
            
            # Run a short simulation to let system respond to trap (e.g., 100 steps)
            sim.run(10000)
            sim.analyzer.print_force_info()
            # Compute Radius of Gyration (Rg)
            Rg = sim.analyzer.compute_RG()
            
            # Store result
            results.append((kr, Rg))

        # ----------------------
        # Final summary table
        # ----------------------
        print("\n" + "="*50)
        print(f" Radius of Gyration (Rg) vs Trap Stiffness (kr) | N = {Np}")
        print("="*50)
        print(f"{'Trap Stiffness (kr)':<20} {'Radius of Gyration (nm)':<25}")
        print("-"*50)
        for kr, rg in results:
            print(f"{kr:<20} {rg:<25.4f}")
        print("="*50)

    def harmtrap_with_self_avoidance(self):
        # List of trap stiffness values to iterate over
        r_rep_values = [0.1, 1, 1.5]
        Np=1000
        # Initialize topology generator and generate polymer topology
        generator = TopologyGenerator()
        generator.gen_top([Np])  # Generate a chain of 1000 units

        # Store results: list of (kr, Rg)
        results = []

        # Iterate over trap stiffness values
        for r_rep in r_rep_values:
            print(f"\n[INFO] Running simulation with Harmonic Trap stiffness kr = {r_rep}\n" + "-"*60)
            
            # Initialize simulation object
            sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="CPU", output_dir=f"harmtrap_rep_{r_rep}")
            
            # Setup system with harmonic trap
            sim.system_setup(mode='harmtrap_with_self_avoidance', r_rep=r_rep)
            
            # Setup simulation (platform, integrator, positions)
            sim.simulation_setup()
            
            # Run a short simulation to let system respond to trap (e.g., 100 steps)
            sim.run(10000)
            sim.analyzer.print_force_info()
            # Compute Radius of Gyration (Rg)
            Rg = sim.analyzer.compute_RG()
            
            # Store result
            results.append((r_rep, Rg))

        # ----------------------
        # Final summary table
        # ----------------------
        print("\n" + "="*50)
        print(f" Radius of Gyration (Rg) vs repulsion distance | N = {Np}")
        print("="*50)
        print(f"{'Trap Stiffness (kr)':<20} {'Radius of Gyration (nm)':<25}")
        print("-"*50)
        for kr, rg in results:
            print(f"{kr:<20} {rg:<25.4f}")
        print("="*50)
    
if __name__ == "__main__":
    test = Tester()
    # test.harmtrap()
    test.harmtrap_with_self_avoidance()