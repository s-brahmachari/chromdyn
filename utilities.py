from topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
from logger import LoggerManager  # Assuming you have a centralized LoggerManager

class Run:
    def __init__(self):
        self.logger = LoggerManager().get_logger(__name__)
        self.logger.info("[Tester] Initialized.")

    def harmtrap(self):
        kr_values = [0, 0.1, 1, 5]
        Np = 1000

        generator = TopologyGenerator()
        generator.gen_top([Np])

        results = []

        for kr in kr_values:
            self.logger.info(f"Running simulation with Harmonic Trap stiffness kr = {kr}")
            self.logger.info("-" * 80)

            sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"harmtrap_kr_{kr}")
            sim.system_setup(mode='harmtrap', k_res=kr)
            sim.simulation_setup()

            sim.run(10000)
            sim.print_force_info()

            Rg = sim.analyzer.compute_RG()
            results.append((kr, Rg))

        self._log_final_results(results, "Radius of Gyration (Rg) vs Trap Stiffness (kr)", Np)

    def harmtrap_with_self_avoidance(self):
        r_rep_values = [0.1, 1, 1.5]
        Np = 1000

        generator = TopologyGenerator()
        generator.gen_top([Np])

        results = []

        for r_rep in r_rep_values:
            self.logger.info(f"Running simulation with Harmonic Trap + Self-Avoidance (r_rep = {r_rep})")
            self.logger.info("-" * 80)

            sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"harmtrap_rep_{r_rep}")
            sim.system_setup(mode='harmtrap_with_self_avoidance', r_rep=r_rep)
            sim.simulation_setup()

            sim.run(10000)
            sim.print_force_info()

            Rg = sim.analyzer.compute_RG()
            results.append((r_rep, Rg))

        self._log_final_results(results, "Radius of Gyration (Rg) vs Repulsion Distance (r_rep)", Np)

    def bad_solvent_with_self_avoidance(self):
        chi_values = [0.0, -0.2,-0.5,-1.0]
        Np = 1000

        generator = TopologyGenerator()
        generator.gen_top([Np])

        results = []

        for chi in chi_values:
            self.logger.info(f"Running simulation in Bad Solvent Condition (Chi = {chi})")
            self.logger.info("-" * 80)

            sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"bad_solvent_chi_{chi}")
            sim.system_setup(mode='bad_solvent_collapse', chi=chi)
            sim.simulation_setup()

            sim.run(10000)
            sim.print_force_info()

            Rg = sim.analyzer.compute_RG()
            results.append((chi, Rg))

        self._log_final_results(results, "Radius of Gyration (Rg) vs Chi (Bad Solvent)", Np)
    
    def rouse(self,):
        
        Np = 1000

        generator = TopologyGenerator()
        generator.gen_top([Np])

        self.logger.info(f"Running simulation: SAW)")
        self.logger.info("-" * 80)

        sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"rouse")
        sim.system_setup(mode='rouse')
        sim.simulation_setup()

        sim.run(10000)
        sim.print_force_info()

        Rg = sim.analyzer.compute_RG()
        
        self.logger.info(f"Rouse Radius of Gyration {Rg}")
        self.logger.info("-"*60)

    def saw(self,):
        
        Np = 1000

        generator = TopologyGenerator()
        generator.gen_top([Np])

        self.logger.info(f"Running simulation: SAW)")
        self.logger.info("-" * 80)

        sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=f"rouse")
        sim.system_setup(mode='saw')
        sim.simulation_setup()

        sim.run(10000)
        sim.print_force_info()

        Rg = sim.analyzer.compute_RG()
        
        self.logger.info(f"SAW Radius of Gyration {Rg}")
        self.logger.info("-"*60)

    def _log_final_results(self, results, title, Np):
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f" {title} | N = {Np}")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Parameter':<20} {'Radius of Gyration (nm)':<30}")
        self.logger.info("-" * 60)
        for param, rg in results:
            self.logger.info(f"{param:<20} {rg:<30.4f}")
        self.logger.info("=" * 60)
