from topology import TopologyGenerator
from ChromatinDynamics import ChromatinDynamics
from logger import LogManager
import numpy as np

class Run:
    def __init__(self):
        self.logger = LogManager(log_file="logs/run_report.log").get_logger(__name__)
        self.logger.info("[Run] Initialized.")

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

    def SAW_bad_solvent(self, chi=0.0, N=1000):

        generator = TopologyGenerator()
        generator.gen_top([N])

        results = []

        self.logger.info("=" * 160)
        self.logger.info(f"Running simulation in Bad Solvent Condition (Chi = {chi})")
        self.logger.info("-" * 80)
        output_dir=f"bad_solvent_chi_{chi}_{N}"
        sim = ChromatinDynamics(generator.topology, integrator='langevin', platform_name="OpenCL", output_dir=output_dir, log_file=output_dir+'/CD.log')
        sim.system_setup(mode='SAW_bad_solvent', chi=chi)
        # sim.system_setup(mode='bad_solvent_collapse', chi=chi, r_rep=1.2)
        sim.simulation_setup()
        sps = sim.run(10000)
        for _ in range(5):
            sps = sim.run(10000)
            sim.print_force_info()
            Rg = sim.analyzer.compute_RG()
            results.append(Rg)
            
        return (np.mean(results), np.std(results))
        # self._log_final_results(results, "Radius of Gyration (Rg) vs Chi (Bad Solvent)", N)
    
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

        sps = sim.run(10000)
        sim.print_force_info()

        Rg = sim.analyzer.compute_RG()
        
        self.logger.info(f"SAW Radius of Gyration {Rg}")
        self.logger.info("-"*60)
    
    def _log_final_results(self, results, title, Np):
        self.logger.info("=" * 60)
        self.logger.info(f" {title} | N = {Np}")
        self.logger.info("=" * 60)
        self.logger.info(f"{'Parameter':<20} {'Radius of Gyration':<30} {'Ep':<20} {'Ek':<20} {'Steps/sec':<20}")
        self.logger.info("-" * 60)
        for param, rg, eP, eK, sps in results:
            self.logger.info(f"{param:<20} {rg:<30.3f} {eP:<20.3f} {eK:<20.3f} {sps:<20.0f}")
        self.logger.info("=" * 60)
