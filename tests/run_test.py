from chromdyn import (
    ChromatinDynamics, 
    TopologyGenerator,
    LogManager,
    PlatformManager,
    Analyzer,
    TrajectoryLoader
)
import openmm.unit as unit
import numpy as np

class run_tests:
    def __init__(self):
        self.logger = LogManager().get_logger(__name__)
        self.logger.info("Run tests initialized.")

    def platforms(self):
        platform=PlatformManager()
        platform.list_platforms()
        return platform.available_platforms
        
    def harmonic_bonds(self, platform="CPU"):
        kb_values = [1, 5, 10, 50]
        Np = 100

        generator = TopologyGenerator()
        generator.gen_top([Np])
        self.logger.info("-" * 80)
        self.logger.info(f"Running {platform} simulations to test harmonic bonds")
        self.logger.info("-" * 50)
        self.logger.info(f"{'k_b':<8} {'sigma_b':^20}")
        self.logger.info(f"{'':<8} {'expected':^10} {'sim':^10}")
        self.logger.info("-" * 50)
        for kb in kb_values:
            out_str = f"{kb:<8.2f} {np.sqrt(1.0/kb):^10.3f} "
            sim = ChromatinDynamics(generator.topology, name=f'test_bonds_{kb}', platform_name=platform, console_stream=False)
            sim.force_field_manager.add_harmonic_bonds(r0=1.0, k=kb)
            
            #set up the simulation
            sim.simulation_setup(
                init_struct='randomwalk',
                integrator='langevin',
                temperature=120.0,
                timestep=0.01,
                save_pos=True,
                save_energy=True,
                energy_report_interval=3_000,
                pos_report_interval=1000,              
                )
            #collapse run
            sim.run(5_000, report=False)
            
            # check the context
            sim.print_force_info()
            
            # save structures while running using reporters
            sim.run(10_000, report=True)
            
            #Simulations done -- save reports
            sim.save_reports()
            
            xyz = TrajectoryLoader.load(sim.reporters.get('position').filename)
            bond_lengths = np.linalg.norm(xyz[:,1:,:]-xyz[:,:-1,:], axis=2).flatten()
            sigma_b = np.std(bond_lengths)
            out_str+=f"{sigma_b:^10.3f}"
            self.logger.info(out_str)
        self.logger.info("-" * 80)

    def confinement(self, platform="CPU"):
        Rc_values = [5, 8, 12]
        Np = 100

        generator = TopologyGenerator()
        generator.gen_top([1]*Np)
        self.logger.info("-" * 80)
        self.logger.info(f"Running {platform} simulations to test confinement")
        self.logger.info("-" * 50)
        self.logger.info(f"{'Rc':<8} {'Rad. Gyr.':^20}")
        self.logger.info(f"{'':<8} {'expected':^10} {'sim':^10}")
        self.logger.info("-" * 50)
        for rc in Rc_values:
            out_str = f"{rc:<8.2f} {rc:^10.2f} "
            sim = ChromatinDynamics(generator.topology, name=f'test_conf_{rc}', platform_name=platform, console_stream=False)
            sim.force_field_manager.add_flat_bottom_harmonic(k=1.0, r0=rc)
            
            #set up the simulation
            sim.simulation_setup(
                init_struct='randomwalk',
                integrator='langevin',
                temperature=120.0,
                timestep=0.01,
                save_pos=True,
                save_energy=True,
                energy_report_interval=3_000,
                pos_report_interval=1000,              
                )
            #collapse run
            sim.run(5_000, report=False)
            
            # check the context
            sim.print_force_info()
            
            # save structures while running using reporters
            sim.run(10_000, report=True)
            
            #Simulations done -- save reports
            sim.save_reports()
            
            xyz = TrajectoryLoader.load(sim.reporters.get('position').filename)
            rg = np.mean(Analyzer.compute_RG(xyz))
            out_str+=f"{rg:^10.2f}"
            self.logger.info(out_str)
        self.logger.info("-" * 80)

    def self_avoidance(self, platform="CPU"):
        Np_values = [50, 100, 500]
        self.logger.info("-" * 80)
        self.logger.info(f"Running {platform} simulations to test self avoidance")
        self.logger.info("-" * 50)
        self.logger.info(f"{'N':<8} {'Rad. Gyr.':^20}")
        self.logger.info(f"{'':<8} {'expected':^10} {'sim':^10}")
        self.logger.info("-" * 50)
        
        for Np in Np_values:
            generator = TopologyGenerator()
            generator.gen_top([Np])

            out_str = f"{Np:<8.0f}{0.6*Np**(0.6):^10.2f} "
            sim = ChromatinDynamics(generator.topology, name=f'test_saw_{Np}', platform_name=platform, console_stream=False)
            sim.force_field_manager.add_harmonic_bonds(r0=1.0, k=30.0)
            sim.force_field_manager.add_self_avoidance()
            
            #set up the simulation
            sim.simulation_setup(
                init_struct='randomwalk',
                integrator='langevin',
                temperature=120.0,
                timestep=0.01,
                save_pos=True,
                save_energy=True,
                energy_report_interval=3_000,
                pos_report_interval=1000,              
                )
            #collapse run
            sim.run(50_000, report=False)
            
            # check the context
            sim.print_force_info()
            
            # save structures while running using reporters
            sim.run(50_000, report=True)
            
            #Simulations done -- save reports
            sim.save_reports()
            
            xyz = TrajectoryLoader.load(sim.reporters.get('position').filename)
            rg = np.mean(Analyzer.compute_RG(xyz))
            out_str+=f"{rg:^10.2f}"
            self.logger.info(out_str)
        self.logger.info("-" * 80)
    
    def remove_force(self, platform="CPU"):
        
        self.logger.info("-" * 80)
        self.logger.info(f"Running {platform} simulations to test force removal during sims")
        self.logger.info("-" * 50)
        self.logger.info(f"{'Type':<8} {'Rad. Gyr.':^20}")
        self.logger.info(f"{'':<8} {'expected':^10} {'sim':^10}")
        self.logger.info("-" * 50)
        
        Np=100
        Rc=15.0
        for val in ['polymer', 'gas']:
            generator = TopologyGenerator()
            if val=='polymer':        
                generator.gen_top([Np])
                out_str = f"{val:<8} {0.5*Np**(0.5):^10.2f} "
            elif val=='gas':
                generator.gen_top([1]*Np)
                out_str = f"{val:<8} {Rc:^10.2f} "
            
            sim = ChromatinDynamics(generator.topology, name=f'test_remove_force_{val}', platform_name=platform, console_stream=False)
            if val=='polymer':
                sim.force_field_manager.add_harmonic_bonds(r0=1.0, k=30.0)
            
            sim.force_field_manager.add_flat_bottom_harmonic(k=0.2, r0=Rc)
            
            #set up the simulation
            sim.simulation_setup(
                init_struct='randomwalk',
                integrator='langevin',
                temperature=120.0,
                timestep=0.01,
                save_pos=True,
                save_energy=True,
                energy_report_interval=3_000,
                pos_report_interval=1000,              
                )
            #collapse run
            sim.run(50_000, report=False)
            
            # check the context
            sim.print_force_info()
            
            # save structures while running using reporters
            sim.run(50_000, report=True)
            
            #Simulations done -- save reports
            sim.save_reports()
            
            xyz = TrajectoryLoader.load(sim.reporters.get('position').filename)
            rg = np.mean(Analyzer.compute_RG(xyz))
            out_str+=f"{rg:^10.2f}"
            self.logger.info(out_str)
        self.logger.info("-" * 80)

    def bad_solvent(self, platform="CPU"):
        self.logger.info("-" * 80)
        self.logger.info(f"Running {platform} simulations to test type-to-type interactions")
        self.logger.info("-" * 50)
        self.logger.info(f"{'N':<8} {'Rad. Gyr.':^20}")
        self.logger.info(f"{'':<8} {'expected':^10} {'sim':^10}")
        self.logger.info("-" * 50)
        
        for Np in [50, 100, 500]:
            generator = TopologyGenerator()
            generator.gen_top([Np])
            out_str = f"{Np:<8} {0.8*Np**(0.33):^10.2f} "
            
            sim = ChromatinDynamics(generator.topology, name=f'test_collapse_{Np}', platform_name=platform, console_stream=False)
            sim.force_field_manager.add_harmonic_bonds(r0=1.0, k=30.0)
            sim.force_field_manager.add_self_avoidance()
            type_labels = ["A", "B"]
            interaction_matrix = [[-0.5, 0.0], [0.0, 0.0]]
            sim.force_field_manager.add_type_to_type_interaction(interaction_matrix, type_labels, rc=1.75)
            
            #set up the simulation
            sim.simulation_setup(
                init_struct='randomwalk',
                integrator='langevin',
                temperature=120.0,
                timestep=0.01,
                save_pos=True,
                save_energy=True,
                energy_report_interval=3_000,
                pos_report_interval=1000,              
                )
            #collapse run
            sim.run(50_000, report=False)
            
            # check the context
            sim.print_force_info()
            
            # save structures while running using reporters
            sim.run(50_000, report=True)
            
            #Simulations done -- save reports
            sim.save_reports()
            
            xyz = TrajectoryLoader.load(sim.reporters.get('position').filename)
            rg = np.mean(Analyzer.compute_RG(xyz))
            out_str+=f"{rg:^10.2f}"
            self.logger.info(out_str)
        self.logger.info("-" * 80)

def test_minimal_simulation_runs(tmp_path):

    generator = TopologyGenerator()
    generator.gen_top([10])
    sim = ChromatinDynamics(generator.topology, name=f'test', output_dir=tmp_path, platform_name="CPU", console_stream=False)
    sim.force_field_manager.add_harmonic_bonds(r0=1.0, k=30.0)
            
    #set up the simulation
    sim.simulation_setup(
        init_struct='randomwalk',
        integrator='langevin',
        temperature=120.0,
        timestep=0.01,
        save_pos=False,
        save_energy=False,
        )
    #collapse run
    sim.run(10, report=False)
    context = sim.simulation.context
    state = context.getState(getPositions=True)
    
    positions = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer
            )

    assert positions.shape == (10, 3)
    assert np.all(np.isfinite(positions))