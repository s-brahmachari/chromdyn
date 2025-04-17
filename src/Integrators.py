from openmm import (LangevinIntegrator,BrownianIntegrator, CustomIntegrator)
from Utilities import LogManager
# -------------------------------------------------------------------
# Integrator Manager: Creates integrators for Brownian or Langevin dynamics
# -------------------------------------------------------------------
class IntegratorManager:
    VALID_INTEGRATORS = ["langevin", "brownian", "active-brownian"]  # List of supported integrators
    DEFAULT_PARAMS = {
        "temperature": 300.0,
        "friction": 0.1,
        "timestep": 0.01,
        "tcorr": 1.0,
        "Fact": 0.0  # Placeholder if needed in future
    }

    def __init__(self, logger=None, **kwargs):
        """
        Initialize the IntegratorManager with flexible parameters.

        Args:
            integrator (str): Type of integrator ('brownian', 'langevin', 'active').
            logger (Logger, optional): Logger instance. If None, initializes a default logger.
            **kwargs: Optional parameters (temperature, friction, timestep, tcorr, Fact).
        """
        self.logger = logger or LogManager().get_logger(__name__)
        self.integrator_name = kwargs.get('integrator', 'langevin')
        # self.logger.info('-'*60)
        # Load parameters with defaults
        self.temperature = kwargs.get("temperature", self.DEFAULT_PARAMS["temperature"])
        self.friction = kwargs.get("friction", self.DEFAULT_PARAMS["friction"])
        self.timestep = kwargs.get("timestep", self.DEFAULT_PARAMS["timestep"])
        self.tcorr = kwargs.get("tcorr", self.DEFAULT_PARAMS["tcorr"])
        self.Fact = kwargs.get("Fact", self.DEFAULT_PARAMS["Fact"])

        # Log initialization summary
        self.logger.info(f"Valid integrators: {self.VALID_INTEGRATORS}| Selected: {self.integrator_name}")
        self.integrator = self.create_integrator(self.integrator_name)
        # self.logger.info('-'*60)

    def create_integrator(self, integrator):
        """
        Create and return an OpenMM integrator based on initialized settings.

        Returns:
            OpenMM Integrator object.
        """
        try:
            if integrator == "brownian":
                # self.logger.info(f"Creating {integrator} ...")
                self.logger.info(f"BrownianIntegrator: temperatute={self.temperature} | friction={self.friction} | timestep={self.timestep}")
                return BrownianIntegrator(self.temperature, self.friction, self.timestep)

            elif integrator == "langevin":
                # self.logger.info(f"Creating {integrator} ...")
                self.logger.info(f"LangevinIntegrator: temperatute={self.temperature} | friction={self.friction} | timestep={self.timestep}")
                return LangevinIntegrator(self.temperature, self.friction, self.timestep)

            elif integrator == "active":
                # self.logger.info(f"Creating {integrator} ...")
                self.logger.info(f"ActiveBrownianIntegrator: temperatute={self.temperature} | friction={self.friction} | timestep={self.timestep} | corr_time={self.tcorr} | active_force={self.Fact}")
                return ActiveBrownianIntegrator(
                    temperature=self.temperature,
                    collision_rate=self.friction,
                    timestep=self.timestep,
                    corr_time=self.tcorr
                )

        except Exception as e:
            self.logger.exception(f"[ERROR] Failed to create integrator: {e}")
            raise
        
# -------------------------------------------------------------------
# ActiveBrownianIntegrator: Custom Integrator Class
# -------------------------------------------------------------------

class ActiveBrownianIntegrator(CustomIntegrator):
    R"""
    The :class:`~.ActiveBrownianIntegrator` class is a custom Brownian integrator. Just like all Brownian integrators there are no velocities, there are only forces and displacements.
    Here we use the velocities as a proxy to keep track of the active force for each monomer. 
    Hence velocity v of a monomer represents the active force f = gamma * v.
    """
    def __init__(self,
                 temperature=120.0,
                 collision_rate=1.0,
                 timestep=0.001,
                 corr_time=10.0,
                 constraint_tolerance=1e-8,
                 ):

        # Create a new CustomIntegrator
        super(ActiveBrownianIntegrator, self).__init__(timestep)
        
        # add global variables
        kbT = 0.008314 * temperature
        self.addGlobalVariable("kbT", kbT)
        self.addGlobalVariable("g", collision_rate)
        self.addGlobalVariable("Ta", corr_time)
        self.setConstraintTolerance(constraint_tolerance)

        # add attributes
        self.collisionRate = collision_rate
        self.corrTime = corr_time
        self.temperature = temperature 

        self.addPerDofVariable("x1", 0) # for constraints
        self.addUpdateContextState()

        # update velocities. note velocities represent the active force and only depend on their value in previous time step
        # IMPORTANT: the force-group "0" is associated with keeping track of the active noise
        self.addComputePerDof("v", "(exp(- dt / Ta ) * v) + ((sqrt(1 - exp( - 2 * dt / Ta)) * f0 / g) * gaussian)")
        self.addConstrainVelocities()
        
        # compute position update
        # note that the update computed below *OVER* estimates the contribution from the force-group "0" or the active force
        # the intended contribution of the active force is in the (v * dt) term, but the (f * dt / g) term also has 
        # the contribution of force-group 0, which needs to be taken out -- as is done in the following line
        self.addComputePerDof("x", "x + (v * dt) + (dt * f / g) + (sqrt(2 * (kbT / g) * dt) * gaussian)")
        self.addComputePerDof("x", "x - (dt  * f0 / g)")

        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained