from openmm import (LangevinIntegrator,BrownianIntegrator, CustomIntegrator)

# -------------------------------------------------------------------
# Integrator Manager: Creates integrators for Brownian or Langevin dynamics
# -------------------------------------------------------------------
class IntegratorManager:
    def __init__(self, integrator="langevin", temperature=300.0, friction=0.1, timestep=0.01, tcorr=1.0):
        self.integrator = integrator.lower()
        self.temperature = temperature
        self.friction = friction
        self.timestep = timestep
        self.tcorr = tcorr

    def create_integrator(self):
        if self.integrator == "brownian":
            return BrownianIntegrator(self.temperature, self.friction, self.timestep)
        elif self.integrator == "langevin":
            return LangevinIntegrator(self.temperature, self.friction, self.timestep)
        elif self.integrator == "active":
            return ActiveBrownianIntegrator(self.temperature, self.friction, self.timestep, self.tcorr)
        else:
            raise ValueError("Unsupported dynamics type. Use 'brownian' or 'langevin'.")


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