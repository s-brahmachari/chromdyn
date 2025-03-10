from openmm import LangevinIntegrator,BrownianIntegrator

# -------------------------------------------------------------------
# Integrator Manager: Creates integrators for Brownian or Langevin dynamics
# -------------------------------------------------------------------
class IntegratorManager:
    def __init__(self, integrator="langevin", temperature=300.0, friction=0.1, timestep=0.01):
        self.integrator = integrator.lower()
        self.temperature = temperature
        self.friction = friction
        self.timestep = timestep

    def create_integrator(self):
        if self.integrator == "brownian":
            return BrownianIntegrator(self.temperature, self.friction, self.timestep)
        elif self.integrator == "langevin":
            return LangevinIntegrator(self.temperature, self.friction, self.timestep)
        else:
            raise ValueError("Unsupported dynamics type. Use 'brownian' or 'langevin'.")

