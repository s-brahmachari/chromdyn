from openmm import Platform
# -------------------------------------------------------------------
# Platform Manager: Selects GPU/CPU platform
# -------------------------------------------------------------------
class PlatformManager:
    def __init__(self, platform_name="CUDA"):
        self.platform_name = platform_name

    def get_platform(self):
        return Platform.getPlatformByName(self.platform_name)

    def list_openmm_platforms(self,):
        num_platforms = Platform.getNumPlatforms()
        print(f"Number of available OpenMM platforms: {num_platforms}\n")
        print(f"{'Index':<8} {'Platform Name':<20} {'Speed (estimated)':<20}")

        for i in range(num_platforms):
            platform = Platform.getPlatform(i)
            name = platform.getName()
            speed = platform.getSpeed()
            print(f"{i:<8} {name:<20} {speed:<20}")

    
