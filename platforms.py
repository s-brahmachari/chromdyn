from openmm import Platform
# -------------------------------------------------------------------
# Platform Manager: Selects GPU/CPU platform
# -------------------------------------------------------------------
class PlatformManager:
    def __init__(self, platform_name="CUDA"):
        self.platform_name = platform_name

    def get_platform(self):
        return Platform.getPlatformByName(self.platform_name)

