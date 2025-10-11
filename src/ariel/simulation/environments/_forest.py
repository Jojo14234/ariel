# Standard library
from dataclasses import dataclass

# Third-party libraries
# Local libraries
from ariel.parameters.ariel_types import (
    Dimension,
    Rotation,
)
from ariel.simulation.environments._compound_world import CompoundWorld
from ariel.simulation.environments.heightmap_functions import (
    rugged_heightmap,
    hills_heightmap,
)
from ariel.utils.noise_gen import NormMethod

@dataclass
class ForestTerrainWorld(CompoundWorld):
    """A forest-like terrain world with ruggedness (CompoundWorld)."""

    name: str = "forest-world"

    floor_size: Dimension = (20, 20, 5)  # meters (width, height, depth)
    floor_tilt: Rotation = (0, 0, 0)  # degrees (x, y, z)
    floor_rot_sequence: str = "XYZ"  # xyzXYZ, assume intrinsic
    checker_floor: bool = False

    # Overall heightmap parameters
    dims: tuple[int, int] = (200, 200)

    # Rugged heightmap parameters
    height_of_noise: float = 0.5
    scale_of_noise: int = 10
    normalize: NormMethod = "none"

    hills_frequency: float = 0.1
    hills_amplitude: float = 2.0
    hills_octaves: int = 3

    def __post_init__(self) -> None:
        # Rugged part of heightmap
        rugged_part = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )
        rugged_part *= self.height_of_noise

        # Hills part of heightmap
        hills_part = hills_heightmap(
            dims=self.dims,
            frequency=self.hills_frequency,
            amplitude=self.hills_amplitude,
            octaves=self.hills_octaves,
            normalize=self.normalize,
        )

        self.floor_heightmap = rugged_part + hills_part

        super().__init__(
            name=self.name,
            floor_size=self.floor_size,
            floor_tilt=self.floor_tilt,
            floor_rot_sequence=self.floor_rot_sequence,
            checker_floor=self.checker_floor,
            floor_heightmap=self.floor_heightmap,
        )