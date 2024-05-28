import enum

import gin

from .transforms import drop, min_max_norm


@gin.constants_from_enum
class TRANSFORMS(enum.Enum):
	MIN_MAX_NORM = "min_max_norm"  # Use string values for enum members
	DROP = "drop"


# Create a dictionary mapping enum values to functions
TRANSFORMS_DICT = {
	TRANSFORMS.MIN_MAX_NORM: min_max_norm,
	TRANSFORMS.DROP: drop,
}
