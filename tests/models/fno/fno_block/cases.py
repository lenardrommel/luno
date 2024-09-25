from pytest_cases import AUTO, parametrize

from typing import NamedTuple


class FNOBlockCase(NamedTuple):
    grid_shape_in: tuple[int, ...]
    num_channels_in: int

    num_modes: tuple[int, ...]

    grid_shape_out: tuple[int, ...]
    num_channels_out: int


@parametrize(
    "grid_shape,num_modes",
    (
        ((2,), (1,)),
        ((11,), (9,)),
        ((16, 32), (8, 8)),
        ((11, 13), (4, 6)),
    ),
    idgen=AUTO,
)
def case_truncation(
    grid_shape: tuple[int, ...], num_modes: tuple[int, ...]
) -> FNOBlockCase:
    return FNOBlockCase(
        grid_shape_in=grid_shape,
        num_channels_in=4,
        num_modes=num_modes,
        grid_shape_out=grid_shape,
        num_channels_out=2,
    )


@parametrize("grid_shape", ((1,), (2,), (5,), (16, 16)), idgen=AUTO)
def case_no_truncation(grid_shape: tuple[int, ...]) -> FNOBlockCase:
    return FNOBlockCase(
        grid_shape_in=grid_shape,
        num_channels_in=3,
        num_modes=grid_shape,
        grid_shape_out=grid_shape,
        num_channels_out=3,
    )


@parametrize(
    "grid_shape_in,num_modes,grid_shape_out",
    (
        ((3,), (3,), (6,)),
        ((5,), (2,), (3,)),
        ((16, 8), (3, 4), (33, 12)),
    ),
    idgen=AUTO,
)
def case_interpolation(
    grid_shape_in: tuple[int, ...],
    num_modes: tuple[int, ...],
    grid_shape_out: tuple[int, ...],
) -> FNOBlockCase:
    return FNOBlockCase(
        grid_shape_in=grid_shape_in,
        num_channels_in=1,
        num_modes=num_modes,
        grid_shape_out=grid_shape_out,
        num_channels_out=5,
    )
