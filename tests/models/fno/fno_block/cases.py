from pytest_cases import AUTO, parametrize

from typing import NamedTuple


class FNOBlockCase(NamedTuple):
    input_grid_shape: tuple[int, ...]
    num_input_channels: int

    modes_shape: tuple[int, ...]

    output_grid_shape: tuple[int, ...] | None
    num_output_channels: int


@parametrize(
    "grid_shape,modes_shape",
    (
        ((32,), (16,)),
        ((2,), (1,)),
        ((11,), (6,)),
        ((16, 32), (12, 12)),
        ((11, 13), (4, 6)),
    ),
    idgen=AUTO,
)
def case_truncation(
    grid_shape: tuple[int, ...], modes_shape: tuple[int, ...]
) -> FNOBlockCase:
    return FNOBlockCase(
        input_grid_shape=grid_shape,
        num_input_channels=4,
        modes_shape=modes_shape,
        output_grid_shape=None,
        num_output_channels=2,
    )


@parametrize(
    "grid_shape",
    (
        (16,),
        (1,),
        (2,),
        (3,),
        (12, 12),
        (8, 8, 8),
    ),
    idgen=AUTO,
)
def case_no_truncation(grid_shape: tuple[int, ...]) -> FNOBlockCase:
    return FNOBlockCase(
        input_grid_shape=grid_shape,
        num_input_channels=3,
        modes_shape=grid_shape[:-1] + (grid_shape[-1] // 2 + 1,),
        output_grid_shape=None,
        num_output_channels=3,
    )


@parametrize(
    "input_grid_shape,modes_shape,output_grid_shape",
    (
        ((3,), (3,), (6,)),
        ((5,), (2,), (3,)),
        ((16, 8), (3, 4), (33, 12)),
    ),
    idgen=AUTO,
)
def case_interpolation(
    input_grid_shape: tuple[int, ...],
    modes_shape: tuple[int, ...],
    output_grid_shape: tuple[int, ...],
) -> FNOBlockCase:
    return FNOBlockCase(
        input_grid_shape=input_grid_shape,
        num_input_channels=1,
        modes_shape=modes_shape,
        output_grid_shape=output_grid_shape,
        num_output_channels=5,
    )
