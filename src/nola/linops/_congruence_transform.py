import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform

########################################################################################
# (LinearOperator, IsotropicScalingPlusLowRank) ########################################
########################################################################################


class CongruenceTransform_LinearOperator_IsotropicScalingPlusLowRank(
    CongruenceTransform
):
    pass


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: linox.IsotropicScalingPlusLowRank,
) -> CongruenceTransform_LinearOperator_IsotropicScalingPlusLowRank:
    return CongruenceTransform_LinearOperator_IsotropicScalingPlusLowRank(A, B)


@linox.diagonal.dispatch
def _(A: CongruenceTransform_LinearOperator_IsotropicScalingPlusLowRank) -> jax.Array:
    # TODO: IsotropicScalingPlusLowRank should inherit from AddLinearOperator
    B_summands = (linox.Scalar(A.B.scalar), linox.LowRank(A.B.U, A.B.S))

    return sum(
        linox.diagonal(linox.congruence_transform(A.A, summand))
        for summand in B_summands
    )


########################################################################################
# (BlockMatrix, Scalar) ################################################################
########################################################################################


class CongruenceTransform_BlockMatrix_Scalar(CongruenceTransform):
    pass


@linox.congruence_transform.dispatch
def _(A: linox.BlockMatrix, B: linox.Scalar) -> CongruenceTransform_BlockMatrix_Scalar:
    return CongruenceTransform_BlockMatrix_Scalar(A, B)


@linox.diagonal.dispatch
def _(A: CongruenceTransform_BlockMatrix_Scalar) -> jax.Array:
    blocks = A.A.blocks

    return jnp.concatenate(
        [
            sum(
                linox.diagonal(linox.congruence_transform(block, A.B))
                for block in block_row
            )
            for block_row in range(len(blocks))
        ]
    )
