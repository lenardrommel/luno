import jax
import linox
from linox._arithmetic import CongruenceTransform

#################################################
# (LinearOperator, IsotropicScalingPlusLowRank) #
#################################################


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
