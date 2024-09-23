import jax
import linox
from linox._arithmetic import CongruenceTransform, ScaledLinearOperator

########################################################################################
# Congruence Transforms ################################################################
########################################################################################


##########################################################
# (LinearOperator, IsotropicScalingPlusSymmetricLowRank) #
##########################################################


class CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank(
    CongruenceTransform
):
    pass


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: linox.IsotropicScalingPlusSymmetricLowRank,
) -> CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank:
    return CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank(A, B)


@linox.diagonal.dispatch
def _(
    A: CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank,
) -> jax.Array:
    return sum(
        linox.diagonal(linox.congruence_transform(A.A, summand))
        for summand in A.B.operator_list
    )


##########################################
# (LinearOperator, ScaledLinearOperator) #
##########################################


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: ScaledLinearOperator,
) -> ScaledLinearOperator:
    return B.scalar * linox.congruence_transform(A, B.operator)
