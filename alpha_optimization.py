from dE_dalpha import dE_dalpha

def alpha_opt_on_fly(position_vec, alpha):
    dE_da = dE_dalpha(position_vec, alpha)
    alpha = alpha - 0.01 * dE_da  # Learning rate = 0.01
    return alpha, dE_da
