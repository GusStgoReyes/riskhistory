import numpy as np

##### DRIFT FUNCTIONS #####
def drift_linear(gain, v_gain, 
                loss, v_loss, 
                alpha, v_Intercept, 
                choice_prev1, v_choice_prev1,
                choice_prev2, v_choice_prev2, 
                choice_prev3, v_choice_prev3,
                choice_prev4, v_choice_prev4,
                choice_prev5, v_choice_prev5,
                choice_prev6, v_choice_prev6,
                v_loss_choice_prev1, v_gain_choice_prev1,
                v_loss_choice_prev2, v_gain_choice_prev2,
                v_loss_choice_prev3, v_gain_choice_prev3,
                v_loss_choice_prev4, v_gain_choice_prev4,
                v_loss_choice_prev5, v_gain_choice_prev5,
                v_loss_choice_prev6, v_gain_choice_prev6,
                ):
    """
        Drift function for the DDM model using prospect theory with lagged choice effects.
    """
    # Compute the subjective value based on prospect theory
    svs = v_gain * np.power(gain, alpha) + v_loss * np.power(loss, alpha)
    svs += v_Intercept

    # Add the lagged choice effects
    svs += v_choice_prev1 * choice_prev1
    svs += v_choice_prev2 * choice_prev2
    svs += v_choice_prev3 * choice_prev3
    svs += v_choice_prev4 * choice_prev4
    svs += v_choice_prev5 * choice_prev5
    svs += v_choice_prev6 * choice_prev6

    # Add the gain and loss effects
    svs += choice_prev1 * v_gain_choice_prev1 * gain
    svs += choice_prev1 * v_loss_choice_prev1 * loss
    svs += choice_prev2 * v_gain_choice_prev2 * gain
    svs += choice_prev2 * v_loss_choice_prev2 * loss
    svs += choice_prev3 * v_gain_choice_prev3 * gain
    svs += choice_prev3 * v_loss_choice_prev3 * loss
    svs += choice_prev4 * v_gain_choice_prev4 * gain
    svs += choice_prev4 * v_loss_choice_prev4 * loss
    svs += choice_prev5 * v_gain_choice_prev5 * gain
    svs += choice_prev5 * v_loss_choice_prev5 * loss
    svs += choice_prev6 * v_gain_choice_prev6 * gain
    svs += choice_prev6 * v_loss_choice_prev6 * loss

    return svs

##### BOUND FUNCTIONS #####
def bound(a, theta, t):
    """
    Bound function for the DDM model.
        - theta is in radians and should be in the range of [0, pi/2]
        - a is the boundary separation and should be positive 
        - LAN (Fengler, A., Frank, M., et al. (2021)) paper suggests good parameter recoverability for this boundary function
          as opposed to the weibull, for example. 
    """
    return a - np.tan(theta) * t

##### STARTING POINT FUNCTIONS #####
def startingposition(z,
                     choice_prev1, z_choice_prev1,
                     choice_prev2, z_choice_prev2,
                     choice_prev3, z_choice_prev3,
                     choice_prev4, z_choice_prev4,
                     choice_prev5, z_choice_prev5,
                     choice_prev6, z_choice_prev6):
    """
    Starting position function for the DDM model with lagged choice effects.
     - z lies between -1 and 1
    """
    s = z

    # Add the lagged choice effect if it accepted
    s += z_choice_prev1 * choice_prev1
    s += z_choice_prev2 * choice_prev2
    s += z_choice_prev3 * choice_prev3
    s += z_choice_prev4 * choice_prev4
    s += z_choice_prev5 * choice_prev5
    s += z_choice_prev6 * choice_prev6

    # Ensure starting position is within bounds [-1, 1]
    s = 1/(1 + np.exp(-s)) * 2 - 1

    return s

##### NON-DECISION TIME FUNCTION #####
def nondecision(ter):
    """
    Non-decision time function for the DDM model.
        - ter is the non-decision time in seconds and should be positive
    """
    return ter

##### NOISE FUNCTION #####
def noise(noise):
    """
    Noise function for the DDM model.
        - noise is the standard deviation of the noise in the drift rate
    """
    return noise