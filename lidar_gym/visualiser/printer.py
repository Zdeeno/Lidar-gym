import numpy as np


def ray_string(action_in):
    # create string to visualise action in console
    to_print = np.empty(action_in.shape, dtype=str)
    divider = np.empty(action_in.shape[0] + 2, dtype=str)
    divider[:] = '-'
    to_print[:] = ' '
    to_print[action_in] = '+'
    ret = '\n'
    ret += ''.join(divider)
    ret += '\n'
    for i in range(action_in.shape[1]):
        ret += '|'
        ret += ''.join(to_print[:, i])
        ret += '|\n'
    ret += ''.join(divider) + '\n\n'
    return ret
