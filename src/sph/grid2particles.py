import numpy as np
from scipy import interpolate


def W_grid_2_particles(W_grid, W_df):
    xs, ys = W_grid.shape

    # num_channel should be 6 including mass, inertia, 
    # vx, vy, angle, spin 
    num_channel = 6
    z = np.zeros((xs, ys, num_channel))
    for x in range(xs):
        for y in range(ys):
            if not W_grid[x, y]:
                z[x, y, :] = np.zeros((1, 6))
                continue
            info_list = np.asarray(
                [
                    W_df.loc[i].values * w
                    for i, w in W_grid[x, y]
                ]
            )
            z[x, y, :] = np.sum(info_list, axis=0)

    xx, yy = np.meshgrid(range(xs), range(ys))
    f = interp2d(xx, yy, z)
    return f


# since interpolate.interp2d can just solve with
# condition that z is 1-D array
def interp2d(xx, yy, z, kind='linear'):
    num_channel = z.shape[2]
    f = [
        interpolate.interp2d(xx, yy, z[:, :, i], kind='cubic')
        for i in range(num_channel)
    ]

    return lambda x, y: np.matrix([d(x, y)[0] for d in f])
