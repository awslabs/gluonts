import os
import numpy as np

import consts
from data.ode_solvers import rungekutta4


class Environment(object):
    def __init__(self, n_obs, n_state, n_ctrl, dt):
        self.n_obs = n_obs
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.dt = dt

    def enter(self, *args, **kwargs):
        raise NotImplemented()

    def observe(self, *args, **kwargs):
        raise NotImplemented()

    def transit(self, ctrl):
        raise NotImplemented()


class Pendulum3DCoordEnvironment(Environment):
    def __init__(self,
                 dt=0.1,
                 center_pendulum=np.array((0, 0, 2.0)),
                 cov_pendulum=np.array((0.1, 0.2, 0.1)),
                 noise_std_obs=0.05,
                 radius=1.0, g=9.81, mass=1.0, dampening=0.25,
                 noise_std_dyn=0.0,
                 f=1.0,  # focal length
                 ):
        super().__init__(
            n_obs=2,
            n_state=2,
            # We use first order ODE representation --> state is (angle, d/dt angle)
            n_ctrl=None,  # skip ctrl for now
            dt=dt,
        )
        self.g = g
        self.mass = mass
        self.dampening = dampening
        self.cov_pendulum = cov_pendulum
        self.noise_std_obs = noise_std_obs
        self.noise_std_dyn = noise_std_dyn
        self.center_pendulum = center_pendulum
        # initial position of pendulum. last entry is the distance from the camera, assumed to
        # have the same initial coordinate system, with just a translation in direction z.
        self.r_rotation_pendulum = radius
        self.f = f

    def _stochastic_differential_equation(self, state, t):
        w = np.random.normal(loc=0.0, scale=self.noise_std_dyn)
        return np.array([state[1], - ((self.g / 2.0) * np.sin(state[0])
                                      + (self.dampening / self.mass) * state[
                                          1] + w)])

    def enter(self):
        self.state = np.zeros(self.n_state)
        self.state[0] = np.pi + np.clip(np.random.randn(1), -2, 2) * 1.0
        self.state[1] = np.clip(np.random.randn(1), -2, 2) * 4.0
        self.t = 0.0

    def transit(self, ctrl):
        self.state, self.t = rungekutta4(state=self.state, t=self.t, dt=self.dt,
                                         f=self._stochastic_differential_equation)
        return self.state, self.t

    def observe(self, perspective):
        pendulum_loc, _ = self.project_to_pixel_space(perspective=perspective)
        pendulum_loc_noisy = pendulum_loc + np.random.normal(
            loc=0, scale=self.noise_std_obs, size=pendulum_loc.shape)
        return pendulum_loc_noisy, pendulum_loc

    def project_to_pixel_space(self, perspective):
        """
        1) Rotate pendulum, represented as 3D normal distribution,
        2) rotate Camera to some perspective with fixed distance to center of pendulum rotation point,
        3) project the scene into image space
        4) and further into pixel space
        This function works with homogenous coordinates (hom)
        and assumes the initial camera location at (0, 0, 0).
        The transformed covariance matrix is calculated by transforming the vectors of variance directions.
        :param angle: float, angle of pendulum, around z-axis
        :param perspective: list or tuple of length 2, rotation of camera arond x and y axis.
        :return: 1D-array: pixel values of projected image
        -------------------------------------------

        Example:
        dims_img = (16, 16, 1)
        env = PendulumWithPerspectiveEnvironment(dims_img=dims_img)
        env.enter()
        for angle in np.linspace(0, 2 * np.pi, 20):
            plt.cla()
            obs = env.observe(state=angle, perspective=[0, np.pi / 4]).reshape(dims_img[:2])
            plt.imshow(obs, cmap='gray')
            plt.pause(0.02)
        """
        # 1) Rotate Pendulum around z-axis
        angle = self.state[0]
        M_pendulum = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                               [np.sin(angle), np.cos(angle), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        mu_pendulum_hom = np.append(self.center_pendulum
                                    - np.array(
            [0, self.r_rotation_pendulum, 0]), 1)
        cov_pendulum_hom = np.diag(np.append(self.cov_pendulum, 1))

        mu_pendulum_rotated = M_pendulum.dot(mu_pendulum_hom)
        cov_pendulum_rotated = M_pendulum.dot(cov_pendulum_hom).dot(
            M_pendulum.T)

        # 2) Rotate Camera around pendulum center on x-axis and/or y-axis
        x, y, z = self.center_pendulum
        T_camera = np.array([[1, 0, 0, -x],
                             [0, 1, 0, -y],
                             [0, 0, 1, -z],
                             [0, 0, 0, 1]])
        inv_T_camera = np.array([[1, 0, 0, x],
                                 [0, 1, 0, y],
                                 [0, 0, 1, z],
                                 [0, 0, 0, 1]])

        c = np.cos(perspective[0])
        s = np.sin(perspective[0])
        rx = np.array([[1, 0, 0, 0],
                       [0, c, -s, 0],
                       [0, s, c, 0],
                       [0, 0, 0, 1]])

        c = np.cos(perspective[1])
        s = np.sin(perspective[1])
        ry = np.array([[c, 0, s, 0],
                       [0, 1, 0, 0],
                       [-s, 0, c, 0],
                       [0, 0, 0, 1]])

        R_camera = rx.dot(
            ry)  # rotation around center of camera, first rotate around y, then x.
        M_camera = inv_T_camera.dot(R_camera).dot(
            T_camera)  # rotation around center of pendulum

        # To rotate the mean, need to project it first to origin, then rotate, then project back
        mu_pendulum_camera = M_camera.dot(mu_pendulum_rotated)
        # The vectors for cov-matrix are already vectors w.r.t. origin, no need to translate
        cov_pendulum_camera = R_camera.dot(cov_pendulum_rotated).dot(R_camera.T)

        # 3) Project 3D coordinates of mu and cov vector
        # into image-space, using focal length of camera
        f = self.f
        F = np.array([[f, 0, 0, 0],
                      [0, f, 0, 0],
                      [0, 0, 1, 0]])
        mu_pendulum_projected = np.dot(F, mu_pendulum_camera)
        cov_pendulum_projected = np.dot(F, cov_pendulum_camera).dot(F.T)

        # 4) Transform to Pixel coordinates
        M_pixel = np.array([[1 / mu_pendulum_projected[-1], 0, 0],
                            [0, 1 / mu_pendulum_projected[-1], 0]])
        mu_pendulum_pixel = M_pixel.dot(mu_pendulum_projected)
        cov_pendulum_pixel = M_pixel.dot(cov_pendulum_projected).dot(M_pixel.T)
        return mu_pendulum_pixel, cov_pendulum_pixel


def generate_dataset(seed=42):
    from utils.local_seed import local_seed
    n_train = 5000
    n_test = 1000
    n_timesteps = 150
    perspective = (0.0, 0.0)

    data_path = os.path.join(consts.data_dir, consts.Datasets.pendulum_3D_coord)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def generate_sequence(n_timesteps_obs, n_steps_sim_per_obs=10, dt_sim=0.01):
        assert isinstance(n_steps_sim_per_obs, int) and n_steps_sim_per_obs > 0
        env = Pendulum3DCoordEnvironment(dt=dt_sim)
        env.enter()
        observations = []
        observations_gt = []
        states = []
        for t_obs in range(n_timesteps_obs):
            obs, obs_gt = env.observe(perspective=perspective)
            observations.append(obs)
            observations_gt.append(obs_gt)
            states.append(env.state)
            for t_sim in range(n_steps_sim_per_obs):
                env.transit(ctrl=None)
        observations = np.stack(observations, axis=0)
        observations_gt = np.stack(observations_gt, axis=0)
        states = np.stack(states, axis=0)
        return observations, observations_gt, states

    def generate_dataset(n_data, n_timesteps):
        observations_dataset = []
        observations_gt_dataset = []
        states_dataset = []
        for idx_sample in range(n_data):
            if idx_sample % 100 == 0:
                print(f"sequence {idx_sample}/{n_data}")
            observations, observations_gt, states = generate_sequence(
                n_timesteps_obs=n_timesteps)
            observations_dataset.append(observations)
            observations_gt_dataset.append(observations_gt)
            states_dataset.append(states)
        observations_dataset = np.stack(observations_dataset, axis=0)
        observations_gt_dataset = np.stack(observations_gt_dataset, axis=0)
        states_dataset = np.stack(states_dataset, axis=0)
        return observations_dataset, observations_gt_dataset, states_dataset

    with local_seed(seed=seed):
        observations_train, observations_gt_train, states_train = generate_dataset(
            n_data=n_train, n_timesteps=n_timesteps)
        observations_test, observations_gt_test, states_test = generate_dataset(
            n_data=n_test, n_timesteps=n_timesteps)

    np.savez(os.path.join(data_path, "train.npz"),
             obs=observations_train,
             obs_gt=observations_gt_train,
             state=states_train)
    np.savez(os.path.join(data_path, "test.npz"),
             obs=observations_test,
             obs_gt=observations_gt_test,
             state=states_test)


if __name__ == '__main__':
    generate_dataset()
