import os
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

import consts
from data.pendulum3D.pendulum3D_coord_environment import (
    Pendulum3DCoordEnvironment,
)


class Pendulum3DImageEnvironment(Pendulum3DCoordEnvironment):
    """
    This class can be used to generate images from a pendulum, swinging in 3D around the z-axis.
    The swing is only in 2 dimensions, but the camera perspective is in 3D.
    camera rotates around x or y axis, whereas pendulum rotates around z axis.
    """

    def __init__(
        self,
        center_pendulum=np.array((0, 0, 2.0)),
        cov_pendulum=np.array((0.1, 0.2, 0.1)),
        noise_std_obs=0.2,
        radius=1.0,
        g=9.81,
        mass=1.0,
        dampening=0.25,
        noise_std_dyn=0.0,
        dt=0.1,
        dims_img=(32, 32,),
        f=1,
        cx=1,
        cy=1,
    ):
        """
        These params depend on each other.
        the ratio of focal length and distance of the pendulum (z-axis of center-pendulum)
        define where the image will be projected in image space. cx and cy further define
        the position in pixel space. f*r/center_pendulum[-1] == 1 and cx = cy = 2 is default choice.
        The distance in z and the rotation radius define how much the pendulum will appear bigger
        or smaller due to perspective.

        :param focal_length: The focal length is the length between the focal point of the camera
        :param radius: radius of rotation around z-axis
        :param center_pendulum: 3D world coordinates of pendulum center (cameras is origin)
        :param cx: pixel locations are [-cx, ..., cx]
        :param cy: same
        :param cov_pendulum: size of pendulum in the 3 dimensions
        :param dims_img: number of pixels of the form [nx, ny]
        """

        assert len(dims_img) == 2
        super().__init__(
            dt=dt,
            g=g,
            center_pendulum=center_pendulum,
            mass=mass,
            dampening=dampening,
            radius=radius,
            cov_pendulum=cov_pendulum,
            noise_std_dyn=noise_std_dyn,
            noise_std_obs=noise_std_obs,
            f=f,
        )
        self.dims_img = dims_img
        self.n_obs = np.prod(
            dims_img
        )  # required to overwrite this due to bad design...
        self.cx, self.cy = cx, cy

    def observe(self, perspective):
        mu_pendulum_pixel, cov_pendulum_pixel = self.project_to_pixel_space(
            perspective=perspective
        )
        x = np.linspace(-self.cx, self.cx, self.dims_img[0])
        # Image vertical dimensions go from top to bottom.
        # --> Invert by starting with positive numbers top .
        y = np.linspace(self.cy, -self.cy, self.dims_img[1])
        xv, yv = np.meshgrid(x, y)
        pixels = np.concatenate(
            (xv.ravel()[:, np.newaxis], yv.ravel()[:, np.newaxis]), 1
        )

        img = scipy.stats.multivariate_normal.pdf(
            pixels, mean=mu_pendulum_pixel, cov=cov_pendulum_pixel
        )
        img_noisy = img + np.random.normal(
            loc=0, scale=self.noise_std_obs, size=img.shape
        )
        return img_noisy, img

    def plot(self, observations, n_samples=3, n_timesteps=None):
        assert observations.ndim == 3
        assert observations.shape[-1] == self.n_obs
        if n_timesteps is None or n_timesteps > observations.shape[0]:
            n_timesteps = observations.shape[0]
        if n_samples is None or n_samples > observations.shape[1]:
            n_samples = observations.shape[1]

        fig, axs = plt.subplots(
            n_samples,
            n_timesteps,
            figsize=(n_timesteps * 2, n_samples * 2),
            squeeze=False,
        )
        plt.tight_layout()
        for t in range(n_timesteps):
            for n in range(n_samples):
                axs[n, t].set_xticks([])
                axs[n, t].set_yticks([])

        for t in range(n_timesteps):
            for n in range(n_samples):
                obs = observations[t, n, : self.dims_img[0] * self.dims_img[1]]
                img = obs.reshape((self.dims_img[0], self.dims_img[1]))
                axs[n, t].imshow(img, cmap="binary", interpolation="none")
        return fig, axs

    def plot_mean(self, observations, n_timesteps=None):
        assert observations.ndim == 3  # TxBxF
        assert observations.shape[-1] == self.n_obs

        if n_timesteps == None:
            n_timesteps = observations.shape[0]
        fig, axs = plt.subplots(
            1, n_timesteps, figsize=(n_timesteps * 2, 2), squeeze=True
        )
        fig.tight_layout()
        for t in range(n_timesteps):
            axs[t].set_xticks([])
            axs[t].set_yticks([])

        for t in range(n_timesteps):
            obs = np.mean(
                observations[t, : self.dims_img[0] * self.dims_img[1]], axis=0
            )
            img = obs.reshape((self.dims_img[0], self.dims_img[1]))
            axs[t].imshow(img, cmap="binary", interpolation="none")
        return fig, axs


def generate_dataset(seed=42):
    from utils.local_seed import local_seed

    n_train = 5000
    n_test = 1000
    n_timesteps = 150
    dims_img = (
        32,
        32,
    )
    perspective = (0.0, 0.0)

    data_path = os.path.join(
        consts.data_dir, consts.Datasets.pendulum_3D_image
    )
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def generate_sequence(
        n_timesteps_obs, n_steps_sim_per_obs=10, dt_sim=0.01
    ):
        assert isinstance(n_steps_sim_per_obs, int) and n_steps_sim_per_obs > 0
        env = Pendulum3DImageEnvironment(dt=dt_sim, dims_img=dims_img)
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
                n_timesteps_obs=n_timesteps
            )
            observations_dataset.append(observations)
            observations_gt_dataset.append(observations_gt)
            states_dataset.append(states)
        observations_dataset = np.stack(observations_dataset, axis=0)
        observations_gt_dataset = np.stack(observations_gt_dataset, axis=0)
        states_dataset = np.stack(states_dataset, axis=0)
        return observations_dataset, observations_gt_dataset, states_dataset

    with local_seed(seed=seed):
        (
            observations_train,
            observations_gt_train,
            states_train,
        ) = generate_dataset(n_data=n_train, n_timesteps=n_timesteps)
        (
            observations_test,
            observations_gt_test,
            states_test,
        ) = generate_dataset(n_data=n_test, n_timesteps=n_timesteps)

    np.savez(
        os.path.join(data_path, "train.npz"),
        obs=observations_train,
        obs_gt=observations_gt_train,
        state=states_train,
    )
    np.savez(
        os.path.join(data_path, "test.npz"),
        obs=observations_test,
        obs_gt=observations_gt_test,
        state=states_test,
    )


if __name__ == "__main__":
    generate_dataset()
