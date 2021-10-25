import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase, run_tests

from ..forward_backward import (
    forward_pass_hmm,
    backward_pass_hmm,
    forward_backward_hmm,
    forward_pass_hsmm,
    backward_pass_hsmm,
    forward_backward_hsmm,
)


class ForwardBackwardTest(TestCase):
    def setUp(self):
        super().setUp()
        self.init_pi = torch.tensor([0.5, 0.5]).view(1, 2)

        self.mat_a = torch.tensor(
            np.array(
                [
                    [
                        [[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 0.5]],
                    ]
                ],
                dtype=np.float32,
            )
        )

        self.mat_b = torch.tensor(
            np.array(
                [[[0.5, 0.75], [0.5, 0.75], [0.5, 0.25], [0.5, 0.25]]],
                dtype=np.float32,
            )
        )

        self.u = (
            torch.tensor(
                np.array(
                    [[1 / 2, 2 / 5, 0], [9 / 10, 2 / 3, 0]], dtype=np.float32
                )
            )
            .view(1, 1, 2, 3)
            .repeat(1, 3, 1, 1)
        )
        self.eps = torch.finfo(torch.float32).eps

    def test_forward_hmm(self):
        fwd_logprob, fwd_lognorm = forward_pass_hmm(
            torch.log(self.mat_a),
            torch.log(self.mat_b),
            torch.log(self.init_pi),
        )
        fwd_prob = torch.exp(fwd_logprob)
        fwd_norm = torch.cumprod(torch.exp(fwd_lognorm), dim=1)
        fwd_norm = fwd_norm[:, :, None]
        target_value = np.array(
            [
                [
                    [1.0 / 4.0, 3.0 / 8.0],
                    [5.0 / 32.0, 15.0 / 64.0],
                    [25.0 / 256.0, 25.0 / 512.0],
                    [75.0 / 2048.0, 75.0 / 4096.0],
                ]
            ],
            dtype=np.float32,
        )
        self.assertEqual(torch.tensor(target_value), fwd_prob * fwd_norm)

    def test_forward_hsmm(self):
        fwd_logprob = forward_pass_hsmm(
            torch.log(self.mat_a)[:, :-1, ...],
            torch.log(self.mat_b)[:, :-1, ...],
            torch.log(self.init_pi),
            torch.log(self.u + self.eps),
        )
        fwd_prob = torch.exp(fwd_logprob)
        target_value = np.array(
            [
                [
                    [[1.0 / 4.0, 0.0, 0.0], [3.0 / 8.0, 0.0, 0.0]],
                    [
                        [13.0 / 320.0, 1.0 / 16.0, 0.0],
                        [39.0 / 640.0, 81.0 / 320.0, 0.0],
                    ],
                    [
                        [949.0 / 25600.0, 13.0 / 1280.0, 1.0 / 80.0],
                        [949.0 / 51200.0, 351.0 / 25600.0, 27.0 / 640],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        self.assertEqual(torch.tensor(target_value), fwd_prob)

    def test_backward_hsmm(self):
        fwd_logprob = backward_pass_hsmm(
            torch.log(self.mat_a)[:, :-1, ...],
            torch.log(self.mat_b)[:, :-1, ...],
            torch.log(self.init_pi),
            torch.log(self.u + self.eps),
        )
        fwd_prob = torch.exp(fwd_logprob)
        target_value = np.array(
            [
                [
                    [
                        [269.0 / 1280.0, 639.0 / 3200.0, 133.0 / 640.0],
                        [1393.0 / 6400.0, 493.0 / 1920.0, 133.0 / 640.0],
                    ],
                    [
                        [7.0 / 16.0, 17.0 / 40.0, 3.0 / 8.0],
                        [21.0 / 80.0, 7.0 / 24.0, 3.0 / 8.0],
                    ],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ],
            dtype=np.float32,
        )
        self.assertEqual(torch.tensor(target_value), fwd_prob)

    def test_forward_backward_hsmm(self):
        _, _, log_gamma, log_p_XY = forward_backward_hsmm(
            torch.log(self.mat_a)[:, :-1, ...],
            torch.log(self.mat_b)[:, :-1, ...],
            torch.log(self.init_pi),
            torch.log(self.u + self.eps),
        )
        gamma = torch.exp(log_gamma)
        p_XY = torch.exp(log_p_XY)
        target_p_XY = np.asarray([6869.0 / 51200.0], dtype=np.float32)
        target_gamma = (
            np.array(
                [
                    [
                        [
                            [269.0 / 5120.0, 0.0, 0.0],
                            [4179.0 / 51200.0, 0.0, 0.0],
                        ],
                        [
                            [91.0 / 5120.0, 17.0 / 640.0, 0.0],
                            [819.0 / 51200.0, 189.0 / 2560.0, 0.0],
                        ],
                        [
                            [949.0 / 25600.0, 13.0 / 1280.0, 1.0 / 80.0],
                            [949.0 / 51200.0, 351.0 / 25600.0, 27.0 / 640.0],
                        ],
                    ]
                ],
                dtype=np.float32,
            )
            / target_p_XY
        )
        self.assertEqual(torch.tensor(target_gamma), gamma)
        self.assertEqual(torch.tensor(p_XY), target_p_XY)

    def test_backward_hmm(self):
        bwd_logprob = backward_pass_hmm(
            torch.log(self.mat_a),
            torch.log(self.mat_b),
            torch.log(self.init_pi),
        )
        bwd_prob = torch.exp(bwd_logprob)

        target_value = np.array(
            [
                [
                    [45.0 / 512.0, 45.0 / 512.0],
                    [9.0 / 64.0, 9.0 / 64.0],
                    [3.0 / 8.0, 3.0 / 8.0],
                    [1.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        self.assertEqual(torch.tensor(target_value), bwd_prob)

    def test_forward_backward_hmm(self):
        _, _, log_gamma, log_xi, log_data = forward_backward_hmm(
            torch.log(self.mat_a),
            torch.log(self.mat_b),
            torch.log(self.init_pi),
        )
        gamma, xi = torch.exp(log_gamma), torch.exp(log_xi)
        gamma_target = np.array(
            [
                [
                    [90.0 / 225.0, 135.0 / 225.0],
                    [90.0 / 225.0, 135.0 / 225.0],
                    [150.0 / 225.0, 75.0 / 225.0],
                    [150.0 / 225.0, 75.0 / 225.0],
                ]
            ],
            dtype=np.float32,
        )
        xi_target = np.array(
            [
                [
                    [[1 / 4, 1 / 4], [1 / 4, 1 / 4]],
                    [
                        [36.0 / 225.0, 54.0 / 225.0],
                        [54.0 / 225.0, 81.0 / 225.0],
                    ],
                    [
                        [60.0 / 225.0, 90.0 / 225.0],
                        [30.0 / 225.0, 45.0 / 225.0],
                    ],
                    [
                        [100.0 / 225.0, 50.0 / 225.0],
                        [50.0 / 225.0, 25.0 / 225.0],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        target_log_data = np.log(np.asarray([225 / 4096], dtype=np.float32))
        print(xi, xi_target)
        self.assertEqual(torch.tensor(target_log_data), log_data)
        self.assertEqual(torch.tensor(gamma_target), gamma)
        self.assertEqual(torch.tensor(xi_target), xi)


if __name__ == "__main__":
    run_tests()
