import pygame
import pymunk.pygame_util
import numpy as np
import os
from pygame.color import THECOLORS as color
import consts


# Note: This code is taken from
# https://github.com/simonkamronn/kvae/blob/master/kvae/datasets/box.py
# Made only minor adjustments, e.g. the saving paths,
# number of time-steps for test data.

scale = 1


class Pong:
    def __init__(
        self, dt=0.2, res=(32, 32), init_pos=(3, 3), init_std=0, wall=None
    ):
        pygame.init()

        self.dt = dt
        self.res = res
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
            pygame.display.set_mode(res, 0, 24)
            self.screen = pygame.Surface(res, pygame.SRCCOLORKEY, 24)
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, res[0], res[1]), 0)
        else:
            self.screen = pygame.display.set_mode(res, 0, 24)
        self.gravity = (0.0, 0.0)
        self.initial_position = init_pos
        self.initial_std = init_std
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = None

        self.dd = 2

    def _clear(self):
        self.screen.fill(color["black"])

    class Paddle:
        def __init__(self, pong, position):
            self.pong = pong
            self.area = pong.res
            if position == "left":
                self.rect = pymunk.Segment(
                    pong.space.static_body,
                    (0, self.area[1] / 2 + 3 * scale),
                    (0, self.area[1] / 2 - 3 * scale),
                    1.0,
                )
            else:
                self.rect = pymunk.Segment(
                    pong.space.static_body,
                    (self.area[0] - 2, self.area[1] / 2 + 3 * scale),
                    (self.area[0] - 2, self.area[1] / 2 - 3 * scale),
                    1.0,
                )
            self.speed = 2 * scale
            self.rect.elasticity = 0.99
            self.rect.color = color["white"]
            self.rect.collision_type = 1

        def update(self, ball):
            a, b = self.rect.a, self.rect.b
            center = (a.y + b.y) / 2
            if ball.body.position.y < center - self.speed / 2:
                delta_y = min(b.y, self.speed)
                a.y -= delta_y
                b.y -= delta_y
                self.rect.unsafe_set_endpoints(a, b)
            if ball.body.position.y > center + self.speed / 2:
                delta_y = min(self.area[1] - a.y, self.speed)
                a.y += delta_y
                b.y += delta_y
                self.rect.unsafe_set_endpoints(a, b)
            return self.rect

    def add_walls(self):
        self.static_lines = []

        # Add floor
        self.static_lines.append(
            pymunk.Segment(
                self.space.static_body, (0, 1), (self.res[1], 1), 0.0
            )
        )

        # Add roof
        self.static_lines.append(
            pymunk.Segment(
                self.space.static_body,
                (0, self.res[1]),
                (self.res[1], self.res[1]),
                0.0,
            )
        )

        # Set properties
        for line in self.static_lines:
            line.elasticity = 0.99
            line.color = color["white"]
        self.space.add(self.static_lines)
        return True

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        position = np.array(
            self.initial_position
        ) + self.initial_std * np.random.normal(size=(2,))
        position = np.clip(
            position, self.dd + radius + 1, self.res[0] - self.dd - radius - 1
        )
        body.position = position

        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.9
        shape.color = color["white"]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def run(
        self,
        iterations=20,
        sequences=500,
        angle_limits=(0, 360),
        radius=3,
        save=None,
        filepath="/tmp/data/pong.npz",
        delay=None,
    ):
        if save:
            data = np.empty(
                (sequences, iterations, self.res[0], self.res[1]),
                dtype=np.float32,
            )
        controls = None, None

        # Add roof and floor
        self.add_walls()

        for s in range(sequences):
            if s % 100 == 0:
                print(f"sequence {s}/{sequences}")

            angle = np.random.uniform(*angle_limits)
            velocity = 10 * scale
            ball = self.fire(angle, velocity, radius)

            # Create pong paddles
            paddle1 = self.Paddle(self, "left")
            paddle2 = self.Paddle(self, "right")

            for i in range(iterations):
                self._clear()

                # Add paddles
                self.space.add(paddle1.update(ball))
                self.space.add(paddle2.update(ball))

                # Step
                self.space.step(self.dt)

                # Draw objects
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == "png":
                    pygame.image.save(
                        self.screen,
                        os.path.join(
                            filepath, "bouncing_balls_%02d_%02d.png" % (s, i)
                        ),
                    )
                elif save == "npz":
                    data[s, i] = (
                        pygame.surfarray.array2d(self.screen)
                        .swapaxes(1, 0)
                        .astype(np.float32)
                        / 255
                    )

                # Remove the paddles
                self.space.remove(paddle1.rect)
                self.space.remove(paddle2.rect)

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)

        if save == "npz":
            np.savez(os.path.abspath(filepath), images=data, controls=controls)


def generate_dataset(seed=1234, n_timesteps_train=20, n_timesteps_test=100):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    np.random.seed(seed=seed)

    # Create data dir
    data_path = os.path.join(consts.data_dir, consts.Datasets.pong)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    cannon = Pong(
        dt=0.2,
        res=(32 * scale, 32 * scale),
        init_pos=(16 * scale, 16 * scale),
        init_std=3,
        wall=None,
    )
    cannon.run(
        delay=None,
        iterations=n_timesteps_train,
        sequences=5000,
        radius=3 * scale,
        angle_limits=(0, 360),
        filepath=os.path.join(data_path, "train.npz"),
        save="npz",
    )
    np.random.seed(5678)
    cannon.run(
        delay=None,
        iterations=n_timesteps_test,
        sequences=1000,
        radius=3 * scale,
        angle_limits=(0, 360),
        filepath=os.path.join(data_path, "test.npz"),
        save="npz",
    )


if __name__ == "__main__":
    generate_dataset()
