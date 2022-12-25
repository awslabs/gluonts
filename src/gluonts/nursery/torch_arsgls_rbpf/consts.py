import os

home_dir = os.environ["HOME"]
data_dir = os.path.join(home_dir, "data")  # consider saving on other disk.
log_dir = os.path.join(home_dir, "logs")  # consider saving on other disk.


class Datasets:
    pendulum_3D_image = "pendulum_3D_image"
    pendulum_3D_coord = "pendulum_3D_coord"
    box = "box"
    box_gravity = "box_gravity"
    polygon = "polygon"
    pong = "pong"
