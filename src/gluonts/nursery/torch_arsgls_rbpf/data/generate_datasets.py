from data.pymunk_kvae import box, box_gravity, polygon, pong
from data.pendulum3D import pendulum3D_coord_environment

if __name__ == "__main__":
    box.generate_dataset()
    box_gravity.generate_dataset()
    polygon.generate_dataset()
    pong.generate_dataset()
    pendulum3D_coord_environment.generate_dataset()
    # pendulum3D_image_environment.generate_dataset()
