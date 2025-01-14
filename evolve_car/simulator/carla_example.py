#!/usr/bin/env python3

import carla
import random
import time
import os
import numpy as np
import cv2

def main():
    ############################
    # Connection to the server #
    ############################
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Increase if needed

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    #################
    # Spawn Vehicle #
    #################
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in the current map.")

    vehicle_spawn = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, vehicle_spawn)
    if not vehicle:
        raise RuntimeError("Failed to spawn vehicle. Try adjusting spawn point or blueprint.")

    print(f"[INFO] Vehicle spawned at {vehicle_spawn.location}")

    # Enable autopilot (simple Traffic Manager autopilot)
    vehicle.set_autopilot(True)

    #################
    # Setup Cameras #
    #################
    # All cameras share the same blueprint with the same attributes
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")

    # Define 6 different placements for the cameras:
    # 1) Front
    # 2) Back
    # 3) Two on the left side (slightly different angles)
    # 4) Two on the right side (slightly different angles)
    camera_setups = [
        ("front", carla.Transform(
            carla.Location(x=1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))),

        ("back", carla.Transform(
            carla.Location(x=-1.5, y=0.0, z=2.4),
            carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))),

        ("left1", carla.Transform(
            carla.Location(x=0.0, y=-1.2, z=2.4),
            carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0))),

        ("left2", carla.Transform(
            carla.Location(x=0.0, y=-1.2, z=2.4),
            carla.Rotation(pitch=0.0, yaw=-120.0, roll=0.0))),

        ("right1", carla.Transform(
            carla.Location(x=0.0, y=1.2, z=2.4),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0))),

        ("right2", carla.Transform(
            carla.Location(x=0.0, y=1.2, z=2.4),
            carla.Rotation(pitch=0.0, yaw=120.0, roll=0.0))),
    ]

    # Keep references to camera actors so we can destroy them later
    cameras = []

    # Create a main folder to store images
    main_dir = "multi_cameras"
    os.makedirs(main_dir, exist_ok=True)

    # We'll maintain a frame counter for each camera
    frame_counters = {}

    # Before spawning each camera, create a subfolder for that camera
    for (cam_id, _) in camera_setups:
        subfolder = os.path.join(main_dir, cam_id)
        os.makedirs(subfolder, exist_ok=True)
        frame_counters[cam_id] = 0

    # Callback function to save images
    def save_image(image, cam_id):
        # Convert raw_data to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        # Convert BGRA to RGB
        array = array[:, :, :3][:, :, ::-1]
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        # Construct subfolder path and filename
        subfolder = os.path.join(main_dir, cam_id)
        filename = os.path.join(subfolder, f"frame_{frame_counters[cam_id]:05d}.png")

        # Increment the camera's frame count
        frame_counters[cam_id] += 1

        # Save image with OpenCV
        cv2.imwrite(filename, array)
        print(f"[INFO] {cam_id} saved {filename}")

    # Spawn and listen for each camera
    for cam_id, cam_transform in camera_setups:
        camera_actor = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
        camera_actor.listen(lambda image, cid=cam_id: save_image(image, cid))
        cameras.append(camera_actor)
        print(f"[INFO] Camera '{cam_id}' spawned and listening.")

    ###################
    # Simulation loop #
    ###################
    try:
        print("[INFO] Press Ctrl+C to quit.")
        while True:
            # Wait for the server tick
            world.wait_for_tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Cleaning up...")
    finally:
        # Clean up cameras
        for cam in cameras:
            if cam.is_alive:
                cam.stop()
                cam.destroy()

        # Destroy vehicle
        if vehicle.is_alive:
            vehicle.destroy()

        print("[INFO] All actors destroyed. Bye!")

if __name__ == "__main__":
    main()
