import habitat_sim
# from habitat_sim.viewer import Viewer
import json
import os
import gzip
import numpy as np
from PIL import Image
import cv2
import magnum as mn
import segmentation_models_pytorch as smp

# Create a Habitat-Sim configuration with RGB, depth, and optional semantic sensors
def make_sim_config(scene_dataset_config_file, scene_id, width=1280, height=720, include_semantic=True):  
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config_file
    sim_cfg.scene_id = scene_id
    print("scene id: ", scene_id)
    sim_cfg.load_semantic_mesh = include_semantic

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    # RGB sensor specification using CameraSensorSpec
    print("Configuring rgb sensor")
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = np.array(mn.Vector3([0, 0.8, 0])) # 0.8m mimicks the height of a unitree go2
    rgb_sensor.hfov = 90

    print("Configuring depth sensor")
    # Depth sensor specification using CameraSensorSpec
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [height, width]
    rgb_sensor.hfov = 90
    rgb_sensor.position = np.array(mn.Vector3([0, 0.8, 0])) # 0.8m mimicks the height of a unitree go2

    sensors = [rgb_sensor, depth_sensor]

    if include_semantic:
        print("Adding semantic sensor!")
        semantic_sensor = habitat_sim.CameraSensorSpec()
        semantic_sensor.uuid = "semantic"
        semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor.resolution = [height, width]
        semantic_sensor.hfov = 90
        semantic_sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        rgb_sensor.position = np.array(mn.Vector3([0, 0.8, 0])) # 0.8m mimicks the height of a unitree go2
        sensors.append(semantic_sensor)

    agent_cfg.sensor_specifications = sensors

    # Define action space
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.4)),
        "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def save_rgb(image, path):
    img = Image.fromarray(image)
    img.save(path)

def save_depth(depth, path):
    np.save(path, depth)

def save_semantic(semantic, path):
    np.save(path, semantic)

def load_episode_json(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        return json.load(file)

def interactive_control(sim, agent, id_to_name, images_dir, target_object, include_semantic=True):
    trajectory = []
    actions = []
    step = 0
    print("\nInteractive Control Mode:")
    print("  Press W to move_forward")
    print("  Press A to turn_left")
    print("  Press D to turn_right")
    print("  Press R to RESTART this episode's trajectory collection")
    print("  Press Q to FINISH and SAVE this episode's trajectory")

    while True:
        obs = sim.get_sensor_observations()
        rgb = obs['rgb']
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        display_image = rgb_bgr.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        line_type = cv2.LINE_AA
        cv2.putText(display_image, f"Target: {target_object}", (10, 60), font, font_scale, color, thickness, line_type)
        cv2.imshow("Agent View", display_image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            print("Finishing trajectory collection for this episode.")
            cv2.destroyAllWindows()
            return actions, trajectory

        elif key == ord('r'): # New restart condition
            print("Restarting trajectory collection for this episode...")
            # Clear any previously saved images for this attempt
            for f in os.listdir(images_dir):
                if f.startswith("state_") and f.endswith("_rgb.png"):
                    os.remove(os.path.join(images_dir, f))
            cv2.destroyAllWindows()
            return "restart", None

        elif key == ord('w'):
            action = "move_forward"
        elif key == ord('a'):
            action = "turn_left"
        elif key == ord('d'):
            action = "turn_right"
        else:
            print("Invalid key. Use W, A, D, R, or Q.")
            continue

        # Save the RGB image before taking the step (represents the state for the chosen action)
        rgb_path = os.path.join(images_dir, f"state_{step:04d}_rgb.png")
        save_rgb(obs['rgb'], rgb_path) # Save the observation corresponding to the action taken

        # Take the step
        sim.step(action)
        actions.append(action)
        agent_state = agent.get_state()

        # Collect semantic info if enabled
        semantic_info = None
        visible_objects = None
        if include_semantic:
            # Semantic info is part of obs, but re-query agent state for position/rotation
            semantic_state = agent_state.sensor_states['semantic']
            print("Semantic sensor position:", semantic_state.position, "rotation:", semantic_state.rotation)
            semantic_info = obs.get('semantic') # Use .get for safety
            if semantic_info is not None:
                 visible_objects = get_object_bboxes(semantic_info, id_to_name)
            else:
                 print("Warning: Semantic sensor included but no semantic observation found.")

        trajectory.append({
            "step": step,
            "position": agent_state.position.tolist(),
            "orientation": [agent_state.rotation.x,
                            agent_state.rotation.y,
                            agent_state.rotation.z,
                            agent_state.rotation.w],
            "action": action,
            "rgb_image_path": os.path.relpath(rgb_path, start=os.path.dirname(images_dir)), # Store relative path
            # "depth": obs['depth'],
            # 'semantic': semantic_info,
            # 'visible_objects': visible_objects,
        })

        print(f"Step {step}: Action={action}, Position={agent_state.position.tolist()}")
        step += 1

# Launch the Habitat-Sim GUI viewer in a separate thread.
# def run_viewer(sim):
#     viewer = habitat_sim.viewer.Viewer(sim, "Habitat-Sim GUI")
#     viewer.run()

def get_object_bboxes(semantic, id_to_name):
    visible_objects = []
    categories = id_to_name['category_to_task_category_id']
    objects = list(categories.keys())
    object_ids = list(categories.values())

    for obj_id in object_ids:
        mask = (semantic == obj_id)
        if mask.sum() == 0:
            continue
        if obj_id != 0:
            print("Object found! ", objects[obj_id])
        ys, xs = np.where(mask)
        x_min, x_max = xs.min().item(), xs.max().item()
        y_min, y_max = ys.min().item(), ys.max().item()
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        index = object_ids.index(obj_id)
        object_name = objects[index]
        visible_objects.append({"label": object_name, "bbox": bbox})
    
    return visible_objects

def main():
    # Configuration paths
    scene_dataset_config_file = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    objectnav_dataset_path = "../../../data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2"
    output_dir = "../../../output_dataset"

    if not os.path.exists(scene_dataset_config_file):
        raise FileNotFoundError(f"Scene dataset config file not found at {scene_dataset_config_file}")
    if not os.path.exists(objectnav_dataset_path):
        raise FileNotFoundError(f"ObjectNav dataset not found at {objectnav_dataset_path}")

    os.makedirs(output_dir, exist_ok=True)

    with gzip.open(os.path.join(objectnav_dataset_path, 'val/val.json.gz'), 'rt', encoding='utf-8') as file:
        category_id_map = json.load(file)
    print("Category ID Map:", category_id_map)

    episode_files = [file for file in os.listdir('../../../data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/val/content') if file.endswith('.json.gz')]
    scene_episodes = []
    for episode in episode_files:
        episode_path = os.path.join('../../../data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/val/content', episode)
        scene_episodes.append(load_episode_json(episode_path))
    print("Number of episode files:", len(scene_episodes))
    if not scene_episodes:
        raise ValueError("No episodes found in the ObjectNav dataset")
    
    print("Number of scenes: ", len(scene_episodes))

    num_scenes_coll = 0
    for i in range(len(scene_episodes)):
        for ep_idx, episode in enumerate(scene_episodes[i]['episodes']):
            sim = None # Ensure sim is None initially in case of errors before assignment
            ep_idx = num_scenes_coll
            num_scenes_coll = num_scenes_coll + 1
            try:
                scene_id = episode['scene_id']
                start_position = np.array(episode['start_position'])
                start_rotation = episode['start_rotation']
                object_category = episode.get('object_category', 'unknown')
                info = episode.get('info', {})

                print("start position: ", start_position)
                print("start rotation: ", start_rotation)
                print("object category: ", object_category)

                print(f"\n--- Processing Episode {ep_idx} (Scene: {scene_id}, Target: {object_category}) ---")
                print(f"Start Position: {start_position}, Start Rotation: {start_rotation}")

                ep_dir = os.path.join(output_dir, f"episode_{ep_idx:04d}")
                images_dir = os.path.join(ep_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

                # Initialize simulator
                sim_config = make_sim_config(scene_dataset_config_file, scene_id, width=640, height=480, include_semantic=False)
                print("Environment configured")
                sim = habitat_sim.Simulator(sim_config)
                agent = sim.get_agent(0)
                initial_agent_state = habitat_sim.AgentState()
                initial_agent_state.position = np.array([start_position[0], start_position[1], 0.8])
                initial_agent_state.rotation = start_rotation

                actions = None
                trajectory = None
                while True:
                    print("Resetting agent to start state for trajectory collection attempt.")
                    agent.set_state(initial_agent_state, infer_sensor_states=False)

                    # Start interactive control
                    result, data = interactive_control(sim, agent, category_id_map, images_dir, object_category, include_semantic=False)

                    if result == "restart":
                        print("Restart signal received. Clearing previous attempt data and restarting interaction.")

                        print(f"Deleting images from previous attempt in: {images_dir}")
                        deleted_count = 0
                        try:
                            for filename in os.listdir(images_dir):
                                if filename.startswith("state_") and filename.endswith("_rgb.png"):
                                    file_path = os.path.join(images_dir, filename)
                                    os.remove(file_path)
                                    deleted_count += 1
                            print(f"Deleted {deleted_count} image files.")
                        except OSError as e:
                            print(f"Error deleting image files: {e}")

                        # actions and trajectory will be overwritten in the next loop iteration
                        continue
                    else:
                        # If not "restart", assume successful collection ('q' was pressed)
                        actions = result
                        trajectory = data
                        print("Trajectory collection complete.")
                        break # Exit the while loop for this episode

                if actions is not None and trajectory is not None and len(trajectory) > 0:
                    num_actions_taken = len(actions)
                    final_position = trajectory[-1]['position']

                    metadata = {
                        "scene_id": scene_id,
                        "episode_id": ep_idx,
                        "start_position": start_position.tolist(),
                        "start_rotation": start_rotation.tolist() if isinstance(start_rotation, np.ndarray) else start_rotation,
                        "final_position": final_position,
                        "object_category": object_category,
                        "info": info,
                        "trajectory": trajectory,
                        "actions": actions
                    }
                    metadata_path = os.path.join(ep_dir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"Saved metadata and {num_actions_taken} trajectory steps to {metadata_path}")
                elif actions is not None and trajectory is not None and len(trajectory) == 0:
                     print(f"Episode {ep_idx} finished with 0 actions. No metadata saved.")
                else:
                     print(f"Episode {ep_idx} aborted or failed. No metadata saved.")


            except Exception as e:
                print(f"Error processing episode {ep_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # Ensure simulator is closed properly
                if sim is not None:
                    sim.close()
                    print(f"Simulator closed for episode {ep_idx}.")

if __name__ == "__main__":
    main()
