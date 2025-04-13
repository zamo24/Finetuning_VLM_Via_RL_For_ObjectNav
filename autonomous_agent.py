import habitat_sim
# from habitat_sim.viewer import Viewer
import json
import os
import gzip
import numpy as np
from PIL import Image
import math
import cv2
import threading
import magnum as mn
import segmentation_models_pytorch as smp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import random

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
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = np.array(mn.Vector3([0, 0.8, 0])) # 0.8m mimicks the height of a unitree go2
    rgb_sensor.hfov = 90

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
        "move forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.2)),
        "turn left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        "turn right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
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
    
def load_model_and_processor():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "remyxai/SpaceQwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.float16,
        # quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")

    return model, processor

def predict_action(model, processor, image):
    prompt_text = (
        "<image> You are an agent with an RGB-D camera looking for a television. Observe the image and think"
        " step by step about the action you should take to eventually find the television. Once the television is found, walk up to the television, "
        "but do not interact with it. Look at the position of the TV relative to the center of the image. If the TV is nearly centered "
        "(e.g., within ±10–20 percent from the middle), then the best action is to move forward. Otherwise, turn in the direction that aligns the TV more centrally. "
        "If the television is not visible, think about the best action from the available actions to take next. Reason in short sentences.\n"
        "Example:\nObservation: Descibe the entire image\n"
        "Reasoning: Think step-by-step about the best action to take to get closer to the television taking into account your observations. "
        "If the television is in sight, move towards it. If the television is not in sight, think about where it can be relative to your current position "
        "and what action will get you closer to it.\nAction: Pick an action to take from { move forward, turn left, turn right }\nNow, given the current image, "
        "please provide your Observation, detailed Reasoning, and Action from the available actions { move forward, turn left, turn right } to find the television. "
        "Please do not repeat this prompt; only generate your response as a JSON object. \n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # print(messages)

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    #     do_sample=False,
    #     temperature=0.90,
    #     num_beams=1,
    #     # min_length=128,
    #     top_p=0.80,
    #     repetition_penalty=10.0,
    #     length_penalty=1.0,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text[0])

    _, sep, vlm_output = output_text[0].partition("json")
    if sep:
        vlm_output, sep, _ = vlm_output.partition("```")
    else:
        vlm_output = output_text[0]

    # print("here")
    # if sep:
    #     print("VLM output: ", vlm_output)

    try:
        answer = json.loads(vlm_output)
        action = answer['Action']
    except json.JSONDecodeError as e:
        # Use a random action
        rand_int = random.randint(0, 2)

        if rand_int == 0:
            action = "move forward"
        elif rand_int == 1:
            action = "turn right"
        else:
            action = "turn left"

    print('Action: ', action)

    return action


# Control the agent interactively using keyboard inputs.
# Uses OpenCV to capture key presses:
#     - W: move_forward
#     - A: turn_left
#     - D: turn_right
#     - Q: quit and finish recording
def autonomous_agent(sim, agent, id_to_name, images_dir, include_semantic=True):
    trajectory = []
    actions = []
    step = 0
    # print("Interactive Control Mode:")
    # print("  Press W to move_forward")
    # print("  Press A to turn_left")
    # print("  Press D to turn_right")
    # print("  Press Q to quit")

    model, processor = load_model_and_processor()

    # seg_model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=21, activation=None)
    
    while True:
        obs = sim.get_sensor_observations()
        rgb = obs['rgb']
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Agent View", rgb_bgr)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            print("Exiting interactive control.")
            break
        # elif key == ord('w'):
        #     action = "move_forward"
        # elif key == ord('a'):
        #     action = "turn_left"
        # elif key == ord('d'):
        #     action = "turn_right"
        # else:
        #     print("Invalid key. Use W, A, D, or Q.")
        #     continue

        image = Image.fromarray(rgb)

        action = predict_action(model, processor, image)
        
        sim.step(action)
        actions.append(action)
        agent_state = agent.get_state()
        rgb_state = agent_state.sensor_states['rgb']
        print("RGB sensor position:", rgb_state.position, "rotation:", rgb_state.rotation)

        if include_semantic:
            semantic_state = agent_state.sensor_states['semantic']
            print("Semantic sensor position:", semantic_state.position, "rotation:", semantic_state.rotation)
            semantic_info = obs['semantic']
            visible_objects = get_object_bboxes(semantic_info, id_to_name)

        trajectory.append({
            "step": step,
            "position": agent_state.position.tolist(),
            "orientation": [agent_state.rotation.x,
                            agent_state.rotation.y,
                            agent_state.rotation.z,
                            agent_state.rotation.w],
            "action": action,
            # "depth": obs['depth'],
            # 'semantic': semantic_info,
            # 'visible_objects': visible_objects,
        })
        # Save the current RGB image.
        rgb_path = os.path.join(images_dir, f"state_{step:04d}_rgb.png")
        save_rgb(rgb, rgb_path)
        step += 1
        
    cv2.destroyAllWindows()
    return actions, trajectory

# Launch the Habitat-Sim GUI viewer in a separate thread.
# def run_viewer(sim):
#     viewer = habitat_sim.viewer.Viewer(sim, "Habitat-Sim GUI")
#     viewer.run()

def get_object_bboxes(semantic, id_to_name):
    visible_objects = []
    categories = id_to_name['category_to_task_category_id']
    objects = list(categories.keys())
    object_ids = list(categories.values())
    # print(objects)
    # print(object_ids)
    # print(semantic)
    for obj_id in object_ids:
        # if obj_id == 0:  # 0 might be background in some datasets
        #     continue
        mask = (semantic == obj_id)
        # print(mask)
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
    output_dir = "../../../autonomous_dataset"

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

    # For demonstration, process the first episode.
    for i in range(len(scene_episodes)):
        for ep_idx, episode in enumerate(scene_episodes[i]['episodes']):
            try:
                # print("episode dir: ", dir(episode))
                keys = episode.keys()
                print("Episode keys: ", keys)
                scene_id = episode['scene_id']
                start_position = episode['start_position']
                print("start position: ", start_position)
                start_rotation = episode['start_rotation']
                print("Selected episode with scene:", scene_id)

                # Create output directory for this episode.
                ep_dir = os.path.join(output_dir, f"episode_{ep_idx:04d}")
                images_dir = os.path.join(ep_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

                # Initialize simulator with semantic sensor enabled (if desired).
                sim_config = make_sim_config(scene_dataset_config_file, scene_id, width=640, height=480, include_semantic=False)
                print("Environment configured for scene", scene_id)
                sim = habitat_sim.Simulator(sim_config)
                print("Simulator started! The Habitat-Sim GUI should appear shortly.")
                
                # Launch the Habitat-Sim GUI viewer in a separate thread.
                # viewer_thread = threading.Thread(target=run_viewer, args=(sim,))
                # viewer_thread.daemon = True
                # viewer_thread.start()
                
                agent = sim.get_agent(0)
                agent_state = habitat_sim.AgentState()
                agent_state.position = np.array([start_position[0], start_position[1], 0.8])
                agent_state.rotation = start_rotation
                agent.set_state(agent_state)

                print("Entering interactive control mode. Use W, A, D to control the agent; Q to quit.")
                actions, trajectory = autonomous_agent(sim, agent, category_id_map, images_dir, include_semantic=False)

                metadata = {
                    "scene_id": scene_id,
                    "episode_id": ep_idx,
                    "start_position": start_position,
                    "trajectory": trajectory,
                    "actions": actions
                }
                with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                sim.close()
                print(f"Processed episode {ep_idx} successfully with {len(actions)} actions.")

            except Exception as e:
                print(f"Error processing episode {ep_idx}: {str(e)}")
                if 'sim' in locals():
                    sim.close()
                continue

if __name__ == "__main__":
    main()