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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import random
import argparse

DEFAULT_DATASET_DIR = "../../../autonomous_dataset"
DEFAULT_MODEL_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_PROCESSOR_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_SCENE_CONFIG = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALID_ACTIONS = ["move forward", "turn left", "turn right"]

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
    
def load_model_and_processor(model_path, processor_path, lora_path, device):
    """Loads the VLM model and processor, optionally merging LoRA."""
    print(f"Loading base model from: {model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map=device, # Load on CPU if memory constrained during merge
        trust_remote_code=True
    ) #.to(device) # Move later

    model = base_model # Start with base model
    if lora_path:
        print(f"Attempting to load LoRA adapter from: {lora_path}")
        try:
            from peft import PeftModel # Ensure PeftModel is imported
            model_with_adapter = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapter loaded.")
            merged_model = model_with_adapter.merge_and_unload()
            print("LoRA adapter merged.")
            model = merged_model
            # Optionally delete intermediate models if memory is tight
            # del base_model
            # del model_with_adapter
        except ImportError:
             print("Warning: PEFT library not found. Cannot merge LoRA. Install with 'pip install peft'. Using base model.")
        except Exception as e:
            print(f"Error loading/merging LoRA: {e}. Using base model.")

    model = model.to(device).eval() # Move final model to device and set to eval
    print(f"Model ready on {device}.")

    print(f"Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print("Processor loaded.")
    return model, processor

def predict_action(model, processor, image, target_category, device):
    """ Predicts an action using the VLM. Returns (final_action, was_originally_valid) """
    prompt_text = (
         f"<image> You are an agent with an RGB-D camera looking for a {target_category}. Observe the image and think"
         f" step by step about the action you should take to eventually find the {target_category}. Once the {target_category} is found, walk up to the {target_category}, "
         f"but do not interact with it. Look at the position of the {target_category} relative to the center of the image. If the {target_category} is nearly centered "
         f"(e.g., within ±10–20 percent from the middle), then the best action is to move forward. Otherwise, turn in the direction that aligns the {target_category} more centrally."
         f"If the {target_category} is not visible, think about what room you are in based on the objects you see. Reason in short sentences.\n"
         "Example:\n {\n\"Observation\": \"Descibe the entire image\",\n"
         f"\"Reasoning\": \"Think step-by-step about the best action to take to get closer to the {target_category} taking into account your observations. "
         f"If the {target_category} is in sight, move towards it. If the {target_category} is not in sight, think about where it can be relative to your current position "
         "and what action will get you closer to it.\",\n\"Action\": \"Pick an action to take from { move forward, turn left, turn right }\"\n}\nNow, given the current image, "
         "please provide your Observation, detailed Reasoning, and Action from the available actions { move forward, turn left, turn right } to find the "
         f"{target_category}. Please do not repeat this prompt; only generate your response as a valid JSON object. Do now forget the commas in the JSON object!\n"
    )
    message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]

    was_originally_valid = False
    final_action = np.random.choice(VALID_ACTIONS) # Default fallback

    try:
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_input, _ = process_vision_info(message)
        inputs = processor(text=[text], images=image_input, return_tensors="pt", padding=True).to(device)

        gen_config = GenerationConfig(max_new_tokens=256,
                                      pad_token_id=processor.tokenizer.pad_token_id,
                                      eos_token_id=processor.tokenizer.eos_token_id,
                                      repetition_penalty=1.05,   # <-- activates the processor
                                    #   no_repeat_ngram_size=3,
                                    )

        with torch.no_grad():
            generated_ids = model.generate(**inputs, generation_config=gen_config)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        _, sep, vlm_output = output_text.partition("json")
        if sep: vlm_output, _, _ = vlm_output.partition("```")
        else: vlm_output = output_text # Fallback if json marker not found

        # Strip potential leading/trailing whitespace before parsing JSON
        vlm_output = vlm_output.strip()
        try:
            # print(vlm_output)
            answer = json.loads(vlm_output)
            initial_predicted_action = answer.get('Action', '').lower() # Use .get for safety

            if initial_predicted_action in VALID_ACTIONS:
                was_originally_valid = True
                final_action = initial_predicted_action
            else:
                 if initial_predicted_action: # Only warn if it predicted *something* invalid
                     print(f"Warning: Model predicted invalid action '{initial_predicted_action}'. Using random: {final_action}.")
                 else: # Warn if Action key was missing or empty
                    #  print(f"Warning: Could not find valid 'Action' in model JSON output: {vlm_output}. Using random: {final_action}.")
                    print(f"Warning: Could not find valid 'Action' in model JSON output. Using random: {final_action}.")

        except json.JSONDecodeError:
            for action in VALID_ACTIONS:
                 if action in vlm_output:
                    was_originally_valid = True
                    final_action = action

            if not was_originally_valid:
                print(f"Warning: Could not decode JSON from model output: '{vlm_output}'. Using random: {final_action}")


    except Exception as e:
        print(f"Error during model prediction/processing: {e}. Using random action: {final_action}")

    return final_action, was_originally_valid


# Control the agent interactively using keyboard inputs.
# Uses OpenCV to capture key presses:
#     - W: move_forward
#     - A: turn_left
#     - D: turn_right
#     - Q: quit and finish recording
def autonomous_agent(args, sim, agent, id_to_name, images_dir, target_category, include_semantic=False):
    trajectory = []
    actions = []
    step = 0
    # print("Interactive Control Mode:")
    # print("  Press W to move_forward")
    # print("  Press A to turn_left")
    # print("  Press D to turn_right")
    # print("  Press Q to quit")

    model, processor = load_model_and_processor(args.model_path, args.processor_path, args.lora_path, DEVICE)

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

        action, _ = predict_action(model, processor, image, target_category, DEVICE)
        
        sim.step(action)
        actions.append(action)
        agent_state = agent.get_state()
        rgb_state = agent_state.sensor_states['rgb']
        # print("RGB sensor position:", rgb_state.position, "rotation:", rgb_state.rotation)

        if include_semantic:
            semantic_state = agent_state.sensor_states['semantic']
            # print("Semantic sensor position:", semantic_state.position, "rotation:", semantic_state.rotation)
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

def main(args):
    # Configuration paths
    scene_dataset_config_file = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    objectnav_dataset_path = "../../../data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2"
    output_dir = DEFAULT_DATASET_DIR

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
                object_category = episode.get('object_category', 'unknown')

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
                actions, trajectory = autonomous_agent(args, sim, agent, category_id_map, images_dir, object_category, include_semantic=False)

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
    parser = argparse.ArgumentParser(description="Evaluate VLM Navigation Agent (Autonomous & Expert Path)")
    parser.add_argument("--visualize",action="store_true",default=False,help="Enable OpenCV visualization for both runs.")
    parser.add_argument("--dataset-dir",type=str,default=DEFAULT_DATASET_DIR,help="Path to the episode dataset directory.")
    parser.add_argument("--model-path",type=str,default=DEFAULT_MODEL_PATH,help="Path to the BASE VLM model.")
    parser.add_argument("--processor-path",type=str,default=DEFAULT_PROCESSOR_PATH,help="Path to the VLM processor.")
    parser.add_argument("--scene-config",type=str,default=DEFAULT_SCENE_CONFIG,help="Path to Habitat scene config.")
    parser.add_argument("--lora-path",type=str,default=None,help="Path to the trained LoRA adapter directory (optional).")
    args = parser.parse_args()
    main(args)