import os, json, glob, gc, argparse, uuid
from pathlib import Path
from collections import deque
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# === New constants for failure mining ===
FAILURE_DATASET_DIR = Path("../../../failure_dataset")
FAILURE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

OSC_PATTERN = [("turn left", "turn right", "turn left"), ("turn right", "turn left", "turn right")]
OSC_PENALTY = -0.2
COLL_PENALTY = -0.5
CONTEXT_BEFORE = 1  # frames kept before oscillation start
CONTEXT_AFTER = 1   # frames kept after oscillation end
DEFAULT_DATASET_DIR = "../../../output_dataset"
DEFAULT_MODEL_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_PROCESSOR_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_SCENE_CONFIG = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_WIDTH, IMG_HEIGHT = 640, 480
MAX_STEPS_AUTONOMOUS = 50

VALID_ACTIONS = ["move forward", "turn left", "turn right"]
# Map potential variations in dataset action names to simulator action keys if needed
ACTION_MAP = {
    "move_forward": "move forward",
    "move forward": "move forward",
    "turn_left": "turn left",
    "turn left": "turn left",
    "turn_right": "turn right",
    "turn right": "turn right"
}

def make_sim_config(scene_dataset_config_file, scene_id, width, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config_file
    sim_cfg.scene_id = scene_id
    sim_cfg.load_semantic_mesh = False

    agent_cfg = habitat_sim.AgentConfiguration()
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = np.array([0, 0.8, 0])
    rgb_sensor.hfov = 90
    agent_cfg.sensor_specifications = [rgb_sensor]

    # Use the mapped action keys for the simulator's action space setup
    agent_cfg.action_space = {
        ACTION_MAP["move forward"]: habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.4)),
        ACTION_MAP["turn left"]: habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        ACTION_MAP["turn right"]: habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
    }
    # Ensure the keys we intend to use from the dataset map to simulator actions
    assert all(action in agent_cfg.action_space for action in ACTION_MAP.values())
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def predict_action(model, processor, image, target_category, device):
    prompt_text = (
         f"<image> You are an agent with an RGB-D camera looking for a {target_category}. Observe the image and think"
         f" step by step about the action you should take to eventually find the {target_category}. Once the {target_category} is found, walk up to the {target_category}, "
         f"but do not interact with it. Look at the position of the {target_category} relative to the center of the image. If the {target_category} is nearly centered "
         f"(e.g., within ±10–20 percent from the middle), then the best action is to move forward. Otherwise, turn in the direction that aligns the {target_category} more centrally."
         f"If the {target_category} is not visible, think about what room you are in based on the objects you see. Reason in short sentences.\n"
         f"Example:\nObservation: Descibe the entire image\n"
         f"Reasoning: Think step-by-step about the best action to take to get closer to the {target_category} taking into account your observations. "
         f"If the {target_category} is in sight, move towards it. If the {target_category} is not in sight, think about where it can be relative to your current position "
         "and what action will get you closer to it.\nAction: Pick an action to take from { move forward, turn left, turn right }\nNow, given the current image, "
         "please provide your Observation, detailed Reasoning, and Action from the available actions { move forward, turn left, turn right } to find the "
         f"{target_category}. Please do not repeat this prompt; only generate your response as a JSON object. \n"
    )
    message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]

    was_originally_valid = False
    final_action = np.random.choice(VALID_ACTIONS)

    try:
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_input, _ = process_vision_info(message)
        inputs = processor(text=[text], images=image_input, return_tensors="pt", padding=True).to(device)

        gen_config = GenerationConfig(max_new_tokens=256,
                                      pad_token_id=processor.tokenizer.pad_token_id,
                                      eos_token_id=processor.tokenizer.eos_token_id,
                                      repetition_penalty=1.05,
                                    #   no_repeat_ngram_size=3,
                                    )

        with torch.no_grad():
            generated_ids = model.generate(**inputs, generation_config=gen_config)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        _, sep, vlm_output = output_text.partition("json")
        if sep: vlm_output, _, _ = vlm_output.partition("```")
        else: vlm_output = output_text

        # Strip potential leading/trailing whitespace before parsing JSON
        vlm_output = vlm_output.strip()
        try:
            # print(vlm_output)
            answer = json.loads(vlm_output)
            initial_predicted_action = answer.get('Action', '').lower()

            if initial_predicted_action in VALID_ACTIONS:
                was_originally_valid = True
                final_action = initial_predicted_action
            else:
                 if initial_predicted_action:
                     print(f"Warning: Model predicted invalid action '{initial_predicted_action}'. Using random: {final_action}.")
                 else:
                    print(f"Warning: Could not find valid 'Action' in model JSON output. Using random: {final_action}.")

        except json.JSONDecodeError:
             print(f"Warning: Could not decode JSON from model output: '{vlm_output}'. Using random: {final_action}")


    except Exception as e:
        print(f"Error during model prediction/processing: {e}. Using random action: {final_action}")

    return final_action, was_originally_valid

def load_model_and_processor(model_path, processor_path, lora_path, device):
    print(f"Loading base model from: {model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map=device, # Load on CPU if memory constrained during merge
        trust_remote_code=True
    )

    model = base_model
    if lora_path:
        print(f"Attempting to load LoRA adapter from: {lora_path}")
        try:
            model_with_adapter = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapter loaded.")
            merged_model = model_with_adapter.merge_and_unload()
            print("LoRA adapter merged.")
            model = merged_model
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

def _write_failure_episode(img_buffers, traj, meta_base, failure_type):
    ep_id = f"{failure_type}_{uuid.uuid4().hex[:8]}"
    ep_dir = FAILURE_DATASET_DIR / f"episode_{ep_id}"
    img_dir = ep_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, rgb in enumerate(img_buffers):
        Image.fromarray(rgb).save(img_dir / f"state_{i:04d}_rgb.png")

    # Build metadata
    meta = {
        "scene_id": meta_base["scene_id"],
        "episode_id": ep_id,
        "failure_type": failure_type,
        "start_position": meta_base["start_position"],
        "start_rotation": meta_base["start_rotation"],
        "object_category": meta_base.get("object_category", "unknown"),
        "trajectory": traj,
        "actions": [step["action"] for step in traj],
    }
    with open(ep_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[FailureMiner] stored {failure_type} episode → {ep_dir}")

def detect_and_store_failure(window_actions, window_rgbs, window_states, meta_base):
    if len(window_actions) < 3:
        return

    last3 = tuple(window_actions)[-3:]
    if last3 in OSC_PATTERN:
        # Build minimal trajectory objects
        traj = []
        for idx, (rgb, state, act) in enumerate(zip(window_rgbs, window_states, window_actions)):
            reward = OSC_PENALTY if idx >= len(window_actions) - 3 else 0.0
            traj.append({
                "step": idx,
                "position": state.position.tolist(),
                "orientation": [state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w],
                "action": act,
                "reward": reward,
            })
        _write_failure_episode(window_rgbs, traj, meta_base, "oscillation")
        return True
    return False

def evaluate_and_mine(args):
    model, processor = load_model_and_processor(args.model_path, args.processor_path, args.lora_path, DEVICE)

    meta_paths = glob.glob(os.path.join(args.dataset_dir, "episode_*", "metadata.json"))
    if not meta_paths:
        print("No metadata found."); return

    for meta_path in meta_paths:
        with open(meta_path) as f:
            meta = json.load(f)
        scene_id = meta["scene_id"]
        target_category = meta.get("object_category", "unknown")
        start_position = np.array(meta["start_position"])
        start_rotation = quat_from_coeffs(meta["start_rotation"])

        # Configure sim
        sim_cfg = make_sim_config(args.scene_config, scene_id, IMG_WIDTH, IMG_HEIGHT)
        sim = habitat_sim.Simulator(sim_cfg)
        agent = sim.get_agent(0)
        st_state = habitat_sim.AgentState()
        st_state.position = np.array([start_position[0], start_position[1], 0.8])
        st_state.rotation = start_rotation
        agent.set_state(st_state, infer_sensor_states=False)

        window_actions, window_rgbs, window_states = deque(maxlen=CONTEXT_BEFORE+3+CONTEXT_AFTER), deque(maxlen=CONTEXT_BEFORE+3+CONTEXT_AFTER), deque(maxlen=CONTEXT_BEFORE+3+CONTEXT_AFTER)

        for step in range(MAX_STEPS_AUTONOMOUS):
            obs = sim.get_sensor_observations(); rgb = obs["rgb"]
            img_pil = Image.fromarray(rgb)
            action, _ = predict_action(model, processor, img_pil, target_category, DEVICE)

            # Update window
            window_actions.append(action)
            window_rgbs.append(rgb)
            window_states.append(agent.get_state())

            # Collision check
            if step >= 1 and np.linalg.norm(window_states[-1].position - window_states[-2].position) < 1e-3:
                # Treat as collision; add penalty and store failure once
                traj = [{
                    "step": 0,
                    "position": window_states[-2].position.tolist(),
                    "orientation": [window_states[-2].rotation.x, window_states[-2].rotation.y, window_states[-2].rotation.z, window_states[-2].rotation.w],
                    "action": window_actions[-2],
                    "reward": COLL_PENALTY,
                },{
                    "step":1,
                    "position": window_states[-1].position.tolist(),
                    "orientation": [window_states[-1].rotation.x, window_states[-1].rotation.y, window_states[-1].rotation.z, window_states[-1].rotation.w],
                    "action": window_actions[-1],
                    "reward": COLL_PENALTY,
                }]
                _write_failure_episode(list(window_rgbs)[-2:], traj, meta, "collision")

            # Oscillation detection every step once context filled
            if detect_and_store_failure(window_actions, window_rgbs, window_states, meta):
                window_actions.clear(); window_rgbs.clear(); window_states.clear()

        sim.close()
        del sim
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate and mine oscillation failures")
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--scene-config", default=DEFAULT_SCENE_CONFIG)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--processor-path", default=DEFAULT_PROCESSOR_PATH)
    p.add_argument("--lora-path", default=None)
    args = p.parse_args()
    evaluate_and_mine(args)
