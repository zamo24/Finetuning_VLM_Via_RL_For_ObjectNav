import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs
import json
import os
import glob
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info
import cv2
import argparse
from peft import PeftModel
import gc

DEFAULT_DATASET_DIR = "../../../output_dataset"
DEFAULT_OUTPUT_DATASET_DIR = "../../../autonomous_dataset"
DEFAULT_MODEL_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_PROCESSOR_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_SCENE_CONFIG = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation Parameters
MAX_STEPS_AUTONOMOUS = 200 # Max steps for the autonomous run
SUCCESS_DISTANCE = 1.0
IMG_WIDTH = 640
IMG_HEIGHT = 480
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

def save_rgb(image, path):
    img = Image.fromarray(image)
    img.save(path)

def make_sim_config(scene_dataset_config_file, scene_id, width, height):
    """Initializes Habitat simulator configuration."""
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

    agent_cfg.action_space = {
        ACTION_MAP["move forward"]: habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.4)),
        ACTION_MAP["turn left"]: habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        ACTION_MAP["turn right"]: habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
    }
    # Ensure the keys we intend to use from the dataset map to simulator actions
    assert all(action in agent_cfg.action_space for action in ACTION_MAP.values())
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def load_model_and_processor(model_path, processor_path, lora_path, device):
    print(f"Loading base model from: {model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # device_map=device, # Load on CPU if memory constrained during merge
        trust_remote_code=True
    )

    model = base_model # Start with base model
    if lora_path:
        print(f"Attempting to load LoRA adapter from: {lora_path}")
        try:
            model_with_adapter = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapter loaded.")
            merged_model = model_with_adapter.merge_and_unload()
            print("LoRA adapter merged.")
            model = merged_model
        except Exception as e:
            print(f"Error loading/merging LoRA: {e}. Using base model.")

    model = model.to(device).eval() # Move final model to device and set to eval
    print(f"Model ready on {device}.")

    print(f"Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    print("Processor loaded.")
    return model, processor

def predict_action(model, processor, image, target_category, device):
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
        else: vlm_output = output_text # Fallback if json marker not found

        # Strip potential leading/trailing whitespace before parsing JSON
        vlm_output = vlm_output.strip()
        try:
            answer = json.loads(vlm_output)
            initial_predicted_action = answer.get('Action', '').lower()

            if initial_predicted_action in VALID_ACTIONS:
                was_originally_valid = True
                final_action = initial_predicted_action
            else:
                 if initial_predicted_action: # Only warn if it predicted something invalid
                     print(f"Warning: Model predicted invalid action '{initial_predicted_action}'. Using random: {final_action}.")
                 else: # Warn if Action key was missing or empty
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

def evaluate_model(args):
    print("Starting evaluation...")
    model, processor = load_model_and_processor(args.model_path, args.processor_path, args.lora_path, DEVICE)

    metadata_files = glob.glob(os.path.join(args.dataset_dir, "episode_*", "metadata.json"))
    if not metadata_files:
        print(f"Error: No metadata files found in {args.dataset_dir}. Check path.")
        return

    print(f"Found {len(metadata_files)} episodes to evaluate.")
    if args.visualize:
        print("Visualization enabled.")

    # Autonomous Run Stats
    total_autonomous_successes = 0
    total_autonomous_steps = 0
    total_autonomous_predictions = 0
    total_autonomous_valid_actions = 0
    # Expert Path Stats
    total_expert_path_steps = 0
    total_expert_path_aligned_actions = 0

    failure_by_getting_stuck = 0
    failure_by_oscillation = 0

    all_episode_results = [] # Store detailed results per episode

    for episode_idx, meta_path in enumerate(metadata_files):
        trajectory = []

        if (episode_idx % 10) != 0:
            continue

        episode_file_name = os.path.basename(os.path.dirname(meta_path))
        print(f"\n--- Evaluating Episode {episode_idx + 1}/{len(metadata_files)} ({episode_file_name}) ---")
        sim = None
        episode_results = {"episode_path": meta_path}

        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            scene_id = metadata['scene_id']
            start_position = np.array(metadata['start_position'])
            start_rotation_coeffs = metadata['start_rotation']
            target_final_pos = np.array(metadata['final_position'])
            object_category = metadata.get('object_category', 'unknown')
            expert_actions_raw = metadata['actions']
            num_expert_path_steps = len(expert_actions_raw)

            if args.flag:
                ep_dir = os.path.join(DEFAULT_OUTPUT_DATASET_DIR, f"episode_{episode_idx:04d}")
                images_dir = os.path.join(ep_dir, "images")
                os.makedirs(images_dir, exist_ok=True)

            expert_actions_mapped = []
            valid_episode = True
            for act in expert_actions_raw:
                act_lower = act.lower()
                if act_lower in ACTION_MAP:
                    expert_actions_mapped.append(ACTION_MAP[act_lower])
                else:
                    print(f"Warning: Episode {episode_file_name} contains unknown expert action '{act}'. Skipping episode.")
                    valid_episode = False
                    break
            if not valid_episode or not expert_actions_mapped:
                print(f"Skipping episode {episode_file_name} due to invalid/empty expert actions.")
                continue

            # Initialize sim
            sim_config = make_sim_config(args.scene_config, scene_id, IMG_WIDTH, IMG_HEIGHT)
            sim = habitat_sim.Simulator(sim_config)
            agent = sim.get_agent(0)
            initial_agent_state = habitat_sim.AgentState()
            initial_agent_state.position = np.array([start_position[0], start_position[1], 0.8])
            initial_agent_state.rotation = quat_from_coeffs(start_rotation_coeffs)
            agent.set_state(initial_agent_state, infer_sensor_states=False)

            # Run 1: Expert Path Action Accuracy
            print("--- Running Evaluation: Expert Path Action Accuracy ---")
            expert_path_current_aligned = 0
            autonomous_current_valid = 0
            current_steps = 0
            for step_idx, expert_action in enumerate(expert_actions_mapped):
                obs = sim.get_sensor_observations()
                rgb_obs = obs['rgb']
                current_image = Image.fromarray(rgb_obs)

                # Predict action based on current state
                predicted_action, valid = predict_action(model, processor, current_image, object_category, DEVICE)
                total_expert_path_steps += 1
                current_steps += 1

                if valid:
                    autonomous_current_valid += 1

                # Check alignment
                if valid and predicted_action == expert_action:
                    expert_path_current_aligned += 1
                    alignment_status = "(Aligned!)"
                else:
                    alignment_status = "(Misaligned)"

                print(f"Expert Step {step_idx}: Pred: {predicted_action} | Expert: {expert_action} {alignment_status}")

                # Visualize if enabled
                if args.visualize:
                    display_image = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
                    font, scale, color, thick, ltype = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    cv2.putText(display_image, f"Expert Path Step: {step_idx}", (10, 20), font, scale, color, thick, ltype)
                    cv2.putText(display_image, f"Target: {object_category}", (10, 40), font, scale, color, thick, ltype)
                    align_color = (0, 255, 0) if predicted_action == expert_action else (0, 0, 255)
                    cv2.putText(display_image, f"Pred: {predicted_action}", (10, 60), font, scale, align_color, thick, ltype)
                    cv2.putText(display_image, f"Expert: {expert_action}", (10, 80), font, scale, color, thick, ltype)
                    cv2.imshow("Evaluation View - Expert Path", display_image)
                    key = cv2.waitKey(1)
                    if key == ord('q'): args.visualize = False; cv2.destroyAllWindows()

                # Execute the expert action
                if expert_action in agent.agent_config.action_space:
                     sim.step(expert_action)
                else:
                     print(f"Error: Expert action '{expert_action}' not in simulator action space! Stopping expert path run.")
                     break

            expert_path_accuracy = (expert_path_current_aligned / len(expert_actions_mapped)) * 100 if expert_actions_mapped else 0
            episode_results["expert_path_action_accuracy"] = expert_path_accuracy
            episode_results["expert_path_steps"] = len(expert_actions_mapped)
            episode_results["expert_path_aligned_count"] = expert_path_current_aligned
            total_expert_path_aligned_actions += expert_path_current_aligned
            valid_steps_percentage = (autonomous_current_valid / current_steps) * 100
            print(f"Expert Path Accuracy for Episode: {expert_path_accuracy:.2f}%, Valid Actions %: {valid_steps_percentage:.2f}%")

            if args.visualize:
                cv2.destroyAllWindows()

            # Run 2: Autonomous Navigation
            print("--- Running Evaluation: Autonomous Navigation ---")
            # Reset agent to start state before autonomous run
            agent.set_state(initial_agent_state, infer_sensor_states=False)
            print("Agent reset to start state for autonomous run.")

            autonomous_steps_taken = 0
            autonomous_current_valid = 0
            autonomous_is_success = False
            agent_positions_history = [initial_agent_state.position.tolist()]
            distance_to_target = 0
            num_dist_no_change = 0
            predicted_action = None
            num_sequential_oscillations = 0

            max_autonomous_steps = max(MAX_STEPS_AUTONOMOUS, 2*num_expert_path_steps)
            while autonomous_steps_taken < max_autonomous_steps:
                obs = sim.get_sensor_observations()
                rgb_obs = obs['rgb']
                current_image = Image.fromarray(rgb_obs)

                # Predict action and check validity
                prev_predicted_action = predicted_action
                predicted_action, was_valid = predict_action(model, processor, current_image, object_category, DEVICE)
                total_autonomous_predictions += 1
                if was_valid:
                    autonomous_current_valid += 1

                # Detect oscillations
                if predicted_action == 'turn left' and prev_predicted_action == 'turn right':
                    num_sequential_oscillations += 1
                elif predicted_action == 'turn right' and prev_predicted_action == 'turn left':
                    num_sequential_oscillations += 1
                else:
                    num_sequential_oscillations = 0

                # Execute predicted action          
                if predicted_action in agent.agent_config.action_space:
                    sim.step(predicted_action)
                    agent_state = agent.get_state()
                    agent_positions_history.append(agent_state.position.tolist())
                else:
                    print(f"Error: Corrected action '{predicted_action}' not in action space? Skipping sim step.")

                current_pos = agent_state.position
                prev_dist_to_target = distance_to_target
                distance_to_target = np.linalg.norm(current_pos - target_final_pos)

                # Check if agent is stuck
                if distance_to_target == prev_dist_to_target and predicted_action == 'move forward':
                    num_dist_no_change += 1
                else:
                    num_dist_no_change = 0

                print(f"Autonomous Step {autonomous_steps_taken}: Pred: {predicted_action} (Valid: {was_valid}) | Dist: {distance_to_target:.2f}m")

                # Visualize if enabled
                if args.visualize:
                    display_image = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
                    font, scale, color, thick, ltype = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    valid_color = (0, 255, 0) if was_valid else (0, 0, 255)
                    cv2.putText(display_image, f"Autonomous Step: {autonomous_steps_taken}/{MAX_STEPS_AUTONOMOUS}", (10, 20), font, scale, color, thick, ltype)
                    cv2.putText(display_image, f"Target: {object_category}", (10, 40), font, scale, color, thick, ltype)
                    cv2.putText(display_image, f"Pred Action: {predicted_action}", (10, 60), font, scale, valid_color, thick, ltype)
                    cv2.putText(display_image, f"Dist to Target: {distance_to_target:.2f}m", (10, 80), font, scale, color, thick, ltype)
                    if autonomous_is_success: cv2.putText(display_image, "SUCCESS!", (IMG_WIDTH // 2 - 50, IMG_HEIGHT // 2), font, 1.0, (0, 255, 255), 2, ltype)
                    cv2.imshow("Evaluation View - Autonomous", display_image)
                    key = cv2.waitKey(1)
                    if key == ord('q'): args.visualize = False; cv2.destroyAllWindows()

                # Check success
                if distance_to_target <= SUCCESS_DISTANCE:
                    autonomous_is_success = True
                    print(f"Success! Reached target within {SUCCESS_DISTANCE}m.")
                    if args.flag:
                        # Save the current RGB image.
                        rgb_path = os.path.join(images_dir, f"state_{autonomous_steps_taken:04d}_rgb.png")
                        save_rgb(sim.get_sensor_observations()['rgb'], rgb_path)
                    if args.visualize: # Show final success frame
                         display_image = cv2.cvtColor(sim.get_sensor_observations()['rgb'], cv2.COLOR_RGB2BGR)
                         cv2.putText(display_image, f"Autonomous Step: {autonomous_steps_taken+1}", (10, 20), font, scale, color, thick, ltype)
                         cv2.putText(display_image, "SUCCESS!", (IMG_WIDTH // 2 - 60, IMG_HEIGHT // 2), font, 1.2, (0, 255, 255), 2, ltype)
                         cv2.imshow("Evaluation View - Autonomous", display_image)
                         cv2.waitKey(500)
                    break

                if args.flag:
                    # Save the current RGB image.
                    rgb_path = os.path.join(images_dir, f"state_{autonomous_steps_taken:04d}_rgb.png")
                    save_rgb(rgb_obs, rgb_path)

                autonomous_steps_taken += 1

                # Agent is stuck
                if num_dist_no_change == 10:
                    failure_by_getting_stuck += 1
                    print("FAILURE: The agent is stuck")
                    break
                elif num_sequential_oscillations == 5:
                    failure_by_oscillation += 1
                    print("FAILURE: The agent is oscillating")
                    break
            # End autonomous while loop

            episode_results["autonomous_success"] = autonomous_is_success
            episode_results["autonomous_steps_taken"] = autonomous_steps_taken
            episode_results["autonomous_valid_action_count"] = autonomous_current_valid
            episode_results["autonomous_predictions_count"] = autonomous_steps_taken
            total_autonomous_successes += int(autonomous_is_success)
            total_autonomous_steps += autonomous_steps_taken
            total_autonomous_valid_actions += autonomous_current_valid
            autonomous_steps_valid = (autonomous_current_valid / total_autonomous_steps) * 100
            print(f"Autonomous Run Ended. Success: {autonomous_is_success}, Steps: {autonomous_steps_taken}, Valid Action %: {autonomous_steps_valid:.2f}%")

            # Append results for this episode
            all_episode_results.append(episode_results)

        except Exception as e:
            print(f"Error during evaluation of episode {meta_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if sim is not None:
                sim.close()
                del sim
                gc.collect()
                torch.cuda.empty_cache()
    # End of episode loop

    if args.visualize:
        cv2.destroyAllWindows()

    # Calculate final metrics
    num_total_trials = len(all_episode_results)
    if num_total_trials == 0:
        print("No trials were successfully processed.")
        return

    # Autonomous metrics
    overall_autonomous_success_rate = (total_autonomous_successes / num_total_trials) * 100
    average_autonomous_steps = total_autonomous_steps / num_total_trials
    # Use total_autonomous_predictions which counts attempts even if trial ended early
    overall_valid_action_percentage = (total_autonomous_valid_actions / total_autonomous_predictions) * 100 if total_autonomous_predictions > 0 else 0

    # Expert path accuracy
    average_expert_path_accuracy = (total_expert_path_aligned_actions / total_expert_path_steps) * 100 if total_expert_path_steps > 0 else 0


    print("\n--- Evaluation Summary ---")
    print(f"Model evaluated: {args.model_path}")
    if args.lora_path: print(f"LoRA adapter merged from: {args.lora_path}")
    print(f"Total Trials Evaluated: {num_total_trials}")
    print("-" * 25)
    print("Autonomous Navigation Metrics:")
    print(f"  - Success Rate: {overall_autonomous_success_rate:.2f}% ({total_autonomous_successes} / {num_total_trials})")
    print(f"  - Average Steps Taken: {average_autonomous_steps:.2f} (Max Steps: {MAX_STEPS_AUTONOMOUS})")
    print(f"  - Valid Action Prediction Rate: {overall_valid_action_percentage:.2f}% ({total_autonomous_valid_actions} / {total_autonomous_predictions})")
    print("-" * 25)
    print("Expert Path Metrics:")
    print(f"  - Average Action Accuracy on Path: {average_expert_path_accuracy:.2f}% ({total_expert_path_aligned_actions} / {total_expert_path_steps})")
    print("-" * 25)

    # --- Save detailed results ---
    results_filename = f"evaluation_results_{os.path.basename(args.model_path)}"
    if args.lora_path: results_filename += f"_lora_{os.path.basename(args.lora_path)}"
    results_filename += ".json"

    try:
        summary_data = {
            "model_path": args.model_path, "lora_path": args.lora_path,
            "total_trials": num_total_trials,
            "autonomous_success_rate_percent": overall_autonomous_success_rate,
            "autonomous_successful_trials": total_autonomous_successes,
            "autonomous_average_steps": average_autonomous_steps,
            "autonomous_max_steps": MAX_STEPS_AUTONOMOUS,
            "autonomous_total_predictions": total_autonomous_predictions,
            "autonomous_valid_actions": total_autonomous_valid_actions,
            "autonomous_valid_action_percent": overall_valid_action_percentage,
            "expert_path_avg_action_accuracy_percent": average_expert_path_accuracy,
            "expert_path_total_steps": total_expert_path_steps,
            "expert_path_total_aligned": total_expert_path_aligned_actions,
            "failure_by_getting_stuck": failure_by_getting_stuck,
            "failure_by_oscillation": failure_by_oscillation,
            "failure_by_inefficient_exploration": num_total_trials - failure_by_getting_stuck - failure_by_oscillation,
        }
        with open(results_filename, "w") as f:
            json.dump({"summary": summary_data, "detailed_results": all_episode_results}, f, indent=2)
        print(f"Detailed results saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to {results_filename}: {e}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no',  'false','f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM Navigation Agent (Autonomous & Expert Path)")
    parser.add_argument("--visualize",action="store_true",default=False,help="Enable OpenCV visualization for both runs.")
    parser.add_argument("--dataset-dir",type=str,default=DEFAULT_DATASET_DIR,help="Path to the episode dataset directory.")
    parser.add_argument("--model-path",type=str,default=DEFAULT_MODEL_PATH,help="Path to the BASE VLM model.")
    parser.add_argument("--processor-path",type=str,default=DEFAULT_PROCESSOR_PATH,help="Path to the VLM processor.")
    parser.add_argument("--scene-config",type=str,default=DEFAULT_SCENE_CONFIG,help="Path to Habitat scene config.")
    parser.add_argument("--lora-path",type=str,default=None,help="Path to the trained LoRA adapter directory (optional).")
    parser.add_argument(
        '--flag',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Enable or disable the flag (use --flag or --flag true/false)'
    )
    args = parser.parse_args()
    evaluate_model(args)