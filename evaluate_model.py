import habitat_sim
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

DEFAULT_DATASET_DIR = "../../../output_dataset"
DEFAULT_MODEL_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_PROCESSOR_PATH = "remyxai/SpaceQwen2.5-VL-3B-Instruct"
DEFAULT_SCENE_CONFIG = "./hm3d_v0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation Parameters
MAX_STEPS = 500
SUCCESS_DISTANCE = 1.0
IMG_WIDTH = 640
IMG_HEIGHT = 480
VALID_ACTIONS = ["move_forward", "turn_left", "turn_right"]
ACTION_MAP = {"move forward": "move_forward", "turn right": "turn_right", "turn left": "turn_left"}

# Initializes Habitat simulator configuration
def make_sim_config(scene_dataset_config_file, scene_id, width, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config_file
    sim_cfg.scene_id = scene_id
    sim_cfg.load_semantic_mesh = False

    agent_cfg = habitat_sim.AgentConfiguration()

    # Define Sensors
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = np.array([0, 0.8, 0])
    rgb_sensor.hfov = 90

    agent_cfg.sensor_specifications = [rgb_sensor]

    # Define Action Space)
    agent_cfg.action_space = {
        "move forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.4)), # Use same amounts as collection
        "turn left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        "turn right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def load_model_and_processor(model_path, processor_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    return model, processor


# Predicts an action using the VLM based on the current image and target
def predict_action(model, processor, image, target_category, device):
    prompt_text = (
        f"<image> You are an agent with an RGB-D camera looking for a {target_category}. Observe the image and think"
        f" step by step about the action you should take to eventually find the {target_category}. Once the {target_category} is found, walk up to the {target_category}, "
        f"but do not interact with it. Look at the position of the {target_category} relative to the center of the image. If the {target_category} is nearly centered "
        f"(e.g., within ±10–20 percent from the middle), then the best action is to move forward. Otherwise, turn in the direction that aligns the {target_category} more centrally. "
        f"If the {target_category} is not visible, think about the best action from the available actions to take next. Reason in short sentences.\n"
        f"Example:\nObservation: Descibe the entire image\n"
        f"Reasoning: Think step-by-step about the best action to take to get closer to the {target_category} taking into account your observations. "
        f"If the {target_category} is in sight, move towards it. If the {target_category} is not in sight, think about where it can be relative to your current position "
        "and what action will get you closer to it.\nAction: Pick an action to take from { move forward, turn left, turn right }\nNow, given the current image, "
        "please provide your Observation, detailed Reasoning, and Action from the available actions { move forward, turn left, turn right } to find the "
        f"{target_category}. Please do not repeat this prompt; only generate your response as a JSON object. \n"
    )
    message = [
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

    was_originally_valid = False
    final_action = np.random.choice(VALID_ACTIONS)

    try:
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_input, _ = process_vision_info(message)
        inputs = processor(text=[text], images=image_input, return_tensors="pt", padding=True)
        inputs = inputs.to(device)

        gen_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False, # Use greedy decoding for deterministic output
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                generation_config=gen_config
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # The VLM sometimes outputs json``` before the actual JSON
        _, sep, vlm_output = output_text[0].partition("json")
        if sep:
            vlm_output, sep, _ = vlm_output.partition("```")
        else:
            vlm_output = output_text[0]

        answer = json.loads(vlm_output)
        initial_predicted_action = answer['Action']

        if initial_predicted_action in VALID_ACTIONS:
            was_originally_valid = True
            final_action = initial_predicted_action
        else:
            print(f"Warning: Model predicted invalid action '{initial_predicted_action}'. Using random: {final_action}.")
            # Keep was_originally_valid as False
            # Keep the random final_action

    except Exception as e:
        print(f"Error during model prediction/processing: {e}. Using random action: {final_action}")
        # Keep was_originally_valid as False
        # Keep the random final_action

    # Return both the action to execute and the validity flag
    return final_action, was_originally_valid


def evaluate_model(args):
    print("Starting evaluation...")
    model, processor = load_model_and_processor(args.model_path, args.processor_path, DEVICE)

    metadata_files = glob.glob(os.path.join(args.dataset_dir, "episode_*", "metadata.json"))
    if not metadata_files:
        print(f"Error: No metadata files found in {args.dataset_dir}. Check path.")
        return

    print(f"Found {len(metadata_files)} episodes to evaluate.")
    if args.visualize:
        print("Visualization enabled.")

    all_trial_results = []
    total_predictions_made = 0
    total_valid_actions_predicted = 0
    total_aligned_actions = 0
    total_expert_steps_considered = 0
    total_successes = 0
    total_steps_taken_all_trials = 0

    for episode_idx, meta_path in enumerate(metadata_files):
        print(f"\n--- Evaluating Episode {episode_idx + 1}/{len(metadata_files)} ({os.path.basename(os.path.dirname(meta_path))}) ---")
        sim = None
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            scene_id = metadata['scene_id']
            start_position = np.array(metadata['start_position'])
            start_rotation = metadata['start_rotation']
            target_final_pos = np.array(metadata['final_position'])
            object_category = metadata.get('object_category', 'unknown')
            expert_actions = metadata['actions']
            sim_config = make_sim_config(args.scene_config, scene_id, IMG_WIDTH, IMG_HEIGHT)
            sim = habitat_sim.Simulator(sim_config)
            agent = sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([start_position[0], start_position[1], 0.8])
            agent_state.rotation = habitat_sim.utils.quat_from_coeffs(start_rotation) if isinstance(start_rotation, list) else start_rotation
            agent.set_state(agent_state, infer_sensor_states=False)

            steps_taken = 0
            current_trial_aligned = 0
            is_success = False
            agent_positions_history = [agent_state.position.tolist()]

            # Run simulation loop
            while steps_taken < MAX_STEPS:
                obs = sim.get_sensor_observations()
                rgb_obs = obs['rgb']
                current_image = Image.fromarray(rgb_obs)

                # Predict action and update counters
                predicted_action, was_valid = predict_action(model, processor, current_image, object_category, DEVICE)
                total_predictions_made += 1
                if was_valid:
                    total_valid_actions_predicted += 1

                expert_action = None
                alignment_text = ""
                if steps_taken < len(expert_actions):
                    expert_action = expert_actions[steps_taken].lower()
                    if predicted_action == expert_action:
                        current_trial_aligned += 1
                        alignment_text = f"Expert: {expert_action} (Aligned!)"
                    else:
                        alignment_text = f"Expert: {expert_action} (Misaligned)"
                else:
                    alignment_text = "(No expert action)"

                # Execute action
                if predicted_action in agent.agent_config.action_space:
                    sim.step(predicted_action)
                    agent_state = agent.get_state()
                    agent_positions_history.append(agent_state.position.tolist())
                else:
                    print(f"Error: Corrected action '{predicted_action}' not in action space? Skipping sim step.")

                current_pos = agent_state.position
                distance_to_target = np.linalg.norm(current_pos - target_final_pos)

                print(f"Step {steps_taken}: Pred: {predicted_action} (Valid: {was_valid}) | {alignment_text} | Dist: {distance_to_target:.2f}m")

                # Visualize actions
                if args.visualize:
                    display_image = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    color = (0, 255, 0)
                    thickness = 1
                    line_type = cv2.LINE_AA

                    cv2.putText(display_image, f"Step: {steps_taken}/{MAX_STEPS}", (10, 20), font, font_scale, color, thickness, line_type)
                    cv2.putText(display_image, f"Target: {object_category}", (10, 40), font, font_scale, color, thickness, line_type)
                    # Show validity of the prediction
                    valid_color = (0, 255, 0) if was_valid else (0, 0, 255) # Green if valid, Red if not
                    cv2.putText(display_image, f"Pred Action: {predicted_action}", (10, 60), font, font_scale, valid_color, thickness, line_type)
                    if expert_action:
                        align_color = (0, 255, 0) if predicted_action == expert_action else (0, 0, 255)
                        cv2.putText(display_image, f"Expert Action: {expert_action}", (10, 80), font, font_scale, align_color, thickness, line_type)
                    cv2.putText(display_image, f"Dist to Target: {distance_to_target:.2f}m", (10, 100), font, font_scale, color, thickness, line_type)
                    if is_success:
                         cv2.putText(display_image, "SUCCESS!", (IMG_WIDTH // 2 - 50, IMG_HEIGHT // 2), font, 1.0, (0, 255, 255), 2, line_type)

                    cv2.imshow("Evaluation View", display_image)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                         print("Visualization quit signal received.")
                         args.visualize = False
                         cv2.destroyAllWindows()

                # Check success condition
                if distance_to_target <= SUCCESS_DISTANCE:
                    is_success = True
                    print(f"Success! Reached target within {SUCCESS_DISTANCE}m.")
                    if args.visualize:
                        display_image = cv2.cvtColor(sim.get_sensor_observations()['rgb'], cv2.COLOR_RGB2BGR)
                        cv2.putText(display_image, f"Step: {steps_taken+1}", (10, 20), font, font_scale, color, thickness, line_type)
                        cv2.putText(display_image, f"Target: {object_category}", (10, 40), font, font_scale, color, thickness, line_type)
                        cv2.putText(display_image, f"Dist: {distance_to_target:.2f}m", (10, 60), font, font_scale, color, thickness, line_type)
                        cv2.putText(display_image, "SUCCESS!", (IMG_WIDTH // 2 - 60, IMG_HEIGHT // 2), font, 1.2, (0, 255, 255), 2, line_type)
                        cv2.imshow("Evaluation View", display_image)
                        cv2.waitKey(500)
                    break

                steps_taken += 1

            # Display trail results
            print(f"Trial ended. Success: {is_success}, Steps Taken: {steps_taken}")
            trial_data = {
                "episode_path": meta_path,
                "success": is_success,
                "steps_taken": steps_taken,
                "aligned_actions": current_trial_aligned,
                "expert_actions_count": len(expert_actions),
                "max_possible_aligned_steps": min(len(expert_actions), steps_taken if not is_success else steps_taken + 1)
            }
            all_trial_results.append(trial_data)
            total_successes += int(is_success)
            total_steps_taken_all_trials += steps_taken
            total_aligned_actions += current_trial_aligned
            comparable_steps_in_trial = min(len(expert_actions), steps_taken if not is_success else steps_taken + 1)
            total_expert_steps_considered += comparable_steps_in_trial

        except Exception as e:
            print(f"Error during evaluation of episode {meta_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if sim is not None:
                sim.close()

    if args.visualize:
        cv2.destroyAllWindows()

    # Calculate final metrics
    num_total_trials = len(all_trial_results)
    if num_total_trials == 0:
        print("No trials were successfully processed.")
        return

    overall_success_rate = (total_successes / num_total_trials) * 100 if num_total_trials > 0 else 0
    average_steps = total_steps_taken_all_trials / num_total_trials if num_total_trials > 0 else 0
    overall_action_alignment = (total_aligned_actions / total_expert_steps_considered) * 100 if total_expert_steps_considered > 0 else 0
    valid_action_percentage = (total_valid_actions_predicted / total_predictions_made) * 100 if total_predictions_made > 0 else 0

    print("\n--- Evaluation Summary ---")
    print(f"Total Trials: {num_total_trials}")
    print(f"Total Predictions Made: {total_predictions_made}")
    print(f"Valid Actions Predicted: {total_valid_actions_predicted} ({valid_action_percentage:.2f}%)") # Added
    print(f"Successful Trials: {total_successes} ({overall_success_rate:.2f}%)")
    print(f"Average Steps Taken per Trial: {average_steps:.2f} (Max Steps per Trial: {MAX_STEPS})")
    print(f"Overall Action Alignment: {overall_action_alignment:.2f}% ({total_aligned_actions} aligned actions out of {total_expert_steps_considered} comparable steps)")

    results_filename = f"evaluation_results_{os.path.basename(args.model_path)}.json"
    try:
        summary_data = {
            "total_trials": num_total_trials,
            "total_predictions_made": total_predictions_made,
            "total_valid_actions_predicted": total_valid_actions_predicted,
            "valid_action_percentage": valid_action_percentage,
            "successful_trials": total_successes,
            "success_rate_percent": overall_success_rate,
            "average_steps_per_trial": average_steps,
            "action_alignment_percent": overall_action_alignment,
            "total_aligned_actions": total_aligned_actions,
            "total_comparable_expert_steps": total_expert_steps_considered
        }
        with open(results_filename, "w") as f:
            json.dump({
                "summary": summary_data,
                "detailed_results": all_trial_results
            }, f, indent=2)
        print(f"Detailed results saved to {results_filename}")
    except Exception as e:
        print(f"Error saving results to {results_filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM Navigation Agent")
    parser.add_argument("--visualize",action="store_true",default=False,help="Enable OpenCV visualization.")
    parser.add_argument("--dataset-dir",type=str,default=DEFAULT_DATASET_DIR,help="Path to the episode dataset directory.")
    parser.add_argument("--model-path",type=str,default=DEFAULT_MODEL_PATH,help="Path to the VLM model.")
    parser.add_argument("--processor-path",type=str,default=DEFAULT_PROCESSOR_PATH,help="Path to the VLM processor.")
    parser.add_argument("--scene-config",type=str,default=DEFAULT_SCENE_CONFIG,help="Path to Habitat scene config.")
    args = parser.parse_args()
    evaluate_model(args)