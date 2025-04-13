import os, json, gzip
from PIL import Image
import numpy as np
import math
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from peft import LoraConfig, get_peft_model
import torch
from torchvision import transforms
import time
from qwen_vl_utils import process_vision_info
from navigation_dataset import MyClass

GOAL_DISTANCE_THRESHOLD = 2.0 # meters

# Function to compute angle between two 3D vectors (in degrees)
def angle_between(vec1, vec2):
    # Normalize vectors
    v1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
    v2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
    # Dot product and angle
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

# This function calculates the reward based on the euclidean distance from the goal object
def calculate_reward(position, next_position, orientation, target_category, target_position):
    reward = 0
    # Distance-based rewards
    done = False
    if position is not None and next_position is not None and target_position is not None:
        current_dist = np.linalg.norm(np.array(target_position) - np.array(position)).tolist()
        next_dist = np.linalg.norm(np.array(target_position) - np.array(next_position)).tolist()
        # Distance improvement (positive if agent got closer)
        dist_diff = current_dist - next_dist
        reward += 0.1 * dist_diff  # this will add positive reward for approaching (and negative if moving away)
        # Check if reached within 2m in the next state (success)
        if next_dist < GOAL_DISTANCE_THRESHOLD:
            reward += 5.0
            done = True

    return reward

# Returns a the SpaceQwen model wrapped in a LoRA module
def load_peft_model_and_processor():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "remyxai/SpaceQwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)

    return model, processor

# Constructs the specialized prompt
def construct_prompt(processor, target_category, image):
    # Specialized prompt
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
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    return text, message

# Converts the dataset into transitions
def get_transitions_from_dataset(data_root):
    episode_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    transitions = []
    for episode_dir in episode_dirs:
        metadata_path = os.path.join(episode_dir, "metadata.json")

        with (open(metadata_path, 'r')) as f:
            episode_data = json.load(f)
        trajectory = episode_data["trajectory"]
        target_category = episode_data.get("object_category", None)
        target_position = episode_data.get("final_position", None)
        
        num_steps = len(trajectory)
        for i in range(num_steps - 1):
            step_info = trajectory[i]
            next_step_info = trajectory[i+1]
            action_str = step_info["action"]

            # Map action string to an index 0,1,2
            if action_str == "move_forward":
                action_index = 0
            elif action_str == "turn_left":
                action_index = 1
            elif action_str == "turn_right":
                action_index = 2
            
            img_path = os.path.join(episode_dir, f"images/state_{step_info['step']:04d}_rgb.png")
            next_img_path = os.path.join(episode_dir, f"images/state_{next_step_info['step']:04d}_rgb.png")
            
            # Collect positions/orientations for reward calculation
            position = step_info.get("position", None)
            orientation = step_info.get("orientation", None)
            
            next_position = next_step_info.get("position", None)
            next_orientation = next_step_info.get("orientation", None)
            
            # Compute the reward
            reward = calculate_reward(position, next_position, orientation, target_category, target_position)

            done = False
            transitions.append({
                "image": img_path,
                "next_image": next_img_path,
                "action": action_index,
                "position": position,
                "orientation": orientation,
                "next_position": next_position,
                "next_orientation": next_orientation,
                "target_category": target_category,
                "target_position": target_position,
                "reward": reward,
                "done": done
            })

        # Mark the final step of the episode as terminal
        if trajectory:
            final_step = trajectory[-1]
            final_img_path = os.path.join(episode_dir, f"state_{final_step['step']:04d}_rgb.png")
            transitions[-1]["done"] = True

    return transitions

# This function runs inference on the model given a message and a text prompt
def model_inference(model, processor, message, text_prompt):
    image_inputs, video_inputs = process_vision_info(message)

    encodings = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True
    ).to('cuda')
    input_ids = encodings["input_ids"].to('cuda')
    pixel_values = encodings["pixel_values"].to('cuda')
    attention_mask = encodings['attention_mask'].to('cuda')

    # Prepare labels (target outputs) by tokenizing target_texts
    # labels = processor.tokenizer(
    #     text=targets, padding=True, truncation=True,
    #     return_tensors="pt"
    # )["input_ids"]

    gen_config = GenerationConfig(
        max_new_tokens=84,
        bos_token_id=model.config.bos_token_id,
    )

    # Inference: Generation of the output
    start_time = time.time()
    generated_ids = model.generate(
        # input_ids=input_ids, 
        # pixel_values=pixel_values,
        # attention_mask=attention_mask,
        **encodings,
        max_new_tokens=84,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.5f} seconds")

    # Decode generated tokens into text
    generated_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("Generated Outputs:")
    for text in generated_texts:
        print(text)

def main():
    transitions = get_transitions_from_dataset("./output_dataset")

    model, processor = load_peft_model_and_processor()

    # Image transform to get a raw pixel tensor
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std) based on model requirements
    ])

    # Get text prompts, images, messages, target objects, and pixel tensors (transformed images)
    text_prompts = []
    images = []
    targets = []
    pixel_tensors = []
    messages = []
    for transition in transitions:
        target_category = transition['target_category']
        image = Image.open(transition['image']).convert("RGB")

        text, message = construct_prompt(processor, target_category, image)

        text_prompts.append(text)
        messages.append(message)
        targets.append(target_category)
        images.append(image)
        pixel_tensors.append(image_transform(image).unsqueeze(0))

    obj = MyClass(20)
    print(obj.say_hello())



if __name__ == "__main__":
    main()
