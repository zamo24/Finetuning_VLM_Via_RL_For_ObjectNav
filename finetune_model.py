import os, json
import numpy as np
import math
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from torchvision import transforms
import time
from qwen_vl_utils import process_vision_info
from navigation_dataset import NavigationDataset
from torch.utils.data import DataLoader
from spaceqwen_awr_agent import SpaceQwenAWRAgent
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import argparse

GOAL_DISTANCE_THRESHOLD = 1.0 # meters
IGNORE_INDEX = -100

num_training_steps = 0

class SaveLoRACallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        out_dir = f"lora_failure_trans_epoch_{epoch:02d}_{num_training_steps}_training_steps"
        pl_module.model.save_pretrained(out_dir)
        print(f"[LoRA] Saved adapter to {out_dir}/")

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

# Returns the SpaceQwen model (base or with LoRA adapter) and the processor
def load_peft_model_and_processor(lora_path=None):
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "remyxai/SpaceQwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")
    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
        print(f"Loaded LoRA adapter from {lora_path}")
    else:
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none")
        model = get_peft_model(base_model, lora_config)
        model.gradient_checkpointing_enable()
    return model, processor

def get_failure_transitions(data_root, transitions):
    episode_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    num_fail_trans = 0
    for episode_dir in episode_dirs:
        metadata_path = os.path.join(episode_dir, "metadata.json")

        with (open(metadata_path, 'r')) as f:
            episode_data = json.load(f)
        trajectory = episode_data["trajectory"]
        target_category = episode_data.get("object_category", None)
        
        num_steps = len(trajectory)
        prev_action = None
        for i in range(num_steps - 1):
            step_info = trajectory[i]
            next_step_info = trajectory[i+1]
            action_str = step_info["action"]
            
            img_path = os.path.join(episode_dir, f"images/state_{step_info['step']:04d}_rgb.png")
            next_img_path = os.path.join(episode_dir, f"images/state_{next_step_info['step']:04d}_rgb.png")
            
            # Collect positions/orientations for reward calculation
            position = step_info.get("position", None)
            orientation = step_info.get("orientation", None)
            
            next_position = next_step_info.get("position", None)
            next_orientation = next_step_info.get("orientation", None)

            reward = step_info['reward']

            # done = False
            transitions.append({
                "image": img_path,
                "next_image": next_img_path,
                "action": action_str,
                "position": position,
                "orientation": orientation,
                "next_position": next_position,
                "next_orientation": next_orientation,
                "target_category": target_category,
                # "target_position": target_position,
                "reward": reward,
                # "done": done
            })
            num_fail_trans += 1
    
    print("Number of failure transitions: ", num_fail_trans)

def get_transitions_from_output_dataset(data_root, transitions):
    episode_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    i = 0
    for episode_dir in episode_dirs:
        if (i % 10) == 0:
            i += 1
            continue
        i += 1

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
            
            img_path = os.path.join(episode_dir, f"images/state_{step_info['step']:04d}_rgb.png")
            next_img_path = os.path.join(episode_dir, f"images/state_{next_step_info['step']:04d}_rgb.png")
            
            # Collect positions/orientations for reward calculation
            position = step_info.get("position", None)
            orientation = step_info.get("orientation", None)
            
            next_position = next_step_info.get("position", None)
            next_orientation = next_step_info.get("orientation", None)
            
            # Compute the reward
            reward = calculate_reward(position, next_position, orientation, target_category, target_position)

            transitions.append({
                "image": img_path,
                "next_image": next_img_path,
                "action": action_str,
                "position": position,
                "orientation": orientation,
                "next_position": next_position,
                "next_orientation": next_orientation,
                "target_category": target_category,
                "target_position": target_position,
                "reward": reward,
                # "done": done
            })

        # Mark the final step of the episode as terminal
        if trajectory:
            final_step = trajectory[-1]
            final_img_path = os.path.join(episode_dir, f"state_{final_step['step']:04d}_rgb.png")
            transitions[-1]["done"] = True

    # get_failure_transitions('./failure_dataset', transitions)

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
                "action": action_str,
                "position": position,
                "orientation": orientation,
                "next_position": next_position,
                "next_orientation": next_orientation,
                "target_category": target_category,
                "target_position": target_position,
                "reward": reward,
                # "done": done
            })

        # Mark the final step of the episode as terminal
        if trajectory:
            final_step = trajectory[-1]
            final_img_path = os.path.join(episode_dir, f"state_{final_step['step']:04d}_rgb.png")
            transitions[-1]["done"] = True

    # get_failure_transitions('./failure_dataset', transitions)

    get_transitions_from_output_dataset('./output_dataset', transitions)

    global num_training_steps
    num_training_steps = len(transitions)

    return transitions

# This function runs inference on the model given a message and a text prompt
def model_inference(model, processor, message, text_prompt):
    image_inputs, _ = process_vision_info(message)

    encodings = processor(
        text=[text_prompt], images=image_inputs,
        return_tensors="pt", padding=True
    ).to('cuda')
    input_ids = encodings["input_ids"].to('cuda')
    pixel_values = encodings["pixel_values"].to('cuda')
    attention_mask = encodings['attention_mask'].to('cuda')

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

def custom_collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch: return None # Return None if batch becomes empty

    elem = batch[0]
    collated = {}
    # Iterate through keys found in the first sample
    for key in elem.keys():
        if key in ["state_encoding", "next_state_encoding"]:
            # Handle dictionary collation separately (stack tensors within dict)
            list_of_dicts = [d[key] for d in batch]
            inner_collated = {}
            # Assuming all inner dicts have same keys and tensors are stackable
            for inner_key in list_of_dicts[0].keys():
                 try:
                     inner_collated[inner_key] = torch.stack([d[inner_key] for d in list_of_dicts])
                 except Exception as e:
                      print(f"Error stacking inner key '{inner_key}' in '{key}': {e}")
                      inner_collated[inner_key] = None
            collated[key] = inner_collated

        elif isinstance(elem[key], torch.Tensor):
            # If it's already a tensor, stack them
            try:
                collated[key] = torch.stack([d[key] for d in batch])
            except Exception as e:
                print(f"Error stacking key '{key}': {e}")
                if key == 'training_labels':
                    collated[key] = pad_sequence([d[key] for d in batch], batch_first=True, padding_value=IGNORE_INDEX)
                else:
                     collated[key] = None # Fallback

        elif isinstance(elem[key], (int, float, bool)):
            # Convert sequences of basic types to tensors
            collated[key] = torch.tensor([d[key] for d in batch])
        else:
             collated[key] = [d[key] for d in batch] # Keep as list if unsure

    return collated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LoRA adapter to load")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    transitions = get_transitions_from_dataset("./train_dataset")

    model, processor = load_peft_model_and_processor(args.lora_path)

    # Pre-tokenize transitions and aggregate into trainable format using torch.utils.data.Dataset extention
    dataset = NavigationDataset(processor, transitions)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch),
        persistent_workers=False
    )

    # Initialize AWR agent
    rl_agent = SpaceQwenAWRAgent(model, processor)

    print(f"Gradient Checkpointing Enabled: {model.is_gradient_checkpointing}")

    # Instantiate the Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=True,
        callbacks=[SaveLoRACallback()],
        # callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        precision='bf16-mixed',
        accumulate_grad_batches=4,
        enable_progress_bar=True,
        # log_every_n_steps=10,
        # val_check_interval=1.0,
    )
    print("Trainer configured. Starting training...")
    start_time = time.time()
    trainer.fit(model=rl_agent, train_dataloaders=loader)
    elapsed_time = time.time() - start_time
    print(f"Training took: {elapsed_time:.4f} seconds")

    adapter_save_directory = f"./final_adapter_{num_training_steps}_training_steps"
    print(f"Saving final LoRA adapter to: {adapter_save_directory}")
    # Access the PEFT model within LightningModule
    peft_model = rl_agent.model
    peft_model.save_pretrained(adapter_save_directory)
    print("Adapter saved.")
    
    print('training done')

if __name__ == "__main__":
    main()
