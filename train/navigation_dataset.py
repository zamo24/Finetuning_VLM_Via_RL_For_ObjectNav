import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

# Define the fixed maximum length for the training labels.
MAX_LABEL_LENGTH = 512
IGNORE_INDEX = -100
ACTION_MAP = {
    "move_forward": "move forward",
    "move forward": "move forward",
    "turn_left": "turn left",
    "turn left": "turn left",
    "turn_right": "turn right",
    "turn right": "turn right"
}

# Creates training labels where only the tokens corresponding to the
# LAST occurrence of `action_text` within the tokenized `target_template`
# are unmasked. Everything else is set to IGNORE_INDEX.
def construct_action_label(tokenizer, action_text, text_prompt, max_length=MAX_LABEL_LENGTH):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must have a pad_token or eos_token set.")
    pad_token_id = tokenizer.pad_token_id

    target_template = f"{str(text_prompt)}\n" + '{ "Observation": "", "Reasoning": "", "Action": "' + str(action_text) + '" }'

    padding_side = tokenizer.padding_side
    encoding = tokenizer(
        target_template,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Use input_ids directly as the source, keeping padding for length consistency
    template_token_ids = encoding["input_ids"][0]
    template_tokens_list = template_token_ids.tolist()

    action_token_ids = tokenizer.encode(action_text, add_special_tokens=False)

    if not action_token_ids:
        print(f"Warning: Could not tokenize action_text: '{action_text}'")
        # Return a fully masked label tensor
        return torch.full_like(template_token_ids, IGNORE_INDEX)

    action_val_start_idx = -1
    len_action = len(action_token_ids)
    len_template = len(template_tokens_list)

    for i in range(len_template - len_action, -1, -1):
         if template_tokens_list[i : i + len_action] == action_token_ids:
              action_val_start_idx = i
              break

    labels = torch.full_like(template_token_ids, IGNORE_INDEX)

    if action_val_start_idx != -1:
        action_val_end_idx = action_val_start_idx + len_action
        labels[action_val_start_idx : action_val_end_idx] = template_token_ids[action_val_start_idx : action_val_end_idx]
        num_unmasked = (labels != IGNORE_INDEX).sum().item()

        if num_unmasked != len_action:
             print(f"Warning: Number of unmasked tokens ({num_unmasked}) doesn't match action token length ({len_action})")
    else:
        print(f"Warning: Could not find action token sequence {action_token_ids} for '{action_text}' in the tokenized template.")

    return labels

# Constructs the specialized prompt
def construct_prompt(processor, target_category, image):
    # Specialized prompt
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
         f"{target_category}. Please do not repeat this prompt; only generate your response as a valid JSON object. Do now forget the commas in the JSON object!"
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
    return prompt_text, text

class NavigationDataset(torch.utils.data.Dataset):
    def __init__(self, processor, transitions):
        print(f"Initializing NavigationDataset with {len(transitions)} transitions...")
        self.processor = processor
        # Store data needed to process one item in __getitem__
        # We need transitions up to len-1 because we look at transition[i+1] for next state
        self.valid_indices = len(transitions) - 1
        self.transitions = transitions[:self.valid_indices] # Only store necessary transitions
        self.next_transitions = transitions[1:] # Store next state info separately

        # Check lengths match
        assert len(self.transitions) == self.valid_indices
        assert len(self.next_transitions) == self.valid_indices

        print(f"Dataset initialized. Processing will occur in __getitem__ for {self.valid_indices} samples.")


    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        if idx >= self.valid_indices:
            raise IndexError("Index out of bounds")

        # Get data for the current transition (state, action, reward)
        curr_trans = self.transitions[idx]
        raw_text, text_prompt = construct_prompt(self.processor, curr_trans['target_category'], Image.open(curr_trans['image']).convert("RGB"))
        action_str = curr_trans['action']
        reward = curr_trans['reward']

        # Get data for the next state
        next_trans = self.next_transitions[idx]
        _, next_text_prompt = construct_prompt(self.processor, next_trans['target_category'], Image.open(next_trans['image']).convert("RGB"))

        # Process state
        try:
            image = Image.open(curr_trans['image']).convert("RGB")
            image_inputs, _ = process_vision_info([{'role': 'user', 'content': [{'type': 'image', 'image': image}]}])
        except Exception as e:
            print(f"Error loading/processing image {curr_trans['image']} for index {idx}: {e}")
            return None

        # Encode current state
        try:
            state_encoding = self.processor(
                text=[text_prompt], # Use the full templated text prompt
                images=image_inputs,
                padding="max_length", # Pad/truncate state encoding
                max_length=MAX_LABEL_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            # Squeeze unnecessary batch dim added by processor
            state_encoding = {k: v.squeeze(0) for k, v in state_encoding.items()}
        except Exception as e:
            print(f"Error encoding state for index {idx}: {e}")
            return None

        # Process next state
        try:
            next_image = Image.open(next_trans['image']).convert("RGB")
            next_image_inputs, _ = process_vision_info([{'role': 'user', 'content': [{'type': 'image', 'image': next_image}]}])
        except Exception as e:
            print(f"Error loading/processing next image {next_trans['image']} for index {idx}: {e}")
            return None

        try:
            next_state_encoding = self.processor(
                text=[next_text_prompt],
                images=next_image_inputs,
                padding="max_length",
                max_length=MAX_LABEL_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
            next_state_encoding = {k: v.squeeze(0) for k, v in next_state_encoding.items()}
        except Exception as e:
            print(f"Error encoding next state for index {idx}: {e}")
            return None

        # Construct the training label
        try:
            training_labels = construct_action_label(
                self.processor.tokenizer,
                ACTION_MAP[action_str],
                raw_text,
                max_length=MAX_LABEL_LENGTH
            )

            # tensor = torch.where(training_labels == -100, torch.full_like(training_labels, self.processor.tokenizer.pad_token_id), training_labels)
            # decoded_label = self.processor.tokenizer.decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print("Trianing label: ", decoded_label)
        except Exception as e:
             print(f"Error constructing label for index {idx}: {e}")
             return None

        # Create the final sample dictionary
        sample = {
            "state_encoding": state_encoding,
            # "action": torch.tensor(action_ids, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float),
            "next_state_encoding": next_state_encoding,
            # "target_category": torch.tensor(target_category_ids, dtype=torch.long),
            "training_labels": training_labels, # The crucial tensor with -100 masking
            # "done": torch.tensor(done, dtype=torch.bool), # Done flag might be needed for weight calc
        }
        return sample