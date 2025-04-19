import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

# Define the fixed maximum length for the training labels.
MAX_LABEL_LENGTH = 512
IGNORE_INDEX = -100

# Constructs a fixed-length target label for teacher forcing
def construct_action_label(tokenizer, action_text, text_prompt, max_length=MAX_LABEL_LENGTH):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must have a pad_token or eos_token set.")
    pad_token_id = tokenizer.pad_token_id

    quote_token_id_list = tokenizer.encode('"', add_special_tokens=False)
    if not quote_token_id_list:
        # Handle tokenizers that might not encode quote alone or error
        print("Warning: Could not encode quote token '\"'. End boundary detection might fail.")
        quote_token_id = None
    else:
        quote_token_id = quote_token_id_list[0]

    target_template = f"{str(text_prompt)}\n" + '{ "Observation": "", "Reasoning": "", "Action": "' + str(action_text) + '" }'

    padding_side = tokenizer.padding_side
    encoding = tokenizer(
        target_template,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        padding_side=padding_side,
        return_tensors="pt"
    )
    original_ids = encoding["input_ids"][0].clone()
    labels = encoding["input_ids"][0].clone()
    labels_list = labels.tolist()

    possible_keys = [
        '"Action":', ' "Action":', '"Action": ', ' "Action": ', 'Action":', 'Action:'
    ]
    action_key_ids_found = None
    found_key_str_used = None
    initial_action_start_idx = None # Index right after the key pattern
    final_action_start_idx = None   # Index after skipping space/quote

    for key_str in possible_keys:
        key_ids_candidate = tokenizer.encode(key_str, add_special_tokens=False)
        if not key_ids_candidate: continue

        for i in range(len(labels_list) - len(key_ids_candidate) + 1):
            if labels_list[i : i + len(key_ids_candidate)] == key_ids_candidate:
                initial_action_start_idx = i + len(key_ids_candidate)
                action_key_ids_found = key_ids_candidate
                found_key_str_used = key_str

                current_idx = initial_action_start_idx
                num_skipped = 0
                skipped_tokens = []

                # Common space prefixes/representations used by tokenizers
                space_indicators = ['Ġ', ' ', '_']

                # Try skipping up to 2 tokens
                for iter_num in range(2):
                    if current_idx < len(labels_list):
                        next_token_id = labels_list[current_idx]
                        token_string = None
                        is_convertible = True
                        # Get the token string representation directly
                        try:
                            # Use convert_ids_to_tokens which gives the underlying token representation
                            token_string = tokenizer.convert_ids_to_tokens([next_token_id])[0]
                        except Exception as e:
                            # Fallback to decode if convert fails? Or just mark as not convertible
                            try:
                                 token_string = tokenizer.decode([next_token_id])
                            except:
                                 is_convertible = False

                        is_space_variant = False
                        if is_convertible and token_string is not None:
                            # Check if the token string itself represents space or starts with a space prefix
                            is_space_variant = token_string.isspace() or \
                                               any(token_string.startswith(prefix) for prefix in space_indicators)
                        else:
                             is_space_variant = False


                        # Explicitly check for quote using the ID
                        is_quote = (quote_token_id is not None and next_token_id == quote_token_id)

                        # Check if the token qualifies as skippable (space variant OR quote)
                        if is_space_variant or is_quote:
                            skipped_tokens.append(f"'{token_string}' (ID:{next_token_id})")
                            current_idx += 1 # Actually advance index for next check/final value
                            num_skipped += 1
                        else:
                            break # Exit the skipping loop (found non-space/quote)
                    else:
                        break # Exit loop if end of sequence reached

                final_action_start_idx = current_idx # This is the index AFTER the last skipped token
                break # Exit inner loop (key instance found)
        if final_action_start_idx is not None:
            break # Exit outer loop (key type found and processed)


    if final_action_start_idx is None: # Check if loop finished without finding key/start
        print(f"ERROR: Could not find action key in template:\n{target_template}")
        print(f"Tokenized IDs ({padding_side}-padded): {labels_list}")
        raise ValueError("Action key not found in the tokenized target after trying variations.")

    # Find Action end index
    action_end_idx = None
    if quote_token_id is not None:
        for j in range(final_action_start_idx, len(labels_list)):
            if labels_list[j] == pad_token_id:
                action_end_idx = j
                break
            if labels_list[j] == quote_token_id:
                action_end_idx = j
                break
        # If loop finished without break (no quote/padding found):
        if action_end_idx is None and j == len(labels_list) - 1:
             action_end_idx = len(labels_list)

    if action_end_idx is None:
        print(f"Warning: Closing quote/padding for action value not found after index {final_action_start_idx}. Supervising until sequence end (length {max_length}).")
        action_end_idx = max_length

    # Mask all takens besides the action token(s)
    if final_action_start_idx >= 0:
         labels[:final_action_start_idx] = IGNORE_INDEX
    else:
         print("Warning: final_action_start_idx is negative, not masking beginning.")
    if action_end_idx <= max_length:
         labels[action_end_idx:] = IGNORE_INDEX
    else:
         print("Warning: action_end_idx is beyond max_length, not masking end.")


    # Mask padding tokens
    if pad_token_id is not None:
        padding_mask = (original_ids == pad_token_id)
        # Count changes only if masking actually happens
        if padding_mask.any():
             initial_ignored = (labels == IGNORE_INDEX).sum()
             labels[padding_mask] = IGNORE_INDEX
             final_ignored = (labels == IGNORE_INDEX).sum()
             if final_ignored > initial_ignored:
                  print(f"Explicitly masked {final_ignored - initial_ignored} padding tokens (ID: {pad_token_id}).")

    return labels

class NavigationDataset(torch.utils.data.Dataset):
    def __init__(self, processor, transitions, text_prompts, messages, raw_texts):
        print(f"Initializing NavigationDataset with {len(transitions)} transitions...")
        self.processor = processor
        # Store data needed to process one item in __getitem__
        # We need transitions up to len-1 because we look at transition[i+1] for next state
        self.valid_indices = len(transitions) - 1
        self.transitions = transitions[:self.valid_indices] # Only store necessary transitions
        self.next_transitions = transitions[1:] # Store next state info separately
        self.text_prompts = text_prompts[:self.valid_indices]
        self.next_text_prompts = text_prompts[1:]
        self.messages = messages[:self.valid_indices]
        self.next_messages = messages[1:]
        self.raw_texts = raw_texts[:self.valid_indices]

        # Check lengths match
        assert len(self.transitions) == self.valid_indices
        assert len(self.text_prompts) == self.valid_indices
        assert len(self.messages) == self.valid_indices
        assert len(self.raw_texts) == self.valid_indices
        assert len(self.next_transitions) == self.valid_indices
        assert len(self.next_text_prompts) == self.valid_indices
        assert len(self.next_messages) == self.valid_indices

        print(f"Dataset initialized. Processing will occur in __getitem__ for {self.valid_indices} samples.")


    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        if idx >= self.valid_indices:
            raise IndexError("Index out of bounds")

        # Get data for the current transition (state, action, reward)
        curr_trans = self.transitions[idx]
        text_prompt = self.text_prompts[idx]
        raw_text = self.raw_texts[idx]
        action_str = curr_trans['action']
        reward = curr_trans['reward']

        # Get data for the next state
        next_trans = self.next_transitions[idx]
        next_text_prompt = self.next_text_prompts[idx]

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

        # action_ids = self.processor.tokenizer.encode(action_str, add_special_tokens=False)
        # target_category_ids = self.processor.tokenizer.encode(target_category_str, add_special_tokens=False)

        # Construct the training label
        try:
            training_labels = construct_action_label(
                self.processor.tokenizer,
                action_str,
                raw_text,
                max_length=MAX_LABEL_LENGTH
            )
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