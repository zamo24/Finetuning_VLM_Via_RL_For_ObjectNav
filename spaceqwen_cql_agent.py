import pytorch_lightning as pl
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

IGNORE_INDEX = -100


class SpaceQwenCQLAgent(pl.LightningModule):
    def __init__(self, model, processor, alpha=01e-4, beta=5e-3, gamma=0.99, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'processor'])

        self.model = model
        self.processor = processor

        self.automatic_optimization = True

        # self.target_model = copy.deepcopy(model)
        # self.target_model.eval()
        # for param in self.target_model.parameters():
        #     param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # Skip empty batches
        if batch is None:
            return None

        # Extract encodings and labels
        state_enc = batch.get("state_encoding")
        labels = batch.get("training_labels")
        rewards = batch.get("reward")
        if state_enc is None or labels is None or rewards is None:
            return None

        # Move to VRAM
        device = self.device
        state_enc = {k: v.to(device) for k, v in state_enc.items()}
        labels    = labels.to(device)
        rewards   = rewards.to(device).float()

        # Forward pass
        outputs = self.model(**state_enc)
        logits  = outputs.logits

        # Shift for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T, V = shift_logits.size()

        # Flatten for CE
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        # CE per-token
        ce_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        per_seq_loss = (
            ce_losses.view(B, T) * (shift_labels != IGNORE_INDEX)
        ).sum(dim=1) / (shift_labels != IGNORE_INDEX).sum(dim=1).clamp(min=1)

        # Normalize rewards to advantages
        r_mean = rewards.mean()
        r_std  = rewards.std(unbiased=False).clamp(min=1e-6)
        adv    = (rewards - r_mean) / r_std
        weights = adv.clamp(min=0)

        # Reward-weighted regression loss
        rwr_loss = (weights * per_seq_loss).mean()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy_per_token = -(log_probs * log_probs.exp()).sum(dim=-1)  # (B, T)
        ent_bonus = entropy_per_token.mean()

        # L2 regularization on LoRA adapter params
        l2_reg = torch.tensor(0.0, device=device)
        for n, p in self.model.named_parameters():
            if "lora_" in n and p.requires_grad:
                l2_reg = l2_reg + (p ** 2).sum()

        loss = rwr_loss - self.hparams.beta * ent_bonus + self.hparams.alpha * l2_reg

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("rwr_loss", rwr_loss, on_step=True, on_epoch=False)
        self.log("ent_bonus", ent_bonus, on_step=True, on_epoch=False)
        self.log("l2_reg", l2_reg, on_step=True, on_epoch=False)
        self.log("avg_weight", weights.mean(), on_step=True, on_epoch=False)

        return loss


    def configure_optimizers(self):
        lr = self.hparams.learning_rate if hasattr(self.hparams, 'learning_rate') else 1e-4
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        return optimizer

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #      # Only update target if training step didn't return None (i.e., wasn't skipped)
    #      if outputs is not None:
    #          # Example: Polyak update target network
    #          tau = 0.005 # Consider making this a hparam
    #          with torch.no_grad():
    #              for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
    #                   # Ensure target_param also doesn't require grad just in case
    #                   # target_param.requires_grad = False
    #                   target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
