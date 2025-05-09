import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100

class SpaceQwenAWRAgent(pl.LightningModule):
    def __init__(
        self,
        model,
        processor,
        alpha: float = 1e-5,
        beta: float = 1e-3,
        # beta: float = 0.0,
        temp_adv: float = 1.0,
        learning_rate: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "processor"])

        # Freeze vision tower for stability / VRAM
        for n, p in model.named_parameters():
            if n.startswith("vision_tower") or "visual" in n:
                p.requires_grad = False

        self.model = model
        self.processor = processor

    # ------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        enc    = batch["state_encoding"]
        labels = batch["training_labels"]
        rewards = batch["reward"].float()
        # rewards.clamp(min=0.0)

        device = self.device
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)
        rewards = rewards.to(device)

        logits = self.model(**enc).logits  # (B, T+1, V)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        B, T, V = shift_logits.shape

        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)
        ce_all = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX, reduction="none").view(B, T)

        token_mask = (shift_labels != IGNORE_INDEX)
        per_seq_ce = (ce_all * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)

        # Advantage weighting (batch‑size‑aware)
        if B > 1:
            adv = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
            weights = torch.softmax(adv / self.hparams.temp_adv, dim=0)  # sum=1
        else: # RWR
            # batch size 1; use bounded sigmoid
            weights = torch.sigmoid(rewards / self.hparams.temp_adv)

        rwr_loss = (weights * per_seq_ce).sum() / B

        # Entropy
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy_tok = -(log_probs * log_probs.exp()).sum(-1)
        ent_bonus = (entropy_tok * token_mask).sum() / token_mask.sum().clamp(min=1)

        # LoRA L2
        l2 = torch.tensor(0.0, device=device)
        for n, p in self.model.named_parameters():
            if "lora_" in n and p.requires_grad:
                l2 += (p**2).sum()

        loss = rwr_loss - self.hparams.beta * ent_bonus + self.hparams.alpha * l2

        # Logging
        self.log_dict({
            "loss": loss,
            "rwr": rwr_loss,
            "entropy": ent_bonus,
            "l2": l2,
            "w_mean": weights.mean(),
        }, on_step=True, prog_bar=True, batch_size=B)
        return loss

    def configure_optimizers(self):
        # Only train parameters that require grad (vision frozen)
        optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.hparams.learning_rate)
        return optim

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.cuda.empty_cache()

