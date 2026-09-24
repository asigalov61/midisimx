# =================================================================================================
#
# multi_label_cls.py - Multi-Label Classifier (MLCLS) transformer module
# Project Los Angeles / Tegridy Code 2026
# Apache 2.0 / Version 1.0.0
#
# =================================================================================================
#
# Dependencies:
#
# x-transformers==2.3.1 by lucidrains
# torch>=2.7.0
#
# ================================================================================================

"""
Simple usage example

NUM_CLASSES = 30
SEQ_LEN     = 1024
PAD_IDX     = 768
VOCAB_SIZE  = 769

model = load_model('model_checkpoint.pth', NUM_CLASSES)

model.cuda()

model.eval()

# ---------------------------------------

inp_seq = [0, 128, 256, 384, ...]

preds, probs = predict(model, [inp_seq])
"""

# =================================================================================================

import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

from torch.utils.data import Dataset, DataLoader

from .x_transformer_2_3_1 import TransformerWrapper, Encoder

# =================================================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================================================

dtype = torch.bfloat16

ctx = torch.amp.autocast(device_type=DEVICE, dtype=dtype)

# =================================================================================================

# -------------------------------------------------------------------------------------------------
# Multi-label classifier model
# -------------------------------------------------------------------------------------------------

def build_mlcls_model(num_classes=30,
                      vocab_size=769,
                      max_seq_len=1024,
                      pad_idx=768,
                      dim=384,
                      depth=8,
                      heads=8,
                      emb_dropout=0.2,
                      layer_dropout=0.2,   # stochastic depth - dropout entire layer
                      attn_dropout=0.2,    # dropout post-attention
                      ff_dropout=0.2,
                      use_cls_token=False,
                      average_pool_embed=True,
                      rotary_pos_emb=True,
                      attn_flash=True,
                      device='cuda'
                     ):
    """
    Transformer encoder -> average pooled -> multi-logits (multi-class).
    """
    model = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        logits_dim=num_classes,
        use_cls_token=use_cls_token,
        average_pool_embed=average_pool_embed,
        emb_dropout=emb_dropout,
        attn_layers=Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            rotary_pos_emb=rotary_pos_emb,
            attn_flash=attn_flash,
            layer_dropout=layer_dropout,   # stochastic depth - dropout entire layer
            attn_dropout=attn_dropout,     # dropout post-attention
            ff_dropout=ff_dropout          # feedforward dropout
        ),
    )
    return model.to(device)
    
# -------------------------------------------------------------------------------------------------

def load_mlcls_model(checkpoint_path, num_classes, device='cuda', **kwargs):
    """
    Rebuilds the architecture, loads weights.
    """
    model = build_mlcls_model(device=device, num_classes=num_classes, **kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# -------------------------------------------------------------------------------------------------

class MLCLSInferenceDataset(Dataset):
    """
    Inference dataset class
    
    src_seq: list of token IDs (ints).
    """
    def __init__(self, data):
        self.data_pairs = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_seq = self.data[idx]
        x = torch.tensor(src_seq, dtype=torch.long)
        return x

# -------------------------------------------------------------------------------------------------

@torch.no_grad()
def predict_mlcls(model, seqs, device='cuda', batch_size=512, pad_idx=768):
    """
    Returns two lists:
      - preds: int class predictions (0, 1, 2, 3)
      - probs: float probabilities for the predicted class
    """
    device = torch.device(device)
    
    model.eval()
    
    # --- 1. Pad sequences and build masks ---
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 1
    
    x = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    
    for i, (seq, l) in enumerate(zip(seqs, lengths)):
        x[i, :l] = torch.tensor(seq, dtype=torch.long)
        mask[i, :l] = True

    all_preds = []
    all_probs = []

    # --- 2. Batched Inference ---
    for i in range(0, len(seqs), batch_size):
        batch_x = x[i:i+batch_size].to(device, non_blocking=True)
        batch_mask = mask[i:i+batch_size].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch_x, mask=batch_mask)  # [B, 4]

        # Softmax to get probabilities, then pick the max
        probs = torch.softmax(logits.float(), dim=-1)  # [B, 4]
        confidences, preds = probs.max(dim=-1)         # [B], [B]

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(confidences.cpu().tolist())

    return all_preds, all_probs

# =================================================================================================
# This is the end of multi_label_cls Python module
# =================================================================================================