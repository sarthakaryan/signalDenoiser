import os
import re

def find_max_epoch(ckpt_directory):
    """Find the maximum epoch number in the checkpoint directory."""
    max_epoch = -1
    for file_name in os.listdir(ckpt_directory):
        match = re.match(r'(\d+)\.pkl', file_name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
    return max_epoch

def loss_fn(model, X):
    clean_audio, noisy_audio = X
    predicted_audio = model(noisy_audio)
    loss = model.compute_loss(clean_audio, predicted_audio)
    loss_dic = {'loss': loss.item()}
    return loss, loss_dic
