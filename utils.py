import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(device))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x, device):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.to(device)
#         x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def vae_weight(global_step, hparams):
    warm_up_step = hparams.vae_warming_up
    if global_step < warm_up_step:
        if global_step % 100 < 1:
            w1 = torch.tensor(hparams.init_vae_weights + global_step / 100  * hparams.vae_weight_multiler, dtype=torch.float32)
        else:
            w1 = torch.tensor(0, dtype=torch.float32)
    else:
        w1 = torch.tensor(0, dtype=torch.float32)

    if global_step > warm_up_step:
        if global_step % 400 < 1:
            w2 = torch.tensor(hparams.init_vae_weights + (global_step - warm_up_step) / 400 * hparams.vae_weight_multiler + warm_up_step / 100 * hparams.vae_weight_multiler, dtype=torch.float32)
        else:
            w2= torch.tensor(0, dtype=torch.float32)
    else:
        w2 = torch.tensor(0, dtype=torch.float32)
    return torch.max(w1, w2)