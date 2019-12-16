import torch

def vae_weight(global_step):
    warm_up_step = hparams.vae_warming_up
    if global_step < warm_up_step:
        if global_step % 100 < 1:
            w1 = torch.tensor(hparams.init_vae_weights + global_step / 100  * hparams.vae_weight_multiler, torch.float32)
        else:
            w1 = torch.tensor(0, torch.float32)
    else:
        w1 = torch.tensor(0, torch.float32)

    if global_step > warm_up_step:
        if global_step % 400 < 1:
            w2 = torch.tensor(hparams.init_vae_weights + (global_step - warm_up_step) / 400 * hparams.vae_weight_multiler + warm_up_step / 100 * hparams.vae_weight_multiler, torch.float32)
        else:
            w2= torch.tensor(0, torch.float32)
    else:
        w2 = torch.tensor(0, torch.float32)
    return torch.max(w1, w2)
