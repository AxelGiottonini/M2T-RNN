import sys
import os
from dataclasses import dataclass
import time
import logging
import torch
import torch.nn.functional as F

from .bvr import BVROutput

ADVERSARIAL_MODES = ["aae"]

def __kl_loss(output:BVROutput):
    return -0.5 * torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp()) / len(output.mu)

def __logvar_loss(output:BVROutput):
    return output.logvar.abs().sum(dim=1).mean()

def __adv_loss(output:BVROutput, discriminator):
    fake = torch.randn_like(output.latent)

    zeros = torch.empty(len(output.latent), 2)
    zeros[:, 0] = 1
    zeros[:, 1] = 0
    zeros = zeros.to(output.latent.device)
    ones = 1 - zeros

    loss = F.binary_cross_entropy_with_logits(discriminator(output.latent), zeros)
    d_loss = F.binary_cross_entropy_with_logits(discriminator(output.latent.detach()), ones)
    d_loss = d_loss + F.binary_cross_entropy_with_logits(discriminator(fake), zeros)

    return loss, d_loss

def loss_fn(output, discriminator, args):
    if args["mode"] == "standard":
        return output.loss
    
    if args["mode"] == "vae":
        return output.loss + args["f_kl"] * __kl_loss(output)
    
    if args["mode"] == "aae":
        adv_loss, d_loss = __adv_loss(output, discriminator)
        return (
            output.loss * args["f_logvar"] * __logvar_loss(output) + args["f_adv"] * adv_loss,
            d_loss
        )
    
    raise NotImplementedError()

def __log_epoch(i_epoch, epoch_metrics, metrics, vdl_is_not_none):
    str_metrics = f"Training Loss: {metrics['training/loss/mean'][-1]:.4f}"
    if vdl_is_not_none:
        str_metrics = str_metrics + " | " + f"Validation Loss:{metrics['validation/loss/mean'][-1]:.4f}"
    str_duration = f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_metrics['start_time']))}"
    str_epoch = f"EPOCH[{i_epoch}]" + "\n\t" + str_metrics + "\n\t" + str_duration
    logging.info(str_epoch)

def train_loop(
    model,
    optimizer,
    args,
    discriminator=None,
    d_optimizer=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    precision=torch.bfloat16,
    save_model=True,
    save_optimizer=True
):
    n_epochs = args["n_epochs"]
    global_batch_size = args["global_batch_size"]
    local_batch_size = args["local_batch_size"]
    accumulation_steps = global_batch_size // local_batch_size
    save_each = args["save_each"]

    model.to(device).to(precision)
    if discriminator is not None:
        discriminator.to(device).to(precision)

    metrics = {
        "best_loss": float("inf"),
        "training/loss/step": [],
        "training/loss/mean": [],
        "validation/loss/mean": []
    }

    def save(dir_name):
        if save_model:
            try:
                model.save(os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name))
            except AttributeError:
                torch.save(
                    model.state_dict(),
                    os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name, "model.bin")
                )

        if save_optimizer:
            torch.save(
                optimizer.state_dict(), 
                os.path.join(args["model_dir"], args["model_name"], args["model_version"], dir_name, "optimizer.bin")
            )

    def decorator(step):
        def wrapper(training_dataloader, validation_dataloader=None):
            for i_epoch in range(1, n_epochs+1):
                epoch_metrics = {
                    "start_time": time.time(),
                    "training/loss": [],
                    "validation/loss": []
                }

                try:
                    model.train()
                    torch.cuda.empty_cache()
                    for i_batch, batch in enumerate(training_dataloader):
                        out = step(model, batch.to(device))
                        loss = loss_fn(out, discriminator, args)

                        if args["mode"] in ADVERSARIAL_MODES:
                            loss, d_loss = loss
                            (d_loss / accumulation_steps).backward()
                        (loss / accumulation_steps).backward()

                        if (
                            (i_batch + 1) % accumulation_steps == 0 or
                            (i_batch + 1) == len(training_dataloader) 
                        ):
                            if args["mode"] in ADVERSARIAL_MODES:
                                d_optimizer.step()
                                d_optimizer.zero_grad()
                            optimizer.step()
                            optimizer.zero_grad()

                        metrics["training/loss/step"].append(loss.item())
                        epoch_metrics["training/loss"].append(loss.item())
                        if (
                            (i_batch + 1) % (save_each * accumulation_steps) == 0 or
                            (i_batch + 1) == len(training_dataloader)
                        ):
                            torch.save(
                                metrics, 
                                os.path.join(args["model_dir"], args["model_name"], args["model_version"], "metrics.bin")
                            )

                    metrics["training/loss/mean"].append(torch.tensor(epoch_metrics["training/loss"]).mean())

                    if validation_dataloader is not None:    
                        model.eval()
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            for i_batch, batch in enumerate(validation_dataloader):
                                out = step(model, batch.to(device))
                                loss = loss_fn(out, discriminator, args)
                                loss, _ = loss if args["mode"] in ADVERSARIAL_MODES else (loss, None)
                                epoch_metrics["validation/loss"].append(loss.item())

                            metrics["validation/loss/mean"].append(torch.tensor(epoch_metrics["validation/loss"]).mean())

                    __log_epoch(i_epoch, epoch_metrics, metrics, validation_dataloader is not None)

                    if validation_dataloader is not None:
                        if metrics['validation/loss/mean'][-1] < metrics["best_loss"]*0.98:
                            save("best")
                            metrics["best_loss"] = metrics['validation/loss/mean'][-1]

                except KeyboardInterrupt:
                    save("crash")
                    sys.exit(0)

            save("final")

        return wrapper
    return decorator