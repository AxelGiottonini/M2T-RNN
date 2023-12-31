import random
import torch

from src import configure, train_loop, get_model, get_dataloaders, ADVERSARIAL_MODES

random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    args = configure()
    training_dataloader, validation_dataloader = get_dataloaders(args)

    if args["mode"] in ADVERSARIAL_MODES:
        model, discriminator, optimizer, d_optimizer = get_model(args, return_optimizer=True)
    else:
        model, optimizer = get_model(args, return_optimizer=True)
        discriminator, d_optimizer = None, None

    @train_loop(
        model = model,
        optimizer = optimizer,
        args = args,
        discriminator=discriminator,
        d_optimizer=d_optimizer,
    )
    def train(model, batch):
        out = model(batch.masked_input_ids, batch.input_ids)
        return out
    
    train(training_dataloader, validation_dataloader)