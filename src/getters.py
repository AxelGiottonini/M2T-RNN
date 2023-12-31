import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .bvr import BVR, BVRTokenizer
from .tokenizer import __get_collate_fn__
from .dataset import Dataset
from .utils import ADVERSARIAL_MODES

def __prune_tuple(t):
    t = tuple(el for el in t if el is not None)
    return t[0] if len(t) == 1 else t

def __get_tokenizer(args):
    return BVRTokenizer.from_pretrained(args["from_tokenizer"])

def __get_collate_fn(args, tokenizer):
    return __get_collate_fn__(tokenizer, mask=args["mask"], mask_rate=args["mask_rate"], frag_coef_a=args["frag_coef_a"], frag_coef_b=args["frag_coef_b"], split=args["split"])

def get_model(args, return_optimizer=False):
    if args["from_pretrained_bert"]:
        model = BVR.from_pretrained_bert(
            args["from_pretrained_bert"],
            hidden_size=args["hidden_size"],
            num_layers=args["num_layers"],
            dropout=args["dropout"],
            latent_size=args["latent_size"],
            cls_token_id=args["cls_token_id"]
        )
    elif args["from_model"]:
        model = BVR.from_pretrained(args["from_model"])
    else:
        raise NotImplementedError()

    optimizer = None
    if return_optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args["learning_rate"], 
            betas=args["betas"], 
            eps=args["eps"], 
            weight_decay=args["weight_decay"]
        )

    discriminator = None
    d_optimizer = None
    if args["mode"] in ADVERSARIAL_MODES:
        discriminator = nn.Sequential(
            nn.Linear(model.config.vae_latent_size, args["discriminator_size"]), 
            nn.ReLU(),
            nn.Linear(args["discriminator_size"], 2), 
        )

        if return_optimizer:
            d_optimizer = torch.optim.AdamW(
                discriminator.parameters(),
                lr=args["d_learning_rate"], 
                betas=args["d_betas"], 
                eps=args["d_eps"], 
                weight_decay=args["d_weight_decay"]
            )

    return __prune_tuple((model, discriminator, optimizer, d_optimizer))

def get_dataloaders(args, shuffle=True, return_validation=True, return_tokenizer=False, return_collate_fn=False):
    tokenizer = __get_tokenizer(args)
    collate_fn = __get_collate_fn(args, tokenizer)

    training_set = Dataset(args["training_set"], args["min_length"], args["max_length"])
    training_dataloader = DataLoader(dataset=training_set, batch_size=args["local_batch_size"], shuffle=shuffle, num_workers=args["num_workers"], collate_fn=collate_fn)
    
    validation_dataloader = None
    if return_validation:
        validation_set = Dataset(args["validation_set"], args["min_length"], args["max_length"])
        validation_dataloader = DataLoader(dataset=validation_set, batch_size=args["local_batch_size"], shuffle=False, num_workers=args["num_workers"], collate_fn=collate_fn)

    return __prune_tuple((
        training_dataloader,
        validation_dataloader,
        tokenizer if return_tokenizer else None,
        collate_fn if return_collate_fn else None
    ))