import typing
from dataclasses import dataclass

import torch

@dataclass
class VAEOutput():
    mu: typing.Optional[torch.Tensor]=None
    logvar: typing.Optional[torch.Tensor]=None
    latent: typing.Optional[torch.Tensor]=None
    recon: typing.Optional[torch.Tensor]=None
    loss: typing.Optional[torch.Tensor]=None

@dataclass
class BVREncoderOutput:
    hidden: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor

@dataclass
class BVRDecoderOutput:
    hidden: torch.Tensor
    logits: torch.Tensor

@dataclass
class BVROutput(VAEOutput):
    logits: typing.Optional[torch.Tensor]=None
