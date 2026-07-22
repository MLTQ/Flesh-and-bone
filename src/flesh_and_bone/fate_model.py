"""Shared learned scorer for coarse anatomical-region commitments."""

from dataclasses import dataclass

import torch
from torch import nn


class RegionalFateMLP(nn.Module):
    """Score every region with one permutation-equivariant shared MLP."""

    def __init__(self, hidden_channels=24):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, shortage_fraction, available, distance_in_spacings):
        features = torch.stack([
            shortage_fraction,
            available,
            distance_in_spacings / 30.0,
        ], dim=-1)
        return self.score(features).squeeze(-1)


def oracle_fate_scores(shortage_fraction, available, distance_in_spacings,
                       distance_weight=0.04):
    """Continuous training target equivalent to the frozen H2 selector."""
    return (
        shortage_fraction
        - distance_weight * distance_in_spacings
        - 4.0 * (1 - available)
    )


@dataclass(frozen=True)
class FateTrainingReport:
    train_loss: float
    holdout_loss: float
    holdout_agreement: float
    steps: int
    examples: int


def _synthetic_batch(batch_size, region_count, generator, device, dtype):
    shortage = torch.rand(
        batch_size, region_count, generator=generator,
        device=device, dtype=dtype,
    )
    absent = torch.rand(
        batch_size, region_count, generator=generator,
        device=device, dtype=dtype,
    ) < 0.24
    shortage = shortage.masked_fill(absent, 0)
    forced = torch.randint(
        region_count, (batch_size,), generator=generator, device=device
    )
    shortage[torch.arange(batch_size, device=device), forced] = torch.rand(
        batch_size, generator=generator, device=device, dtype=dtype
    ).clamp(min=0.05)
    available = (shortage > 0).to(dtype)
    distance = 30 * torch.rand(
        batch_size, region_count, generator=generator,
        device=device, dtype=dtype,
    )
    return shortage, available, distance


def train_fate_model(region_count, seed=7, device=None, dtype=torch.float32,
                     steps=700, batch_size=512, learning_rate=3e-3):
    """Distill the capacity-and-distance oracle on randomized regional states."""
    torch.manual_seed(int(seed))
    model = RegionalFateMLP().to(device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    model.train()
    for _ in range(int(steps)):
        shortage, available, distance = _synthetic_batch(
            batch_size, region_count, generator, device, dtype
        )
        target = oracle_fate_scores(shortage, available, distance)
        prediction = model(shortage, available, distance)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    model.eval()
    with torch.no_grad():
        shortage, available, distance = _synthetic_batch(
            8192, region_count, generator, device, dtype
        )
        target = oracle_fate_scores(shortage, available, distance)
        prediction = model(shortage, available, distance)
        holdout_loss = float((prediction - target).square().mean().item())
        agreement = float(
            (prediction.argmax(dim=1) == target.argmax(dim=1))
            .float().mean().item()
        )
    report = FateTrainingReport(
        train_loss=final_loss,
        holdout_loss=holdout_loss,
        holdout_agreement=agreement,
        steps=int(steps),
        examples=int(steps) * int(batch_size),
    )
    return model, report


class LearnedFateSelector:
    """Adapt a frozen fate model to the scalar selector used by dynamics."""

    def __init__(self, model, spacing, expose_shortage=True):
        self.model = model
        self.spacing = float(spacing)
        self.expose_shortage = bool(expose_shortage)

    def __call__(self, shortage, target_count, distance):
        shortage_fraction = shortage / target_count.clamp(min=1)
        available = (shortage > 0).to(distance.dtype)
        if not self.expose_shortage:
            shortage_fraction = torch.ones_like(shortage_fraction)
            available = torch.ones_like(available)
        with torch.no_grad():
            scores = self.model(
                shortage_fraction[None],
                available[None],
                (distance / self.spacing)[None],
            )[0]
        return int(scores.argmax().item())
