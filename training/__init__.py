"""Training utilities for Briscola reinforcement learning."""

from .episode import (
    MOSSE_PER_GIOCATORE,
    MOSSE_TOTALI_PARTITA,
    EpisodeResult,
    TrajectoryStep,
    collect_episode,
)
from .rewards import (
    PUNTI_TOTALI_PARTITA,
    REWARD_MODES,
    RewardConfig,
    calcola_margine,
    calcola_segno,
    normalizza_margine,
    reward_finale,
    reward_presa,
)

__all__ = [
    "MOSSE_PER_GIOCATORE",
    "MOSSE_TOTALI_PARTITA",
    "PUNTI_TOTALI_PARTITA",
    "REWARD_MODES",
    "EpisodeResult",
    "RewardConfig",
    "TrajectoryStep",
    "calcola_margine",
    "calcola_segno",
    "collect_episode",
    "normalizza_margine",
    "reward_finale",
    "reward_presa",
]
