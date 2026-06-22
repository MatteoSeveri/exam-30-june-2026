"""Single-match evaluation sandbox for frozen Briscola policies."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.environment import Ambiente
from game.rules import (
    NUMERO_GIOCATORI,
    SQUADRA_DISPARI,
    SQUADRA_PARI,
    valida_giocatore_id,
)
from policy import Policy


@dataclass(frozen=True)
class MatchResult:
    """Result of one complete evaluation match."""

    punteggi_finali: dict[str, int]
    squadra_vincitrice: str | None
    margine_squadra_pari: int
    seed_ambiente: int
    seed_policy: int
    primo_giocatore_id: int


def play_match(
    *,
    policies_by_player: dict[int, Policy],
    seed_ambiente: int,
    seed_policy: int,
    primo_giocatore_id: int,
    greedy: bool,
) -> MatchResult:
    """Play one frozen match with the requested action-selection mode."""

    _valida_policies_by_player(policies_by_player)
    valida_giocatore_id(primo_giocatore_id)

    ambiente = Ambiente(
        seed=seed_ambiente,
        primo_giocatore_id=primo_giocatore_id,
    )
    rng_policy = random.Random(seed_policy)

    while not ambiente.finita:
        giocatore_id = ambiente.giocatore_corrente
        osservazione = ambiente.osserva(giocatore_id)
        policy = policies_by_player[giocatore_id]
        azione = policy.select_action(osservazione, rng_policy, greedy=greedy)
        ambiente.gioca(azione)

    ambiente.verifica_integrita_stato()
    punteggi_finali = dict(ambiente.punteggi)

    return MatchResult(
        punteggi_finali=punteggi_finali,
        squadra_vincitrice=ambiente.squadra_vincitrice(),
        margine_squadra_pari=(
            punteggi_finali[SQUADRA_PARI] - punteggi_finali[SQUADRA_DISPARI]
        ),
        seed_ambiente=seed_ambiente,
        seed_policy=seed_policy,
        primo_giocatore_id=primo_giocatore_id,
    )


def _valida_policies_by_player(policies_by_player: dict[int, Policy]) -> None:
    expected_players = set(range(NUMERO_GIOCATORI))
    actual_players = set(policies_by_player)
    if actual_players != expected_players:
        raise ValueError(
            "policies_by_player deve contenere esattamente i giocatori "
            f"{sorted(expected_players)}, ottenuto {sorted(actual_players)}"
        )
    for giocatore_id in policies_by_player:
        valida_giocatore_id(giocatore_id)
