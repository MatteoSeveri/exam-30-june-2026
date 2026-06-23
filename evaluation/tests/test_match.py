from __future__ import annotations

import random
import unittest
from dataclasses import dataclass, field

from game.cards import Carta
from game.observation import Osservazione
from game.rules import SQUADRA_DISPARI, SQUADRA_PARI
from policy import RandomPolicy
from evaluation.match import play_match


@dataclass
class TrackingPolicy:
    name: str
    seen_players: list[int] = field(default_factory=list)
    greedy_values: list[bool] = field(default_factory=list)
    selected_cards: list[Carta] = field(default_factory=list)

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        probability = 1.0 / len(osservazione.azioni_legali)
        return {carta: probability for carta in osservazione.azioni_legali}

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        self.seen_players.append(osservazione.giocatore_id)
        self.greedy_values.append(greedy)
        carta = osservazione.azioni_legali[0]
        self.selected_cards.append(carta)
        return carta


def random_policies() -> dict[int, RandomPolicy]:
    return {giocatore_id: RandomPolicy() for giocatore_id in range(4)}


class TestPlayMatch(unittest.TestCase):
    def test_play_match_produce_risultato_completo(self):
        # An evaluation match must complete a full game without training.
        result = play_match(
            policies_by_player=random_policies(),
            seed_ambiente=10,
            seed_policy=20,
            primo_giocatore_id=0,
            greedy=True,
        )

        self.assertEqual(sum(result.punteggi_finali.values()), 120)
        self.assertEqual(
            result.margine_squadra_pari,
            result.punteggi_finali[SQUADRA_PARI]
            - result.punteggi_finali[SQUADRA_DISPARI],
        )
        if result.margine_squadra_pari > 0:
            self.assertEqual(result.squadra_vincitrice, SQUADRA_PARI)
        elif result.margine_squadra_pari < 0:
            self.assertEqual(result.squadra_vincitrice, SQUADRA_DISPARI)
        else:
            self.assertIsNone(result.squadra_vincitrice)
        self.assertEqual(result.seed_ambiente, 10)
        self.assertEqual(result.seed_policy, 20)
        self.assertEqual(result.primo_giocatore_id, 0)

    def test_stessi_seed_producono_stesso_risultato(self):
        # Environment and policy tie-breaks are reproducible when seeds match.
        first = play_match(
            policies_by_player=random_policies(),
            seed_ambiente=11,
            seed_policy=21,
            primo_giocatore_id=1,
            greedy=True,
        )
        second = play_match(
            policies_by_player=random_policies(),
            seed_ambiente=11,
            seed_policy=21,
            primo_giocatore_id=1,
            greedy=True,
        )

        self.assertEqual(first, second)

    def test_policy_per_giocatore_e_greedy_forzato(self):
        # Each policy sees only its own player and receives greedy=True.
        policies = {
            giocatore_id: TrackingPolicy(name=f"policy_{giocatore_id}")
            for giocatore_id in range(4)
        }

        play_match(
            policies_by_player=policies,
            seed_ambiente=12,
            seed_policy=22,
            primo_giocatore_id=2,
            greedy=True,
        )

        for giocatore_id, policy in policies.items():
            self.assertEqual(set(policy.seen_players), {giocatore_id})
            self.assertEqual(len(policy.selected_cards), 10)
            self.assertEqual(set(policy.greedy_values), {True})

    def test_policies_by_player_deve_contenere_quattro_giocatori(self):
        # The sandbox does not infer missing or extra seats.
        with self.assertRaises(ValueError):
            play_match(
                policies_by_player={0: RandomPolicy(), 1: RandomPolicy()},
                seed_ambiente=13,
                seed_policy=23,
                primo_giocatore_id=0,
                greedy=True,
            )

        with self.assertRaises(ValueError):
            play_match(
                policies_by_player={
                    0: RandomPolicy(),
                    1: RandomPolicy(),
                    2: RandomPolicy(),
                    3: RandomPolicy(),
                    4: RandomPolicy(),
                },
                seed_ambiente=13,
                seed_policy=23,
                primo_giocatore_id=0,
                greedy=True,
            )

    def test_primo_giocatore_non_valido_solleva_value_error(self):
        # Rotation must remain explicit and valid.
        with self.assertRaises(ValueError):
            play_match(
                policies_by_player=random_policies(),
                seed_ambiente=14,
                seed_policy=24,
                primo_giocatore_id=4,
                greedy=True,
            )

    def test_play_match_passa_anche_greedy_false_quando_richiesto(self):
        # match.py runs the received protocol without forcing greedy mode itself.
        policies = {
            giocatore_id: TrackingPolicy(name=f"policy_{giocatore_id}")
            for giocatore_id in range(4)
        }

        play_match(
            policies_by_player=policies,
            seed_ambiente=15,
            seed_policy=25,
            primo_giocatore_id=0,
            greedy=False,
        )

        for policy in policies.values():
            self.assertEqual(set(policy.greedy_values), {False})


if __name__ == "__main__":
    unittest.main()
