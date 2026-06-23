from __future__ import annotations

import unittest

from diagnostics import (
    record_decision_log,
    records_by_trick_position,
    records_chosen_with_probability_below,
    records_for_player,
    records_for_policy,
    records_on_rich_trick,
    records_with_opponent_leading,
    records_with_partner_leading,
)
from game.rules import punti_presa, vincitore_presa
from policy import RandomPolicy


def log_random():
    return record_decision_log(
        policies_by_player={
            giocatore_id: RandomPolicy(name=f"random_{giocatore_id}")
            for giocatore_id in range(4)
        },
        seed_ambiente=100,
        seed_policy=200,
        primo_giocatore_id=0,
        greedy=True,
    )


class TestDiagnosticViews(unittest.TestCase):
    def test_records_for_player_filtra_un_giocatore(self):
        # The player view returns only that player's 10 decisions.
        log = log_random()

        records = records_for_player(log, 0)

        self.assertEqual(len(records), 10)
        self.assertTrue(all(record.giocatore_id == 0 for record in records))

    def test_records_for_policy_filtra_per_nome(self):
        # The policy-name view isolates records produced by that policy.
        log = log_random()

        records = records_for_policy(log, "random_2")

        self.assertEqual(len(records), 10)
        self.assertTrue(all(record.policy_name == "random_2" for record in records))

    def test_records_by_trick_position(self):
        # Each trick position appears 10 times in a complete game.
        log = log_random()

        for posizione in range(4):
            records = records_by_trick_position(log, posizione)

            self.assertEqual(len(records), 10)
            self.assertTrue(
                all(
                    record.osservazione.posizione_nella_presa == posizione
                    for record in records
                )
            )

    def test_records_with_partner_leading(self):
        # Each returned record has the teammate leading the current trick.
        log = log_random()

        records = records_with_partner_leading(log)

        for record in records:
            winner = vincitore_presa(
                record.osservazione.carte_sul_campo,
                seme_briscola=record.osservazione.seme_briscola,
            ).giocatore_id
            self.assertEqual(winner, record.osservazione.compagno_id)

    def test_records_with_opponent_leading(self):
        # Each returned record has an opponent leading the current trick.
        log = log_random()

        records = records_with_opponent_leading(log)

        for record in records:
            winner = vincitore_presa(
                record.osservazione.carte_sul_campo,
                seme_briscola=record.osservazione.seme_briscola,
            ).giocatore_id
            self.assertIn(winner, record.osservazione.avversari)

    def test_records_on_rich_trick(self):
        # The view includes only decisions where the table contains enough points.
        log = log_random()

        records = records_on_rich_trick(log, min_points=10)

        for record in records:
            self.assertGreaterEqual(
                punti_presa(record.osservazione.carte_sul_campo),
                10,
            )

    def test_records_chosen_with_probability_below(self):
        # With RandomPolicy, threshold 0.5 isolates choices with probability below 1/2.
        log = log_random()

        records = records_chosen_with_probability_below(log, threshold=0.5)

        self.assertTrue(records)
        for record in records:
            self.assertLess(record.action_probabilities[record.azione], 0.5)

    def test_records_for_player_rifiuta_id_non_valido(self):
        # In 4-player Briscola, valid players are 0, 1, 2, and 3.
        log = log_random()

        with self.assertRaises(ValueError):
            records_for_player(log, 4)

    def test_records_by_trick_position_rifiuta_posizione_non_valida(self):
        # Valid positions in the trick are 0, 1, 2, and 3.
        log = log_random()

        with self.assertRaises(ValueError):
            records_by_trick_position(log, 4)

    def test_records_chosen_with_probability_below_rifiuta_soglia_negativa(self):
        # A negative threshold has no meaning for probabilities between 0 and 1.
        log = log_random()

        with self.assertRaises(ValueError):
            records_chosen_with_probability_below(log, -0.1)


if __name__ == "__main__":
    unittest.main()
