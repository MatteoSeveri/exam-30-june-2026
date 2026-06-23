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
        # La vista per player restituisce solo le 10 decisioni di quel giocatore.
        log = log_random()

        records = records_for_player(log, 0)

        self.assertEqual(len(records), 10)
        self.assertTrue(all(record.giocatore_id == 0 for record in records))

    def test_records_for_policy_filtra_per_nome(self):
        # La vista per policy name isola i record prodotti da quella policy.
        log = log_random()

        records = records_for_policy(log, "random_2")

        self.assertEqual(len(records), 10)
        self.assertTrue(all(record.policy_name == "random_2" for record in records))

    def test_records_by_trick_position(self):
        # Ogni posizione nella presa compare 10 volte in una partita completa.
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
        # Ogni record restituito ha il compagno in testa nella presa corrente.
        log = log_random()

        records = records_with_partner_leading(log)

        for record in records:
            winner = vincitore_presa(
                record.osservazione.carte_sul_campo,
                seme_briscola=record.osservazione.seme_briscola,
            ).giocatore_id
            self.assertEqual(winner, record.osservazione.compagno_id)

    def test_records_with_opponent_leading(self):
        # Ogni record restituito ha un avversario in testa nella presa corrente.
        log = log_random()

        records = records_with_opponent_leading(log)

        for record in records:
            winner = vincitore_presa(
                record.osservazione.carte_sul_campo,
                seme_briscola=record.osservazione.seme_briscola,
            ).giocatore_id
            self.assertIn(winner, record.osservazione.avversari)

    def test_records_on_rich_trick(self):
        # La vista prende solo decisioni dove il campo contiene abbastanza punti.
        log = log_random()

        records = records_on_rich_trick(log, min_points=10)

        for record in records:
            self.assertGreaterEqual(
                punti_presa(record.osservazione.carte_sul_campo),
                10,
            )

    def test_records_chosen_with_probability_below(self):
        # Con RandomPolicy, soglia 0.5 isola scelte con probabilita' sotto 1/2.
        log = log_random()

        records = records_chosen_with_probability_below(log, threshold=0.5)

        self.assertTrue(records)
        for record in records:
            self.assertLess(record.action_probabilities[record.azione], 0.5)

    def test_records_for_player_rifiuta_id_non_valido(self):
        # In Briscola a 4 i giocatori validi sono 0, 1, 2 e 3.
        log = log_random()

        with self.assertRaises(ValueError):
            records_for_player(log, 4)

    def test_records_by_trick_position_rifiuta_posizione_non_valida(self):
        # Le posizioni valide nella presa sono 0, 1, 2 e 3.
        log = log_random()

        with self.assertRaises(ValueError):
            records_by_trick_position(log, 4)

    def test_records_chosen_with_probability_below_rifiuta_soglia_negativa(self):
        # Una soglia negativa non ha significato per probabilita' tra 0 e 1.
        log = log_random()

        with self.assertRaises(ValueError):
            records_chosen_with_probability_below(log, -0.1)


if __name__ == "__main__":
    unittest.main()
