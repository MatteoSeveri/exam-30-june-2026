from __future__ import annotations

import random
import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from policy import AdvancedHeuristicPolicy


def osservazione(
    *,
    mano: tuple[Carta, ...],
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    seme_briscola: str = "denari",
    posizione_nella_presa: int | None = None,
) -> Osservazione:
    if posizione_nella_presa is None:
        posizione_nella_presa = len(carte_sul_campo)

    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola=seme_briscola,
        briscola_esposta=Carta(seme_briscola, "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_sul_campo,
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=1,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=posizione_nella_presa,
    )


class TestAdvancedHeuristicPolicy(unittest.TestCase):
    def test_apre_con_scarto_meno_costoso(self):
        # On an empty trick, it preserves points and briscola cards.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("coppe", "asso"),
                scarto,
            )
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_compagno_prende_non_ultimo_lascia_compagno_in_testa(self):
        # If an opponent still has to play, it does not load points or overtake the teammate.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("coppe", "asso"),
                Carta("denari", "due"),
                Carta("spade", "tre"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "re")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_compagno_prende_ultimo_carica_con_carico_non_briscola(self):
        # When playing last, it can load points onto the team's trick without risk.
        carico = Carta("bastoni", "asso")
        obs = osservazione(
            mano=(
                Carta("denari", "tre"),
                carico,
                Carta("spade", "due"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("coppe", "re")),
                CartaGiocata(giocatore_id=3, carta=Carta("spade", "sette")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, carico)

    def test_avversario_prende_ultimo_gioca_carico_non_briscola_che_prende(self):
        # When playing last, it can win with a non-briscola point card to secure it.
        carico_che_prende = Carta("coppe", "asso")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                carico_che_prende,
                Carta("coppe", "fante"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "re")),
                CartaGiocata(giocatore_id=2, carta=Carta("spade", "due")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "due")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, carico_che_prende)

    def test_avversario_prende_ultimo_su_presa_povera_non_spende_briscola(self):
        # On a low-value trick, it does not spend a briscola if it cannot win without one.
        scarto = Carta("bastoni", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "due"),
                Carta("spade", "sette"),
                scarto,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
                CartaGiocata(giocatore_id=2, carta=Carta("spade", "cinque")),
                CartaGiocata(giocatore_id=3, carta=Carta("bastoni", "quattro")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, scarto)

    def test_avversario_prende_non_ultimo_su_presa_ricca_spende_briscola_bassa(self):
        # On a high-value trick and not playing last, a low briscola pressures the next opponent.
        briscola_bassa = Carta("denari", "due")
        obs = osservazione(
            mano=(
                Carta("denari", "tre"),
                briscola_bassa,
                Carta("coppe", "re"),
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "asso")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, briscola_bassa)

    def test_avversario_prende_non_ultimo_su_presa_povera_prende_senza_carico_o_briscola(self):
        # On a low-value trick, it wins only if a non-point, non-briscola card is enough.
        presa_povera = Carta("coppe", "sette")
        obs = osservazione(
            mano=(
                Carta("coppe", "asso"),
                Carta("denari", "due"),
                presa_povera,
            ),
            carte_sul_campo=(
                CartaGiocata(giocatore_id=1, carta=Carta("coppe", "sei")),
            ),
        )
        policy = AdvancedHeuristicPolicy()

        action = policy.select_action(obs, rng=random.Random(0))

        self.assertEqual(action, presa_povera)

    def test_probabilita_divise_tra_carte_migliori_equivalenti(self):
        # Equivalent cards both remain available for stochastic selection.
        primo_scarto = Carta("coppe", "due")
        secondo_scarto = Carta("bastoni", "due")
        briscola = Carta("denari", "due")
        obs = osservazione(mano=(primo_scarto, secondo_scarto, briscola))
        policy = AdvancedHeuristicPolicy()

        probabilities = policy.action_probabilities(obs)

        self.assertAlmostEqual(probabilities[primo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[secondo_scarto], 0.5)
        self.assertAlmostEqual(probabilities[briscola], 0.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mano_vuota_solleva_value_error(self):
        # A policy cannot choose if there are no legal actions.
        obs = osservazione(mano=())
        policy = AdvancedHeuristicPolicy()

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
