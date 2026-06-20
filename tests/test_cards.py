import unittest

from game.cards import Carta, RANGHI, SEMI, carta_da_id, crea_mazzo, punti_totali_mazzo


class TestCarte(unittest.TestCase):
    def test_mazzo_ha_40_carte_uniche(self):
        mazzo = crea_mazzo()

        self.assertEqual(len(mazzo), 40)
        self.assertEqual(len(set(mazzo)), 40)

    def test_punti_totali_mazzo_sono_120(self):
        self.assertEqual(punti_totali_mazzo(), 120)

    def test_semi_e_ranghi_sono_in_italiano(self):
        self.assertEqual(SEMI, ("coppe", "denari", "bastoni", "spade"))
        self.assertEqual(
            RANGHI,
            (
                "asso",
                "tre",
                "re",
                "cavallo",
                "fante",
                "sette",
                "sei",
                "cinque",
                "quattro",
                "due",
            ),
        )

    def test_carta_da_id_ricostruisce_la_carta(self):
        carta = Carta(seme="coppe", rango="asso")

        self.assertEqual(carta_da_id(carta.id), carta)

    def test_forza_rispetta_ordine_briscola(self):
        self.assertGreater(Carta("coppe", "asso").forza, Carta("coppe", "tre").forza)
        self.assertGreater(Carta("coppe", "tre").forza, Carta("coppe", "re").forza)
        self.assertGreater(Carta("coppe", "fante").forza, Carta("coppe", "sette").forza)
        self.assertGreater(Carta("coppe", "quattro").forza, Carta("coppe", "due").forza)


if __name__ == "__main__":
    unittest.main()
