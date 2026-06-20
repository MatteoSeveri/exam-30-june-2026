from __future__ import annotations

import unittest

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione


def osservazione(
    *,
    mano_compagno_visibile: bool = False,
    mano_compagno: tuple[Carta, ...] = (),
    proprietario_briscola_esposta: int | None = None,
    carte_sul_campo: tuple[CartaGiocata, ...] = (),
    carte_giocate: tuple[CartaGiocata, ...] = (),
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=(Carta("coppe", "asso"), Carta("bastoni", "due")),
        mano_compagno_visibile=mano_compagno_visibile,
        mano_compagno=mano_compagno,
        seme_briscola="denari",
        briscola_esposta=Carta("denari", "tre"),
        proprietario_briscola_esposta=proprietario_briscola_esposta,
        carte_sul_campo=carte_sul_campo,
        carte_giocate=carte_giocate,
        vincitori_prese=(0, 3),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=22,
        punteggio_avversari=18,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=24,
        indice_presa=2,
        posizione_nella_presa=len(carte_sul_campo),
    )


class TestOsservazione(unittest.TestCase):
    def test_azioni_legali_coincidono_con_la_mano(self):
        obs = osservazione()

        self.assertEqual(obs.azioni_legali, obs.mano)

    def test_mano_compagno_puo_essere_nascosta(self):
        obs = osservazione(mano_compagno_visibile=False, mano_compagno=())

        self.assertFalse(obs.mano_compagno_visibile)
        self.assertEqual(obs.mano_compagno, ())

    def test_mano_compagno_puo_essere_visibile(self):
        mano_compagno = (Carta("spade", "re"), Carta("coppe", "sette"))

        obs = osservazione(
            mano_compagno_visibile=True,
            mano_compagno=mano_compagno,
        )

        self.assertTrue(obs.mano_compagno_visibile)
        self.assertEqual(obs.mano_compagno, mano_compagno)

    def test_briscola_esposta_senza_proprietario(self):
        obs = osservazione(proprietario_briscola_esposta=None)

        self.assertEqual(obs.briscola_esposta, Carta("denari", "tre"))
        self.assertIsNone(obs.proprietario_briscola_esposta)

    def test_briscola_esposta_pescata_da_un_giocatore(self):
        obs = osservazione(proprietario_briscola_esposta=3)

        self.assertEqual(obs.briscola_esposta, Carta("denari", "tre"))
        self.assertEqual(obs.proprietario_briscola_esposta, 3)

    def test_carte_sul_campo_e_carte_giocate_restano_separate(self):
        carta_sul_campo = CartaGiocata(giocatore_id=0, carta=Carta("coppe", "asso"))
        carta_giocata = CartaGiocata(giocatore_id=3, carta=Carta("spade", "tre"))

        obs = osservazione(
            carte_sul_campo=(carta_sul_campo,),
            carte_giocate=(carta_giocata,),
        )

        self.assertEqual(obs.carte_sul_campo, (carta_sul_campo,))
        self.assertEqual(obs.carte_giocate, (carta_giocata,))
        self.assertEqual(obs.posizione_nella_presa, 1)

    def test_non_include_differenza_punteggio(self):
        obs = osservazione()

        self.assertFalse(hasattr(obs, "differenza_punteggio"))

    def test_proprieta_diagnostiche_derivate(self):
        carte_giocate = (
            CartaGiocata(giocatore_id=0, carta=Carta("coppe", "asso")),
            CartaGiocata(giocatore_id=1, carta=Carta("bastoni", "tre")),
        )

        obs = osservazione(carte_giocate=carte_giocate)

        self.assertEqual(obs.numero_carte_giocate, 2)
        self.assertEqual(obs.avversari, (1, 3))
        self.assertEqual(obs.punteggi, {"pari": 22, "dispari": 18})


if __name__ == "__main__":
    unittest.main()
