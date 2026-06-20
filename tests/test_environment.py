import random
import unittest

from game.cards import Carta, crea_mazzo
from game.environment import Ambiente
from game.rules import ordine_giocatori_da


def gioca_partita_deterministica(seed: int, primo_giocatore_id: int = 0) -> Ambiente:
    ambiente = Ambiente(seed=seed, primo_giocatore_id=primo_giocatore_id)
    while not ambiente.finita:
        ambiente.gioca(ambiente.azioni_legali()[0])
        ambiente.verifica_integrita_stato()
    return ambiente


def prepara_presa_vinta_da_tre(ambiente: Ambiente, mazzo: list[Carta]) -> None:
    ambiente.mani = [
        [Carta("coppe", "due")],
        [Carta("coppe", "quattro")],
        [Carta("coppe", "cinque")],
        [Carta("denari", "due")],
    ]
    ambiente.mazzo = list(mazzo)
    ambiente.seme_briscola = "denari"
    ambiente.briscola_esposta = Carta("denari", "asso")
    ambiente.proprietario_briscola_esposta = None
    ambiente.carte_sul_campo = []
    ambiente.carte_giocate = []
    ambiente.vincitori_prese = []
    ambiente.punteggi = {"pari": 0, "dispari": 0}
    ambiente.primo_giocatore_presa = 0
    ambiente.giocatore_corrente = 0
    ambiente.indice_presa = 0
    ambiente.finita = False


class TestAmbiente(unittest.TestCase):
    def test_partita_completa_rispetta_invarianti_del_motore(self):
        ambiente = gioca_partita_deterministica(seed=7)

        self.assertEqual(ambiente.indice_presa, 10)
        self.assertEqual(sum(ambiente.punteggi.values()), 120)
        self.assertEqual(len(ambiente.carte_giocate), 40)
        self.assertEqual(ambiente.carte_giocate_per_giocatore(), [10, 10, 10, 10])

    def test_seed_reproducibility(self):
        prima = gioca_partita_deterministica(seed=42, primo_giocatore_id=1)
        seconda = gioca_partita_deterministica(seed=42, primo_giocatore_id=1)

        prime_giocate = [
            (giocata.giocatore_id, giocata.carta.id) for giocata in prima.carte_giocate
        ]
        seconde_giocate = [
            (giocata.giocatore_id, giocata.carta.id) for giocata in seconda.carte_giocate
        ]

        self.assertEqual(prime_giocate, seconde_giocate)
        self.assertEqual(prima.punteggi, seconda.punteggi)

    def test_distribuzione_iniziale_parte_da_primo_giocatore(self):
        seed = 11
        primo_giocatore_id = 1
        ambiente = Ambiente(seed=seed, primo_giocatore_id=primo_giocatore_id)

        mazzo_atteso = crea_mazzo()
        random.Random(seed).shuffle(mazzo_atteso)
        mani_attese = [[] for _ in range(4)]
        for _ in range(3):
            for giocatore_id in ordine_giocatori_da(primo_giocatore_id):
                mani_attese[giocatore_id].append(mazzo_atteso.pop(0))
        briscola_attesa = mazzo_atteso.pop(0)

        self.assertEqual(ambiente.mani, mani_attese)
        self.assertEqual(ambiente.giocatore_corrente, primo_giocatore_id)
        self.assertEqual(ambiente.primo_giocatore_presa, primo_giocatore_id)
        self.assertEqual(ambiente.briscola_esposta, briscola_attesa)
        self.assertEqual(ambiente.mazzo[-1], briscola_attesa)
        self.assertIsNone(ambiente.proprietario_briscola_esposta)

    def test_esito_mossa_non_completa_avanza_il_giocatore(self):
        ambiente = Ambiente(seed=3, primo_giocatore_id=0)
        carta = ambiente.azioni_legali()[0]

        esito = ambiente.gioca(carta)

        self.assertFalse(esito.presa_completata)
        self.assertIsNone(esito.vincitore_presa)
        self.assertEqual(esito.punti_presa, 0)
        self.assertEqual(esito.carte_presa_completata, ())
        self.assertEqual(esito.carte_pescate, ())
        self.assertEqual(esito.prossimo_giocatore, 1)
        self.assertEqual([evento.tipo for evento in esito.eventi_pubblici], ["carta_giocata"])

    def test_pescata_dopo_presa_parte_dal_vincitore(self):
        ambiente = Ambiente(seed=0, primo_giocatore_id=0)
        mazzo = [
            Carta("bastoni", "due"),
            Carta("bastoni", "quattro"),
            Carta("bastoni", "cinque"),
            Carta("bastoni", "sei"),
        ]
        prepara_presa_vinta_da_tre(ambiente, mazzo=mazzo)

        for _ in range(3):
            ambiente.gioca(ambiente.azioni_legali()[0])
        esito = ambiente.gioca(ambiente.azioni_legali()[0])

        self.assertTrue(esito.presa_completata)
        self.assertEqual(esito.vincitore_presa, 3)
        self.assertEqual(esito.prossimo_giocatore, 3)
        self.assertEqual([pescata.giocatore_id for pescata in esito.carte_pescate], [3, 0, 1, 2])
        self.assertEqual(
            [mano[0] for mano in ambiente.mani],
            [mazzo[1], mazzo[2], mazzo[3], mazzo[0]],
        )
        self.assertTrue(all(pescata.carta_visibile is None for pescata in esito.carte_pescate))
        self.assertEqual(ambiente.carte_sul_campo, [])

    def test_proprietario_briscola_esposta_viene_valorizzato_quando_pescata(self):
        ambiente = Ambiente(seed=0, primo_giocatore_id=0)
        briscola_esposta = Carta("denari", "asso")
        mazzo = [
            Carta("bastoni", "due"),
            Carta("bastoni", "quattro"),
            Carta("bastoni", "cinque"),
            briscola_esposta,
        ]
        prepara_presa_vinta_da_tre(ambiente, mazzo=mazzo)
        ambiente.briscola_esposta = briscola_esposta

        for _ in range(3):
            ambiente.gioca(ambiente.azioni_legali()[0])
        esito = ambiente.gioca(ambiente.azioni_legali()[0])

        self.assertEqual(ambiente.proprietario_briscola_esposta, 2)
        self.assertEqual(esito.carte_pescate[-1].giocatore_id, 2)
        self.assertEqual(esito.carte_pescate[-1].carta_visibile, briscola_esposta)
        self.assertIn("briscola_esposta_pescata", [evento.tipo for evento in esito.eventi_pubblici])

    def test_osservazione_non_espone_stato_nascosto(self):
        ambiente = Ambiente(seed=5)

        osservazione = ambiente.osserva(0)

        self.assertFalse(hasattr(osservazione, "mazzo"))
        self.assertFalse(hasattr(osservazione, "mani"))
        self.assertFalse(hasattr(osservazione, "mano_avversari"))
        self.assertFalse(hasattr(osservazione, "opponent_hands"))

    def test_mano_compagno_visibile_solo_quando_mazzo_vuoto(self):
        ambiente = Ambiente(seed=5)

        osservazione_iniziale = ambiente.osserva(0)
        self.assertFalse(osservazione_iniziale.mano_compagno_visibile)
        self.assertEqual(osservazione_iniziale.mano_compagno, ())

        ambiente.mazzo = []
        osservazione_finale = ambiente.osserva(0)
        self.assertTrue(osservazione_finale.mano_compagno_visibile)
        self.assertEqual(osservazione_finale.mano_compagno, tuple(ambiente.mani[2]))


if __name__ == "__main__":
    unittest.main()
