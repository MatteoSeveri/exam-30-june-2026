"""Policy euristica avanzata basata su regole esplicite."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game.cards import Carta, CartaGiocata
from game.observation import Osservazione
from game.rules import punti_presa, vincitore_presa


@dataclass
class AdvancedHeuristicPolicy:
    """Policy consapevole della squadra e delle casistiche di presa."""

    name: str = "advanced_heuristic"

    def action_probabilities(self, osservazione: Osservazione) -> dict[Carta, float]:
        """Distribuisce probabilita' uniforme sulle carte migliori."""

        carte_migliori = self._carte_migliori(osservazione)
        probabilita = 1.0 / len(carte_migliori)
        return {
            carta: probabilita if carta in carte_migliori else 0.0
            for carta in osservazione.azioni_legali
        }

    def select_action(
        self,
        osservazione: Osservazione,
        rng: random.Random,
        greedy: bool = False,
    ) -> Carta:
        """Sceglie casualmente tra le carte migliori per mantenere i pari merito."""

        return rng.choice(self._carte_migliori(osservazione))

    def _carte_migliori(self, osservazione: Osservazione) -> list[Carta]:
        """Smista l'osservazione nel ramo di regole adatto alla presa."""

        azioni_legali = list(osservazione.azioni_legali)
        if not azioni_legali:
            raise ValueError("No legal actions available")

        if not osservazione.carte_sul_campo:
            return self._minime(
                azioni_legali,
                lambda carta: self._costo_apertura(osservazione, carta),
            )

        vincitore = self._vincitore_corrente(osservazione)
        if vincitore == osservazione.compagno_id:
            if self._ultimo_di_mano(osservazione):
                return self._carte_compagno_prende_ultimo(osservazione, azioni_legali)
            return self._carte_compagno_prende_non_ultimo(osservazione, azioni_legali)

        if vincitore in osservazione.avversari:
            if self._ultimo_di_mano(osservazione):
                return self._carte_avversario_prende_ultimo(osservazione, azioni_legali)
            return self._carte_avversario_prende_non_ultimo(osservazione, azioni_legali)

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_compagno_prende_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Carica punti sicuri quando la squadra non puo' essere superata."""

        carte_che_salvano_presa = [
            carta
            for carta in azioni_legali
            if self._team_prende_dopo_carta(osservazione, carta)
        ]
        candidate = carte_che_salvano_presa or azioni_legali

        carichi_non_briscola = [
            carta
            for carta in candidate
            if self._carico(carta) and not self._briscola(osservazione, carta)
        ]
        if carichi_non_briscola:
            return self._massime(
                carichi_non_briscola,
                lambda carta: (carta.punti, -carta.forza),
            )

        non_briscola = [
            carta for carta in candidate if not self._briscola(osservazione, carta)
        ]
        if non_briscola:
            return self._minime(
                non_briscola,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        return self._minime(
            candidate,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_compagno_prende_non_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Lascia il compagno in testa spendendo il meno possibile."""

        carte_che_lasciano_compagno = [
            carta
            for carta in azioni_legali
            if self._vincitore_dopo_carta(osservazione, carta)
            == osservazione.compagno_id
        ]
        if carte_che_lasciano_compagno:
            return self._minime(
                carte_che_lasciano_compagno,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_avversario_prende_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Se ultimo a giocare, cerca punti usando se possibile carte non briscola."""

        carte_che_prendono = self._carte_che_prendono(osservazione, azioni_legali)
        if not carte_che_prendono:
            return self._minime(
                azioni_legali,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        carichi_non_briscola = [
            carta
            for carta in carte_che_prendono
            if self._carico(carta) and not self._briscola(osservazione, carta)
        ]
        if carichi_non_briscola:
            return self._massime(
                carichi_non_briscola,
                lambda carta: (carta.punti, -carta.forza),
            )

        if self._presa_ricca(osservazione):
            return self._minime(
                carte_che_prendono,
                lambda carta: self._costo_presa_ultimo(osservazione, carta),
            )

        non_briscola = [
            carta
            for carta in carte_che_prendono
            if not self._briscola(osservazione, carta)
        ]
        if non_briscola:
            return self._minime(
                non_briscola,
                lambda carta: self._costo_presa_ultimo(osservazione, carta),
            )

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_avversario_prende_non_ultimo(
        self,
        osservazione: Osservazione,
        azioni_legali: list[Carta],
    ) -> list[Carta]:
        """Prende con cautela perche' un avversario puo' ancora superare."""

        carte_che_prendono = self._carte_che_prendono(osservazione, azioni_legali)
        non_briscola_non_carico = [
            carta
            for carta in carte_che_prendono
            if not self._briscola(osservazione, carta) and not self._carico(carta)
        ]

        if self._presa_ricca(osservazione):
            briscole = [
                carta
                for carta in carte_che_prendono
                if self._briscola(osservazione, carta)
            ]
            if briscole:
                return self._minime(briscole, self._costo_briscola_bassa)

            if non_briscola_non_carico:
                return self._minime(non_briscola_non_carico, self._costo_presa_povera)

            return self._minime(
                azioni_legali,
                lambda carta: self._costo_danno(osservazione, carta),
            )

        if non_briscola_non_carico:
            return self._minime(non_briscola_non_carico, self._costo_presa_povera)

        return self._minime(
            azioni_legali,
            lambda carta: self._costo_danno(osservazione, carta),
        )

    def _carte_che_prendono(
        self,
        osservazione: Osservazione,
        carte: list[Carta],
    ) -> list[Carta]:
        """Filtra le carte che rendono vincente il giocatore corrente."""

        return [
            carta
            for carta in carte
            if self._vincitore_dopo_carta(osservazione, carta)
            == osservazione.giocatore_id
        ]

    def _team_prende_dopo_carta(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> bool:
        """Verifica se dopo la carta la presa resta alla squadra corrente."""

        return self._vincitore_dopo_carta(osservazione, carta) in (
            osservazione.giocatore_id,
            osservazione.compagno_id,
        )

    def _vincitore_corrente(self, osservazione: Osservazione) -> int:
        """Calcola il vincitore provvisorio prima della carta corrente."""

        vincitore = vincitore_presa(
            osservazione.carte_sul_campo,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _vincitore_dopo_carta(self, osservazione: Osservazione, carta: Carta) -> int:
        """Calcola il vincitore provvisorio dopo una carta candidata."""

        presa_candidata = tuple(osservazione.carte_sul_campo) + (
            CartaGiocata(giocatore_id=osservazione.giocatore_id, carta=carta),
        )
        vincitore = vincitore_presa(
            presa_candidata,
            seme_briscola=osservazione.seme_briscola,
        )
        return vincitore.giocatore_id

    def _ultimo_di_mano(self, osservazione: Osservazione) -> bool:
        """Riconosce quando nessun altro giochera' dopo questa carta."""

        return osservazione.posizione_nella_presa == 3

    def _presa_ricca(self, osservazione: Osservazione) -> bool:
        """Considera ricca una presa con almeno dieci punti sul campo."""

        return punti_presa(osservazione.carte_sul_campo) >= 10

    def _briscola(self, osservazione: Osservazione, carta: Carta) -> bool:
        """Verifica se la carta appartiene al seme di briscola."""

        return carta.seme == osservazione.seme_briscola

    def _carico(self, carta: Carta) -> bool:
        """Riconosce asso e tre tramite il valore in punti."""

        return carta.punti >= 10

    def _costo_apertura(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[int, bool, int]:
        """Ordina gli scarti di apertura: pochi punti, non briscola, bassa forza."""

        return (
            carta.punti,
            self._briscola(osservazione, carta),
            carta.forza,
        )

    def _costo_danno(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[int, bool, bool, int]:
        """Ordina il danno: pochi punti, non carico, non briscola, bassa forza."""

        return (
            carta.punti,
            self._carico(carta),
            self._briscola(osservazione, carta),
            carta.forza,
        )

    def _costo_presa_ultimo(
        self,
        osservazione: Osservazione,
        carta: Carta,
    ) -> tuple[bool, int, int]:
        """Da ultimo preferisce prendere senza briscola e con costo basso."""

        return (
            self._briscola(osservazione, carta),
            carta.punti,
            carta.forza,
        )

    def _costo_briscola_bassa(self, carta: Carta) -> tuple[int, int]:
        """Sceglie la briscola meno cara per proteggere una presa ricca."""

        return (carta.punti, carta.forza)

    def _costo_presa_povera(self, carta: Carta) -> tuple[int, int]:
        """Sceglie la presa povera meno cara tra carte gia' ammissibili."""

        return (carta.punti, carta.forza)

    def _minime(self, carte: list[Carta], key) -> list[Carta]:
        """Restituisce tutte le carte migliori secondo un ordine di priorita'."""

        valore_minimo = min(key(carta) for carta in carte)
        return [carta for carta in carte if key(carta) == valore_minimo]

    def _massime(self, carte: list[Carta], key) -> list[Carta]:
        """Restituisce tutte le carte peggiori secondo un ordine di priorita'."""

        valore_massimo = max(key(carta) for carta in carte)
        return [carta for carta in carte if key(carta) == valore_massimo]
