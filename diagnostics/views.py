"""Readable views over diagnostic decision logs."""

from __future__ import annotations

from game.rules import punti_presa, valida_giocatore_id, vincitore_presa

from .decision_log import DecisionLog, DecisionRecord


def records_for_player(
    log: DecisionLog,
    giocatore_id: int,
) -> tuple[DecisionRecord, ...]:
    """Return decisions made by one player."""

    valida_giocatore_id(giocatore_id)
    return tuple(
        record for record in log.records if record.giocatore_id == giocatore_id
    )


def records_for_policy(
    log: DecisionLog,
    policy_name: str,
) -> tuple[DecisionRecord, ...]:
    """Return decisions made by policies with the given name."""

    return tuple(
        record for record in log.records if record.policy_name == policy_name
    )


def records_with_partner_leading(log: DecisionLog) -> tuple[DecisionRecord, ...]:
    """Return decisions where the partner is currently winning the trick."""

    return tuple(
        record for record in log.records if _partner_leading(record)
    )


def records_with_opponent_leading(log: DecisionLog) -> tuple[DecisionRecord, ...]:
    """Return decisions where an opponent is currently winning the trick."""

    return tuple(
        record for record in log.records if _opponent_leading(record)
    )


def records_on_rich_trick(
    log: DecisionLog,
    min_points: int = 10,
) -> tuple[DecisionRecord, ...]:
    """Return decisions where the current trick already contains many points."""

    return tuple(
        record
        for record in log.records
        if punti_presa(record.osservazione.carte_sul_campo) >= min_points
    )


def records_by_trick_position(
    log: DecisionLog,
    posizione: int,
) -> tuple[DecisionRecord, ...]:
    """Return decisions made at one position in the current trick."""

    if posizione not in range(4):
        raise ValueError("posizione deve essere tra 0 e 3")
    return tuple(
        record
        for record in log.records
        if record.osservazione.posizione_nella_presa == posizione
    )


def records_chosen_with_probability_below(
    log: DecisionLog,
    threshold: float,
) -> tuple[DecisionRecord, ...]:
    """Return decisions whose chosen action had probability below threshold."""

    if threshold < 0.0:
        raise ValueError("threshold deve essere non negativa")
    return tuple(
        record
        for record in log.records
        if record.action_probabilities[record.azione] < threshold
    )


def _partner_leading(record: DecisionRecord) -> bool:
    winner = _current_trick_winner(record)
    return winner == record.osservazione.compagno_id


def _opponent_leading(record: DecisionRecord) -> bool:
    winner = _current_trick_winner(record)
    return winner in record.osservazione.avversari


def _current_trick_winner(record: DecisionRecord) -> int | None:
    if not record.osservazione.carte_sul_campo:
        return None
    return vincitore_presa(
        record.osservazione.carte_sul_campo,
        seme_briscola=record.osservazione.seme_briscola,
    ).giocatore_id
