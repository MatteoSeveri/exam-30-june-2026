from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from game.cards import Carta
from game.observation import Osservazione
from policy import BriscolaFeatureExtractor, NeuralSoftmaxPolicy


def osservazione(
    mano: tuple[Carta, ...] = (
        Carta("coppe", "asso"),
        Carta("bastoni", "due"),
        Carta("denari", "tre"),
    ),
) -> Osservazione:
    return Osservazione(
        giocatore_id=0,
        compagno_id=2,
        avversario_sinistro_id=1,
        avversario_destro_id=3,
        mano=mano,
        mano_compagno_visibile=False,
        mano_compagno=(),
        seme_briscola="denari",
        briscola_esposta=Carta("denari", "asso"),
        proprietario_briscola_esposta=None,
        carte_sul_campo=(),
        carte_giocate=(),
        vincitori_prese=(),
        squadra="pari",
        squadra_avversaria="dispari",
        punteggio_squadra=0,
        punteggio_avversari=0,
        primo_giocatore_presa=0,
        giocatore_corrente=0,
        carte_nel_mazzo=28,
        indice_presa=0,
        posizione_nella_presa=0,
    )


class TestNeuralSoftmaxPolicy(unittest.TestCase):
    def test_initialize_crea_parametri_della_dimensione_mlp(self):
        # The flat theta stores all MLP parameters in a reproducible order.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso", "carta_tre"])

        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(0),
            hidden_size=3,
        )

        self.assertEqual(
            len(policy.theta),
            NeuralSoftmaxPolicy.parameter_count(extractor.size(), 3),
        )
        self.assertEqual(policy.theta.dtype, np.float32)
        self.assertIs(policy.feature_extractor, extractor)

    def test_hidden_size_non_valido_solleva_value_error(self):
        # The network needs at least one hidden unit.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])

        with self.assertRaises(ValueError):
            NeuralSoftmaxPolicy.initialize(extractor, hidden_size=0)

    def test_theta_con_dimensione_errata_solleva_value_error(self):
        # This prevents silent bugs when reconstructing neural checkpoints.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])

        with self.assertRaises(ValueError):
            NeuralSoftmaxPolicy(theta=[0.0], feature_extractor=extractor, hidden_size=2)

    def test_probabilita_sommano_a_uno_sulle_carte_legali(self):
        # Softmax must return a distribution only over legal actions.
        obs = osservazione()
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(1),
            hidden_size=4,
        )

        probabilities = policy.action_probabilities(obs)

        self.assertEqual(set(probabilities), set(obs.azioni_legali))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, delta=1e-6)

    def test_select_action_stocastico_restituisce_carta_legale(self):
        # In stochastic mode, it samples from the legal-action distribution.
        obs = osservazione()
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(2),
            hidden_size=4,
        )

        action = policy.select_action(obs, rng=random.Random(3))

        self.assertIn(action, obs.azioni_legali)

    def test_log_probability_tensor_alimenta_autograd(self):
        # PyTorch REINFORCE consumes a differentiable log-probability.
        obs = osservazione()
        action = obs.mano[0]
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(4),
            hidden_size=4,
        )

        log_probability = policy.log_probability_tensor(obs, action)
        (-log_probability).backward()

        self.assertIsInstance(log_probability, torch.Tensor)
        self.assertEqual(log_probability.shape, torch.Size([]))
        self.assertTrue(
            any(parameter.grad is not None for parameter in policy.parameters())
        )

    def test_action_non_legale_solleva_value_error_per_log_probability(self):
        # Log probability is defined only for legal actions.
        obs = osservazione()
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(5),
            hidden_size=4,
        )
        illegal_action = Carta("spade", "asso")

        with self.assertRaises(ValueError):
            policy.log_probability(obs, illegal_action)

        with self.assertRaises(ValueError):
            policy.log_probability_tensor(obs, illegal_action)

    def test_copy_duplica_theta_ma_mantiene_architettura(self):
        # Snapshots must preserve both parameters and hidden size.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = NeuralSoftmaxPolicy.initialize(
            extractor,
            rng=random.Random(7),
            hidden_size=2,
        )

        copied = policy.copy(name="snapshot")
        copied_theta = copied.theta
        copied_theta[0] = copied_theta[0] + 1.0
        copied.theta = copied_theta

        self.assertEqual(copied.name, "snapshot")
        self.assertEqual(copied.hidden_size, policy.hidden_size)
        self.assertFalse(np.allclose(policy.theta, copied.theta))
        self.assertIs(copied.feature_extractor, extractor)

    def test_mano_vuota_solleva_value_error(self):
        # A softmax distribution requires at least one legal action.
        obs = osservazione(mano=())
        policy = NeuralSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(8),
            hidden_size=4,
        )

        with self.assertRaises(ValueError):
            policy.action_probabilities(obs)

        with self.assertRaises(ValueError):
            policy.select_action(obs, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
