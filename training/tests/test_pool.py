from __future__ import annotations

import random
import unittest

import numpy as np

from policy import BriscolaFeatureExtractor, LinearSoftmaxPolicy
from training.pool import SnapshotPool


def policy_con_theta(
    theta: list[float],
    extractor: BriscolaFeatureExtractor,
    name: str,
) -> LinearSoftmaxPolicy:
    return LinearSoftmaxPolicy(theta=theta, feature_extractor=extractor, name=name)


class TestSnapshotPool(unittest.TestCase):
    def test_add_policy_copia_theta_e_congela_snapshot(self):
        # Lo snapshot non deve cambiare se la policy viva viene aggiornata dopo.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = policy_con_theta([1.0], extractor, "learner")
        pool = SnapshotPool(feature_extractor=extractor)

        pool.add_policy(policy, name="snapshot_1", update_index=1)
        policy.theta[0] = 9.0

        self.assertEqual(len(pool), 1)
        self.assertTrue(
            np.allclose(pool.snapshots[0].theta, np.asarray([1.0], dtype=np.float32))
        )
        self.assertFalse(pool.snapshots[0].theta.flags.writeable)

    def test_sample_policy_restituisce_policy_indipendente(self):
        # Ogni campionamento deve avere memoria theta separata dallo snapshot.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = policy_con_theta([2.0], extractor, "learner")
        pool = SnapshotPool(feature_extractor=extractor)
        pool.add_policy(policy, name="snapshot_1", update_index=1)

        sampled = pool.sample_policy(random.Random(0))
        sampled.theta[0] = 7.0

        self.assertIsInstance(sampled, LinearSoftmaxPolicy)
        self.assertEqual(sampled.name, "snapshot_1")
        self.assertIs(sampled.feature_extractor, extractor)
        self.assertTrue(
            np.allclose(pool.snapshots[0].theta, np.asarray([2.0], dtype=np.float32))
        )
        self.assertTrue(
            np.allclose(sampled.theta, np.asarray([7.0], dtype=np.float32))
        )

    def test_due_sample_dallo_stesso_snapshot_non_condividono_theta(self):
        # Due giocatori che pescano lo stesso snapshot non devono condividere memoria.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        policy = policy_con_theta([3.0], extractor, "learner")
        pool = SnapshotPool(feature_extractor=extractor)
        pool.add_policy(policy, name="snapshot_1", update_index=1)

        first = pool.sample_policy(random.Random(0))
        second = pool.sample_policy(random.Random(0))
        first.theta[0] = 8.0

        self.assertTrue(np.allclose(second.theta, np.asarray([3.0], dtype=np.float32)))
        self.assertFalse(np.shares_memory(first.theta, second.theta))

    def test_sample_policy_su_pool_vuoto_solleva_value_error(self):
        # Il campionamento da pool vuoto e' un errore di configurazione.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        pool = SnapshotPool(feature_extractor=extractor)

        with self.assertRaises(ValueError):
            pool.sample_policy(random.Random(0))

    def test_max_size_minore_di_uno_solleva_value_error(self):
        # Il pool deve poter conservare almeno uno snapshot.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])

        with self.assertRaises(ValueError):
            SnapshotPool(feature_extractor=extractor, max_size=0)

    def test_retention_conserva_initial_e_recenti(self):
        # La retention minimale conserva initial e gli snapshot piu' recenti.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        pool = SnapshotPool(feature_extractor=extractor, max_size=4, keep_initial=True)

        for index in range(6):
            pool.add_policy(
                policy_con_theta([float(index)], extractor, f"policy_{index}"),
                name="initial" if index == 0 else f"snapshot_{index}",
                update_index=index,
            )

        self.assertEqual(
            [snapshot.name for snapshot in pool.snapshots],
            ["initial", "snapshot_3", "snapshot_4", "snapshot_5"],
        )

    def test_retention_senza_initial_conserva_solo_recenti(self):
        # Se keep_initial e' falso, il pool tiene semplicemente gli ultimi snapshot.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        pool = SnapshotPool(feature_extractor=extractor, max_size=3, keep_initial=False)

        for index in range(5):
            pool.add_policy(
                policy_con_theta([float(index)], extractor, f"policy_{index}"),
                name=f"snapshot_{index}",
                update_index=index,
            )

        self.assertEqual(
            [snapshot.name for snapshot in pool.snapshots],
            ["snapshot_2", "snapshot_3", "snapshot_4"],
        )

    def test_retention_non_ha_logica_di_score(self):
        # Il pool minimale conserva initial+recenti, senza campi di valutazione.
        extractor = BriscolaFeatureExtractor(feature_names=["carta_asso"])
        pool = SnapshotPool(feature_extractor=extractor, max_size=3, keep_initial=True)

        pool.add_policy(policy_con_theta([0.0], extractor, "initial"), "initial", 0)
        pool.add_policy(policy_con_theta([1.0], extractor, "old"), "old", 1)
        pool.add_policy(policy_con_theta([2.0], extractor, "recent_2"), "recent_2", 2)
        pool.add_policy(policy_con_theta([3.0], extractor, "recent_3"), "recent_3", 3)

        self.assertEqual(
            [snapshot.name for snapshot in pool.snapshots],
            ["initial", "recent_2", "recent_3"],
        )
        self.assertFalse(hasattr(pool.snapshots[0], "evaluation_score"))


if __name__ == "__main__":
    unittest.main()
