from __future__ import annotations

import unittest

from training.rewards import (
    PUNTI_TOTALI_PARTITA,
    REWARD_MODES,
    RewardConfig,
    calcola_margine,
    calcola_segno,
    normalizza_margine,
    reward_finale,
    reward_presa,
)


class TestRewards(unittest.TestCase):
    def test_calcola_margine_dal_punto_di_vista_del_learner(self):
        # A positive margin means the learner's team is ahead.
        self.assertEqual(calcola_margine(70, 50), 20)
        self.assertEqual(calcola_margine(45, 75), -30)
        self.assertEqual(calcola_margine(60, 60), 0)

    def test_calcola_segno_codifica_vittoria_sconfitta_pareggio(self):
        # The sign isolates the win/loss convention in one tested point.
        self.assertEqual(calcola_segno(20), 1.0)
        self.assertEqual(calcola_segno(-1), -1.0)
        self.assertEqual(calcola_segno(0), 0.0)

    def test_normalizza_margine_usa_i_punti_totali_partita(self):
        # Normalization avoids raw point-scale margin rewards.
        self.assertEqual(PUNTI_TOTALI_PARTITA, 120)
        self.assertAlmostEqual(normalizza_margine(30), 0.25)
        self.assertAlmostEqual(normalizza_margine(-60), -0.5)

    def test_reward_config_default_e_mode_ammessi(self):
        # Defaults declare the initial protocol and explicit modes.
        default = RewardConfig()

        self.assertEqual(default.mode, "combined_terminal")
        self.assertEqual(default.alpha, 1.0)
        self.assertEqual(default.lambda_margin, 0.2)
        self.assertEqual(REWARD_MODES, {"combined_terminal", "dense_presa"})
        self.assertEqual(RewardConfig(mode="dense_presa").mode, "dense_presa")

    def test_reward_config_rifiuta_mode_illegale(self):
        # Fail fast on typos in the reward mode name.
        with self.assertRaises(ValueError):
            RewardConfig(mode="non_tra_le_opzioni")  # type: ignore[arg-type]

    def test_reward_config_rifiuta_pesi_negativi(self):
        # Negative weights would reverse the reward meaning without an explicit decision.
        with self.assertRaises(ValueError):
            RewardConfig(alpha=-1.0)

        with self.assertRaises(ValueError):
            RewardConfig(lambda_margin=-0.1)

    def test_reward_finale_combined_terminal_unisce_segno_e_margine(self):
        # Terminal reward combines match outcome and normalized margin.
        config = RewardConfig(mode="combined_terminal", alpha=1.0, lambda_margin=0.2)

        reward = reward_finale(70, 50, config)

        self.assertIsInstance(reward, float)
        self.assertAlmostEqual(reward, 1.0 + 0.2 * (20 / 120))

    def test_reward_finale_combined_terminal_gestisce_sconfitta_e_pareggio(self):
        # The same formula must remain coherent below zero and at zero margin.
        config = RewardConfig(mode="combined_terminal", alpha=1.0, lambda_margin=0.2)

        self.assertAlmostEqual(reward_finale(50, 70, config), -1.0 + 0.2 * (-20 / 120))
        self.assertEqual(reward_finale(60, 60, config), 0.0)

    def test_terminal_win_si_ottiene_con_lambda_margin_zero(self):
        # At zero margin, terminal reward depends only on win or loss.
        config = RewardConfig(mode="combined_terminal", alpha=1.0, lambda_margin=0.0)

        self.assertEqual(reward_finale(61, 59, config), 1.0)
        self.assertEqual(reward_finale(59, 61, config), -1.0)
        self.assertEqual(reward_finale(60, 60, config), 0.0)

    def test_reward_finale_dense_presa_usa_solo_il_segno_finale(self):
        # The dense part lives on tricks; the terminal part remains win/loss.
        config = RewardConfig(mode="dense_presa", alpha=1.0, lambda_margin=0.2)

        reward = reward_finale(70, 50, config)

        self.assertIsInstance(reward, float)
        self.assertEqual(reward, 1.0)
        self.assertEqual(reward_finale(50, 70, config), -1.0)
        self.assertEqual(reward_finale(60, 60, config), 0.0)

    def test_reward_presa_combined_terminal_resta_zero_float(self):
        # combined_terminal does not assign immediate rewards on tricks.
        reward = reward_presa(22, presa_vinta_da_squadra=True)

        self.assertIsInstance(reward, float)
        self.assertEqual(reward, 0.0)

    def test_reward_presa_dense_presa_normalizza_punti_con_segno(self):
        # dense_presa assigns positive or negative normalized points to each trick.
        config = RewardConfig(mode="dense_presa", alpha=1.0, lambda_margin=0.2)

        reward_vinta = reward_presa(24, presa_vinta_da_squadra=True, config=config)
        reward_persa = reward_presa(24, presa_vinta_da_squadra=False, config=config)

        self.assertIsInstance(reward_vinta, float)
        self.assertIsInstance(reward_persa, float)
        self.assertAlmostEqual(reward_vinta, 0.2 * (24 / 120))
        self.assertAlmostEqual(reward_persa, -0.2 * (24 / 120))


if __name__ == "__main__":
    unittest.main()
