def main() -> None:
    from sb3_contrib import MaskablePPO
    from jackdaw.env import DirectAdapter
    from atermaj.atermaj_env import AtermajEnv
    from atermaj.aterlog import Aterlog
    from jackdaw.env import BalatroGymnasiumEnv

    MODEL_NAME = "ATERMaJ_v0"

    # env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True, back_keys=["b_blue"], max_steps=1_000_000)
    env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True, back_keys=["b_blue"], max_steps=2_000_000)
    
    # model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"runs/{MODEL_NAME}")
    model = MaskablePPO.load(path="models/ATERMaJ_v0.zip", env=env, verbose=1, tensorboard_log=f"runs/{MODEL_NAME}")
    model.learn(total_timesteps=200_000, callback=Aterlog())
    model.save(f"models/{MODEL_NAME}")