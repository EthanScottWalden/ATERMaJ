def main() -> None:
    from sb3_contrib import MaskablePPO
    from jackdaw.env import DirectAdapter
    from atermaj.atermaj_env import AtermajEnv

    MODEL_NAME = "ATERMaJ_v0"

    env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True, max_steps=1)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log=f"runs/{MODEL_NAME}")
    model.learn(total_timesteps=1)
    model.save(f"models/{MODEL_NAME}")