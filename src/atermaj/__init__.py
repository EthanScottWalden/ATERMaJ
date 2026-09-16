def main() -> None:
    from sb3_contrib import MaskablePPO
    from jackdaw.env import DirectAdapter
    from .lib.atermaj_env import AtermajEnv

    env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True)

    model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log="runs/balatro_ppo")
    model.learn(total_timesteps=5)
    model.save("balatro_ppo")