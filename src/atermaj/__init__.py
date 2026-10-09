def main() -> None:
    from sb3_contrib import MaskablePPO
    from jackdaw.env import DirectAdapter
    from atermaj.atermaj_env import AtermajEnv
    from atermaj.aterlog import Aterlog
    from jackdaw.env import BalatroGymnasiumEnv

    MODEL_NAME = "ATERMaJ_v0"
    CHUNKS = 5
    ATERLOG = Aterlog()

    ALREADY_RESET_STEPS = False
    
    # env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True, back_keys=["b_blue"], max_steps=1_000_000)
    for _ in range(CHUNKS):
        env = AtermajEnv(adapter_factory=DirectAdapter, reward_shaping=True, back_keys=["b_blue"], max_steps=2_000_000)
        
        try:
            model = MaskablePPO.load(path=f"models/{MODEL_NAME}.zip", env=env, verbose=1, tensorboard_log=f"runs/{MODEL_NAME}")
            model.learn(total_timesteps=200_000, reset_num_timesteps=(not ALREADY_RESET_STEPS), callback=ATERLOG)

            ALREADY_RESET_STEPS = True

            model.save(f"models/{MODEL_NAME}")
        finally:
            env.close()