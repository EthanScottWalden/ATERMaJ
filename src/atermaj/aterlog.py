from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class Aterlog(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.writer = None
        self.episodes = 0
        self.wins = 0
        self.session_max_ante = 1
        self.session_max_round = 0
        self.joker_rounds: dict[str, int] = dict()
        self.consumable_usages: dict[str, int] = dict()

    def _on_training_start(self) -> None:
        output = next(
            (
                item
                for item in self.logger.output_formats
                if isinstance(item, TensorBoardOutputFormat)
            ),
            None,
        )
        if output is None:
            raise RuntimeError("Enable tensorboard_log when creating the model.")
        self.writer = output.writer

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue

            self.episodes += 1
            step = self.episodes

            ante = int(info.get("episode/ante_reached", 1))
            rounds = int(info.get("episode/rounds_beaten", 0))
            won = bool(info.get("episode/won", False))

            self.writer.add_scalar("episode/ante_reached", ante, step)
            self.writer.add_scalar("episode/rounds_beaten", rounds, step)
            self.writer.add_scalar("episode/won", int(won), step)

            self.wins += int(won)
            self.session_max_ante = max(
                self.session_max_ante,
                int(info.get("session/max_ante_reached", ante)),
            )

            self.session_max_round = max(
                self.session_max_round,
                int(info.get("session/max_rounds_beaten", rounds)),
            )

            self.joker_rounds = dict(info.get("session/joker_rounds", {}))
            self.consumable_usages = dict(info.get("session/consumable_usages", {}))

        return True

    def _on_training_end(self) -> None:
        if not self.episodes:
            return

        step = self.episodes
        self.writer.add_scalar("session/win_rate", self.wins / self.episodes, step)
        self.writer.add_scalar("session/max_ante_reached", self.session_max_ante, step)
        self.writer.add_scalar("session/max_rounds_beaten", self.session_max_round, step)

        for name, count in self.joker_rounds.items():
            self.writer.add_scalar(f"session/joker_rounds/{name}", count, step)
        for name, count in self.consumable_usages.items():
            self.writer.add_scalar(f"session/consumable_usages/{name}", count, step)

        self.writer.flush()