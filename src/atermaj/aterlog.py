from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

        self.ante_by_episode: list[int] = []
        self.rounds_by_episode: list[int] = []
        self.wins_by_episode: list[int] = []

    def _add_count_chart(self, tag: str, title: str, counts: dict[str, int]) -> None:
        if not counts:
            return

        items = sorted(counts.items(), key=lambda item: item[1])
        names, values = zip(*items)

        fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.35)))
        ax.barh(names, values)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)
        ax.set_xlabel("Count")
        fig.tight_layout()

        self.writer.add_figure(tag, fig, global_step=self.num_timesteps)
        plt.close(fig)

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

            ante = int(info.get("episode/ante_reached", 1))
            rounds = int(info.get("episode/rounds_beaten", 0))
            won = bool(info.get("episode/won", False))

            self.ante_by_episode.append(ante)
            self.rounds_by_episode.append(rounds)
            self.wins_by_episode.append(int(won))

            # self.writer.add_scalar("episode/ante_reached", ante, step)
            # self.writer.add_scalar("episode/rounds_beaten", rounds, step)
            # self.writer.add_scalar("episode/won", int(won), step)

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

            self.episodes += 1

        return True

    def _on_training_end(self) -> None:
        if not self.episodes:
            return

        self.writer.add_scalar("session/win_rate", self.wins / self.episodes, global_step=self.episodes)

        episodes = range(1, self.episodes + 1)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        series = [
            (axes[0], self.ante_by_episode, "Ante reached"),
            (axes[1], self.rounds_by_episode, "Rounds beaten"),
        ]
        for ax, values, label in series:
            ax.step(episodes, values, where="mid", marker=".", markersize=3)
            ax.set_ylabel(label)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="y", alpha=0.3)

        axes[2].step(episodes, self.wins_by_episode, where="mid")
        axes[2].set_ylabel("Result")
        axes[2].set_yticks([0, 1], ["Loss", "Win"])
        axes[2].grid(True, axis="y", alpha=0.3)
        axes[2].set_xlabel("Episode")

        fig.tight_layout()
        self.writer.add_figure(
            "session/progression", fig, global_step=self.num_timesteps
        )
        plt.close(fig)

        # step = self.episodes
        # self.writer.add_scalar("session/win_rate", self.wins / self.episodes, step)
        # self.writer.add_scalar("session/max_ante_reached", self.session_max_ante, step)
        # self.writer.add_scalar("session/max_rounds_beaten", self.session_max_round, step)

        # for name, count in self.joker_rounds.items():
        #     self.writer.add_scalar(f"session/joker_rounds/{name}", count, step)
        # for name, count in self.consumable_usages.items():
        #     self.writer.add_scalar(f"session/consumable_usages/{name}", count, step)

        self._add_count_chart(
            "session/joker_rounds_chart", "Rounds spent with each joker", self.joker_rounds
        )
        self._add_count_chart(
            "session/consumable_usages_chart", "Consumable uses", self.consumable_usages
        )

        self.writer.flush()