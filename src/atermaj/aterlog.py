from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from collections import Counter

STAKES: dict[int, str] = {
    1: "White",
    2: "Red",
    3: "Green",
    4: "Black",
    5: "Blue",
    6: "Purple",
    7: "Orange",
    8: "Gold"
}

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
        self.wins_by_deck_stake: dict[str, dict[str, list[int]]] = dict()

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

    def _add_percent_chart(self, tag: str, title: str, percents: dict[str, float]) -> None:
        if not percents:
            return

        items = sorted(percents.items(), key=lambda item: item[1])
        names, values = zip(*items)

        fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.35)))
        ax.barh(names, values)
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.set_title(title)
        ax.set_xlabel("Percent")
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
            deck = info.get("episode/deck")
            stake = STAKES[int(info.get("episode/stake"))]

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

            for name, count in info.get("episode/joker_rounds", {}).items():
                self.joker_rounds[name] = self.joker_rounds.get(name, 0) + int(count)

            for name, count in info.get("episode/consumable_usages", {}).items():
                self.consumable_usages[name] = (
                    self.consumable_usages.get(name, 0) + int(count)
                )

            print(self.joker_rounds)
            print(self.consumable_usages)
            
            if self.wins_by_deck_stake.get(deck) is None:
                self.wins_by_deck_stake[deck] = dict()

            if self.wins_by_deck_stake[deck].get(stake) is None:
                self.wins_by_deck_stake[deck][stake] = []

            self.wins_by_deck_stake[deck][stake].append(int(won))

            self.episodes += 1

        return True

    def _on_training_end(self) -> None:
        if not self.episodes:
            return

        self.writer.add_scalar("session/win_rate", self.wins / self.episodes, global_step=self.num_timesteps)

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

        self._add_count_chart(
            "session/joker_rounds_chart", "Rounds spent with each joker", self.joker_rounds
        )
        self._add_count_chart(
            "session/consumable_usages_chart", "Consumable uses", self.consumable_usages
        )

        for deck in self.wins_by_deck_stake.keys():
            stake_rates: dict[str, int] = dict()

            for stake in self.wins_by_deck_stake[deck]:
                stake_wins: Counter = Counter(self.wins_by_deck_stake[deck][stake])
                stake_rates[stake] = stake_wins[1] / len(self.wins_by_deck_stake[deck][stake])

            self._add_percent_chart(
                f"session/win_rate_{deck}", f"Win rate on {deck} for each stake", stake_rates
            )

        self.writer.flush()