import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

plt.rcParams.update({"font.size": 16})

simple_double_list = []
names = ["simpleDQN", "doubleDQN"]
for name in names:
    score = []
    avg_loss_per_step = []
    killed_self = []
    avg_reward_per_step = []
    avg_invalid_actions_per_step = []
    steps_survived = []
    with open(f"score_per_round_{name}_classic.txt") as file:
        file.readline()
        for line in file:
            data = line.split("\t")
            score.append(int(data[3]))
            avg_loss_per_step.append(float(data[4]))
            killed_self_value = True if data[5] == "True" else False
            killed_self.append(killed_self_value)
            avg_reward_per_step.append(float(data[6]))
            avg_invalid_actions_per_step.append(float(data[8]))
            steps_survived.append(int(data[9].strip()))

    lists = [
        score,
        killed_self,
        avg_loss_per_step,
        avg_reward_per_step,
        avg_invalid_actions_per_step,
        steps_survived,
    ]

    simple_double_list.append(lists)

    print(
        f"{name}: score: {np.mean(score)}; avg_steps_survived: {np.mean(steps_survived)}"
    )


# for Score
score_colors = ["red", "blue"]
for name, model_values, color in zip(names, simple_double_list, score_colors):
    values = model_values[0]
    killed_self = model_values[1]
    step = 10
    cmap = ListedColormap(["white", "green"])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = [i for i in range(len(values))]
    ax.plot(x[::step], values[::step], linestyle="--", marker="o", color=color)
    ax.set_xlim((0, 10000))
    ax.set_ylim((0, 10))
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.pcolorfast(
        ax.get_xlim(),
        ax.get_ylim(),
        np.array(killed_self[::step])[None],
        cmap=cmap,
    )
    plt.tight_layout()
    plt.savefig(f"Score_{name}.png", dpi=300)


lists_names = [
    "Average Loss per Step",
    "Average Reward per Step",
    "Average Invalid Actions per Step",
    "Steps Survived",
]
for model_list in simple_double_list:
    model_list.pop(1)
    model_list.pop(0)

for i, list_name in enumerate(lists_names):
    colors = ["red", "blue"]
    model_names = ["standard DQN", "double DQN"]
    fig, ax = plt.subplots(figsize=(12, 5))
    step = 40
    ax.set_xlim((0, 10000))
    ax.set_xlabel("Round")
    ax.set_ylabel(f"{list_name}")
    x = [i for i in range(10000)]
    for index, model_values in enumerate(simple_double_list):
        values = model_values[i]
        ax.plot(
            x[::step],
            values[::step],
            linestyle="--",
            marker="o",
            color=colors[index],
            label=f"{model_names[index]}",
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{list_name}.png", dpi=300)
