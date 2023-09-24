

class DataCollector:

    def __init__(self, file_name="score_per_round.txt"):
        self.file_name = file_name

    def initialize(self):
        with open(self.file_name, "w") as file:
            file.write("training_iter\tround\tepsilon\tscore\tavg_loss_per_step\tkilled_self\tavg_reward_per_step\tinvalid_actions_per_round\tavg_invalid_actions_per_step\tescaped_bombs\n")

    def write(self, train_iter, round, epsilon, score, killed_self, avg_loss_per_step, avg_reward_per_step, invalid_actions_per_round, avg_invalid_actions_per_step,escaped_bombs):
        with open(self.file_name, "a") as file:
            file.write(f"{train_iter}\t{round}\t{epsilon: .4f}\t{score}\t{avg_loss_per_step: .4f}\t{killed_self}\t{avg_reward_per_step: .4f}\t{invalid_actions_per_round}\t{avg_invalid_actions_per_step: .4f}\t{escaped_bombs}\n")