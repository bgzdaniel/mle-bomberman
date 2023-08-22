

class DataCollector:

    def __init__(self, file_name="score_per_round.txt"):
        self.file_name = file_name

    def initialize(self):
        with open(self.file_name, "w") as file:
            file.write("round\tepsilon\tscore\tavg_loss_per_step\tkilled_self\tavg_reward_per_step\tinvalid_actions_per_round\tavg_invalid_actions_per_step\tdropped_bombs\n")

    def write(self, round, epsilon, score, killed_self, avg_loss_per_step, avg_reward_per_step, invalid_actions_per_round, avg_invalid_actions_per_step,dropped_bombs):
        with open(self.file_name, "a") as file:
            file.write(f"{round}\t{epsilon: .4f}\t{score}\t{avg_loss_per_step: .4f}\t{killed_self}\t{avg_reward_per_step: .4f}\t{invalid_actions_per_round}\t{avg_invalid_actions_per_step: .4f}\t{dropped_bombs}\n")
