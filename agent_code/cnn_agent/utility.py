

class DataCollector:

    def __init__(self, file_name="score_per_round.txt"):
        self.file_name = file_name
        self.training_iter = 0
        self.round = 0
        self.epsilon = 0
        self.score = 0
        self.avg_loss_per_step = 0
        self.killed_self = 0
        self.avg_reward_per_step = 0
        self.invalid_actions_per_round = 0

    def initialize(self):
        with open(self.file_name, "w") as file:
            file.write("training_iter\t round\t epsilon\t score\t avg_loss_per_step\t killed_self\t avg_reward_per_step\t invalid_actions_per_round\n")

    def write(self, train_iter, round, epsilon, score, killed_self, avg_loss_per_step, avg_reward_per_step, invalid_actions_per_round):
        with open(self.file_name, "a") as file:
            file.write(f"{train_iter}\t {round}\t {epsilon:.4f}\t {score}\t {avg_loss_per_step:.4f}\t {killed_self}\t {avg_reward_per_step:.4f}\t {invalid_actions_per_round}\n")
