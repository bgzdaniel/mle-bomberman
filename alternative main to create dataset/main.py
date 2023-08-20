import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm
import json
import numpy as np
import shutil

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


#def save_agent_state_and_action(agent_name, state, action, round_id, step):
#    agent_dir = f"{agent_name}_results/round_{round_id:03}"
#    if not os.path.exists(agent_dir):
#        os.makedirs(agent_dir)

#    result_filename = os.path.join(agent_dir, f"step_{step:04d}_result.json")
#    result_data = {"state": state, "action": action}
    
#    with open(result_filename, "w") as result_file:
#        json.dump(result_data, result_file, indent=4, cls=NumpyEncoder)

        

def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)
    
    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        
        round_id = world.round_id  # Get the current round number
        step = 0  # Initialize step counter for this round
        agent_results = {}  # Dictionary to hold results for each agent

        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]
        
            # Advances step (for turn based: only if user input is available)
            #if world.running and not (turn_based and user_input is None):
             #   world.do_step(user_input)
              #  user_input = None
               # 
                # Save state and action to file for the agent
                #state = world.get_state_for_agent(world.agents[0])  # Get state for the first agent (assumes single-player)
                #action = world.agents[0].last_action
                #save_agent_state_and_action(world.agents[0].name, state, action, round_id, step)
                #step += 1  # Increment step counter
            
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None

                for agent in world.agents:
                    state = world.get_state_for_agent(agent)
                    action = agent.last_action
                    events = agent.events
                    if agent not in agent_results:
                        agent_results[agent] = []
                    agent_results[agent].append({"state": state, "action": action, "events": events})

                step += 1  # Increment step counter


            else:
                # Might want to wait
                pass
            
        # Determine the agent with the highest score in this round
        round_scores = {agent: agent.score for agent in world.agents}
        highest_score_agent = max(round_scores, key=round_scores.get)

        # Save state and action data for each agent
        for agent, results in agent_results.items():
            agent_dir = f"{agent.name}_results/round_{round_id:03}"
            if not os.path.exists(agent_dir):
                os.makedirs(agent_dir)

            result_filename = os.path.join(agent_dir, f"step_{step:04d}_result.json")
            with open(result_filename, "w") as result_file:
                json.dump(results, result_file, indent=4, cls=NumpyEncoder)
                
                
        # Delete directories of agents with lower scores
        for agent in world.agents:
            agent_dir = f"{agent.name}_results/round_{world.round_id:03}"
            if agent != highest_score_agent and os.path.exists(agent_dir):
                shutil.rmtree(agent_dir)



        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if gui is not None:
            do_continue = False
            while not do_continue:
                render(True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                            do_continue = True
            


    world.end()


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10000, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames
    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None
    world_controller(world, args.n_rounds,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()
