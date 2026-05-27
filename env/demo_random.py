"""Smoke test for CrowdRecEnv using a random legal policy."""

from __future__ import annotations

import argparse

from crowd_env import CrowdRecEnv, load_split


def run_random_policy(split: str, reward_type: str, max_steps: int | None, seed: int) -> None:
    data = load_split(split)
    env = CrowdRecEnv(data, reward_type=reward_type, seed=seed)
    state = env.reset()

    total_reward = 0.0
    hits = 0
    steps = 0
    done = False

    while not done:
        action = env.sample_random_action(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        hits += int(info["hit"])
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break
        state = next_state

    print("split:", split)
    print("reward_type:", reward_type)
    print("state_shape:", env.spec.state_shape)
    print("action_dim:", env.spec.action_dim)
    print("steps:", steps)
    print("hit_rate:", hits / steps if steps else 0.0)
    print("avg_reward:", total_reward / steps if steps else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--reward-type",
        choices=["worker", "requester", "requester_urgency", "hybrid"],
        default="worker",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_random_policy(args.split, args.reward_type, args.max_steps, args.seed)


if __name__ == "__main__":
    main()
