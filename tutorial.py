# tutorial.py

import gymnasium as gym

def main():
    # render_mode="human" for simulation HalfCheetah
    env = gym.make("HalfCheetah-v5", render_mode="human")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    obs, info = env.reset(seed=42)
    total_reward = 0.0

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    print(f"총 보상: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
