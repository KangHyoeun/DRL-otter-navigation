# Monitoring Training

Monitoring the training process is crucial for understanding how the agent is learning and for diagnosing potential issues. This project uses **TensorBoard** to log and visualize key metrics during training.

## Launching TensorBoard

While your training script is running, open a **new terminal** in the same project root directory (`/home/hyo/DRL-otter-navigation`) and run the following command:

```bash
# Ensure your conda environment is activated
tensorboard --logdir runs
```

This will start the TensorBoard server. You can access the dashboard by opening the URL provided in the terminal (usually `http://localhost:6006`) in your web browser.

## Key Metrics to Watch

When you open TensorBoard, you will see several plots. Here are the most important ones to monitor:

### Evaluation Metrics

These metrics show the agent's actual performance on the task during the evaluation phase at the end of each epoch.

-   **`eval/avg_reward`**: The average total reward per episode. **This is the most important indicator of performance.** It should generally trend upwards.
-   **`eval/goal_rate`**: The percentage of episodes where the agent successfully reached the goal. This should increase towards 1.0.
-   **`eval/collision_rate`**: The percentage of episodes where the agent collided with an obstacle. This should decrease towards 0.0.

### Training & Loss Metrics

These metrics give insight into the health of the PPO algorithm itself.

-   **`train/critic_loss`**: The loss of the critic network. It's normal for this to be high at the beginning, but it should trend downwards as the critic gets better at predicting future rewards.
-   **`train/explained_variance`**: Measures how well the critic's value predictions match the actual returns. A value close to 1.0 is ideal. A low or negative value indicates a problem with the critic's learning.
-   **`train/entropy`**: A measure of the policy's randomness or exploration. It should start high and gradually decrease as the agent becomes more confident in its actions.
-   **`train/approx_kl`**: The KL divergence, which measures how much the policy changes with each update. It should remain small and stable. Large spikes can indicate instability.
-   **`train/clip_fraction`**: The fraction of PPO updates that were clipped. This helps to diagnose if the learning rate is too high or too low.
