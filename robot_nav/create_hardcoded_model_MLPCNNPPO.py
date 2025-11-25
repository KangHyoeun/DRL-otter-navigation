import torch
import numpy as np
from pathlib import Path
from robot_nav.models.PPO.MLPCNNPPO import MLPCNNPPO

def create_hardcoded_model():
    print("🔨 Creating Hardcoded MLPCNNPPO Model...")
    
    # Hyperparameters (must match training script)
    action_dim = 2
    max_action = 1
    state_dim = 12
    device = torch.device("cpu") # Use CPU for creation
    
    # Initialize Model
    model = MLPCNNPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )
    
    # --- HARDCODING WEIGHTS ---
    print("   Setting actor weights to zero and bias to [1.0, 0.0]...")
    
    # 1. Set Actor Weights to 0
    # This ensures that the input features (from CNN/MLP) have NO effect on the output
    model.actor.weight.data.fill_(0.0)
    
    # 2. Set Actor Bias to [1.0, 0.0]
    # This ensures the output is always [1.0, 0.0]
    # Action 0 (Surge): 1.0 -> (1.0+1)*1.5 = 3.0 m/s (Max Speed)
    # Action 1 (Yaw):   0.0 -> 0.0 rad/s (Straight)
    model.actor.bias.data = torch.tensor([1.0, 0.0])
    
    # 3. Set Log Std to very small value (optional, for deterministic sampling)
    model.log_std.data.fill_(-20.0) 
    
    # --- SAVING ---
    save_dir = Path("robot_nav/models/PPO/best_checkpoint")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = "otter_MLPCNNPPO_imazu_00_hardcoded"
    
    model.save(filename=model_name, directory=save_dir)
    print(f"✅ Hardcoded model saved to: {save_dir / (model_name + '_policy.pth')}")

    # --- VERIFICATION ---
    print("\n🔍 Verifying model output...")
    dummy_vec = torch.randn(1, state_dim)
    dummy_grid = torch.randn(1, 1, 64, 64)
    
    action, _, _ = model.get_action(dummy_vec, dummy_grid, sample=False) # Deterministic
    print(f"   Input: Random Noise")
    print(f"   Output Action: {action.tolist()[0]}")
    
    if np.allclose(action.tolist()[0], [1.0, 0.0], atol=1e-5):
        print("   ✅ Verification PASSED: Output is exactly [1.0, 0.0]")
    else:
        print("   ❌ Verification FAILED: Output is NOT [1.0, 0.0]")

if __name__ == "__main__":
    create_hardcoded_model()
