"""
COLREGs-Aware CNNTD3 Model
============================

기존 CNNTD3에 COLREGs 정보를 통합한 모델:
- Input: LiDAR scan + Goal info + COLREGs info (encounter type, risk, DCPA/TCPA)
- Output: [u_ref, r_ref] (surge velocity, yaw rate)

Author: Maritime Robotics Lab
Date: 2025-10-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional


class COLREGsFeatureExtractor(nn.Module):
    """
    COLREGs 정보를 feature vector로 변환
    
    Input: COLREGs info vector (max_targets * 8)
        각 target당 8 values:
        - encounter_type onehot (5)
        - risk_level (1)
        - dcpa_normalized (1)
        - tcpa_normalized (1)
    
    Output: COLREGs feature vector (colregs_feature_dim)
    """
    
    def __init__(self, max_targets: int = 5, colregs_feature_dim: int = 64):
        super(COLREGsFeatureExtractor, self).__init__()
        
        self.max_targets = max_targets
        self.input_dim = max_targets * 8
        
        # COLREGs info → feature vector
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, colregs_feature_dim)
        
        self.bn1 = nn.BatchNorm1d(128)
    
    def forward(self, colregs_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            colregs_info: (batch, max_targets * 8)
        
        Returns:
            features: (batch, colregs_feature_dim)
        """
        x = F.relu(self.bn1(self.fc1(colregs_info)))
        x = F.relu(self.fc2(x))
        return x


class COLREGsAwareActor(nn.Module):
    """
    COLREGs-Aware Actor Network
    
    Inputs:
    - LiDAR scan (CNN)
    - Goal info (distance, cos, sin)
    - COLREGs info (encounter types, risks, DCPA/TCPA)
    
    Output:
    - Action: [u_ref, r_ref]
    """
    
    def __init__(
        self,
        scan_dim: int = 180,
        goal_dim: int = 3,  # distance, cos, sin
        colregs_dim: int = 40,  # max_targets * 8
        action_dim: int = 2,  # u_ref, r_ref
        max_action: float = 1.0,
        colregs_feature_dim: int = 64
    ):
        super(COLREGsAwareActor, self).__init__()
        
        self.scan_dim = scan_dim
        self.goal_dim = goal_dim
        self.colregs_dim = colregs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # ===== LiDAR CNN =====
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        
        # CNN 출력 크기 계산
        conv_out_size = self._get_conv_output_size()
        
        # ===== COLREGs Feature Extractor =====
        self.colregs_extractor = COLREGsFeatureExtractor(
            max_targets=colregs_dim // 8,
            colregs_feature_dim=colregs_feature_dim
        )
        
        # ===== Fusion Layer =====
        fusion_input_dim = conv_out_size + goal_dim + colregs_feature_dim
        
        self.fc1 = nn.Linear(fusion_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
    
    def _get_conv_output_size(self):
        """CNN 출력 크기 계산"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.scan_dim)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            return x.view(1, -1).size(1)
    
    def forward(
        self,
        scan: torch.Tensor,
        goal: torch.Tensor,
        colregs_info: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scan: (batch, scan_dim) - LiDAR scan
            goal: (batch, goal_dim) - distance, cos, sin
            colregs_info: (batch, colregs_dim) - COLREGs info vector
        
        Returns:
            action: (batch, action_dim) - [u_ref, r_ref]
        """
        # LiDAR CNN
        x_scan = scan.unsqueeze(1)  # (batch, 1, scan_dim)
        x_scan = F.relu(self.conv1(x_scan))
        x_scan = F.relu(self.conv2(x_scan))
        x_scan = x_scan.view(x_scan.size(0), -1)  # Flatten
        
        # COLREGs feature extraction
        x_colregs = self.colregs_extractor(colregs_info)
        
        # Concatenate all features
        x = torch.cat([x_scan, goal, x_colregs], dim=1)
        
        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        action = self.max_action * torch.tanh(self.fc3(x))
        
        return action


class COLREGsAwareCritic(nn.Module):
    """
    COLREGs-Aware Critic Network (Twin Q-networks)
    
    Inputs:
    - State: LiDAR scan + Goal info + COLREGs info
    - Action: [u_ref, r_ref]
    
    Outputs:
    - Q1(s, a)
    - Q2(s, a)
    """
    
    def __init__(
        self,
        scan_dim: int = 180,
        goal_dim: int = 3,
        colregs_dim: int = 40,
        action_dim: int = 2,
        colregs_feature_dim: int = 64
    ):
        super(COLREGsAwareCritic, self).__init__()
        
        self.scan_dim = scan_dim
        self.goal_dim = goal_dim
        self.colregs_dim = colregs_dim
        self.action_dim = action_dim
        
        # ===== LiDAR CNN (shared) =====
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_output_size()
        
        # ===== COLREGs Feature Extractor (shared) =====
        self.colregs_extractor = COLREGsFeatureExtractor(
            max_targets=colregs_dim // 8,
            colregs_feature_dim=colregs_feature_dim
        )
        
        # ===== Q1 Network =====
        q_input_dim = conv_out_size + goal_dim + colregs_feature_dim + action_dim
        
        self.q1_fc1 = nn.Linear(q_input_dim, 512)
        self.q1_fc2 = nn.Linear(512, 256)
        self.q1_fc3 = nn.Linear(256, 1)
        
        self.q1_bn1 = nn.BatchNorm1d(512)
        self.q1_bn2 = nn.BatchNorm1d(256)
        
        # ===== Q2 Network =====
        self.q2_fc1 = nn.Linear(q_input_dim, 512)
        self.q2_fc2 = nn.Linear(512, 256)
        self.q2_fc3 = nn.Linear(256, 1)
        
        self.q2_bn1 = nn.BatchNorm1d(512)
        self.q2_bn2 = nn.BatchNorm1d(256)
    
    def _get_conv_output_size(self):
        """CNN 출력 크기 계산"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.scan_dim)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            return x.view(1, -1).size(1)
    
    def forward(
        self,
        scan: torch.Tensor,
        goal: torch.Tensor,
        colregs_info: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scan: (batch, scan_dim)
            goal: (batch, goal_dim)
            colregs_info: (batch, colregs_dim)
            action: (batch, action_dim)
        
        Returns:
            q1: (batch, 1)
            q2: (batch, 1)
        """
        # LiDAR CNN
        x_scan = scan.unsqueeze(1)  # (batch, 1, scan_dim)
        x_scan = F.relu(self.conv1(x_scan))
        x_scan = F.relu(self.conv2(x_scan))
        x_scan = x_scan.view(x_scan.size(0), -1)
        
        # COLREGs feature extraction
        x_colregs = self.colregs_extractor(colregs_info)
        
        # Concatenate: state + action
        x = torch.cat([x_scan, goal, x_colregs, action], dim=1)
        
        # Q1
        q1 = F.relu(self.q1_bn1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_bn2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)
        
        # Q2
        q2 = F.relu(self.q2_bn1(self.q2_fc1(x)))
        q2 = F.relu(self.q2_bn2(self.q2_fc2(q2)))
        q2 = self.q2_fc3(q2)
        
        return q1, q2
    
    def Q1(
        self,
        scan: torch.Tensor,
        goal: torch.Tensor,
        colregs_info: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Q1만 반환 (policy update용)"""
        # LiDAR CNN
        x_scan = scan.unsqueeze(1)
        x_scan = F.relu(self.conv1(x_scan))
        x_scan = F.relu(self.conv2(x_scan))
        x_scan = x_scan.view(x_scan.size(0), -1)
        
        # COLREGs features
        x_colregs = self.colregs_extractor(colregs_info)
        
        # Concatenate
        x = torch.cat([x_scan, goal, x_colregs, action], dim=1)
        
        # Q1 forward
        q1 = F.relu(self.q1_bn1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_bn2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)
        
        return q1


class COLREGsAwareTD3:
    """
    COLREGs-Aware TD3 Agent
    
    기존 TD3에 COLREGs 정보를 통합
    """
    
    def __init__(
        self,
        scan_dim: int = 180,
        goal_dim: int = 3,
        colregs_dim: int = 40,  # max_targets * 8
        action_dim: int = 2,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # Actor
        self.actor = COLREGsAwareActor(
            scan_dim, goal_dim, colregs_dim, action_dim, max_action
        ).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic
        self.critic = COLREGsAwareCritic(
            scan_dim, goal_dim, colregs_dim, action_dim
        ).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def select_action(
        self,
        scan: np.ndarray,
        goal: np.ndarray,
        colregs_info: np.ndarray,
        noise: bool = False
    ) -> np.ndarray:
        """
        액션 선택
        
        Args:
            scan: (scan_dim,)
            goal: (goal_dim,)
            colregs_info: (colregs_dim,)
            noise: 탐색 노이즈 추가 여부
        
        Returns:
            action: (action_dim,)
        """
        scan = torch.FloatTensor(scan).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        colregs_info = torch.FloatTensor(colregs_info).unsqueeze(0).to(self.device)
        
        action = self.actor(scan, goal, colregs_info).cpu().data.numpy().flatten()
        
        if noise:
            action = action + np.random.normal(0, self.max_action * 0.1, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(
        self,
        replay_buffer,
        batch_size: int = 256
    ) -> dict:
        """
        학습 1 step
        
        Returns:
            losses: {'critic_loss': float, 'actor_loss': float}
        """
        self.total_it += 1
        
        # Sample from replay buffer
        (scan, goal, colregs_info, action, 
         next_scan, next_goal, next_colregs_info, 
         reward, done) = replay_buffer.sample(batch_size)
        
        scan = torch.FloatTensor(scan).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)
        colregs_info = torch.FloatTensor(colregs_info).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_scan = torch.FloatTensor(next_scan).to(self.device)
        next_goal = torch.FloatTensor(next_goal).to(self.device)
        next_colregs_info = torch.FloatTensor(next_colregs_info).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (
                self.actor_target(next_scan, next_goal, next_colregs_info) + noise
            ).clamp(-self.max_action, self.max_action)
            
            # Target Q-values
            target_Q1, target_Q2 = self.critic_target(
                next_scan, next_goal, next_colregs_info, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Current Q-values
        current_Q1, current_Q2 = self.critic(scan, goal, colregs_info, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_action = self.actor(scan, goal, colregs_info)
            actor_loss = -self.critic.Q1(scan, goal, colregs_info, actor_action).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0
        }
    
    def save(self, filename: str):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename: str):
        """모델 로드"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
