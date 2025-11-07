"""
Imazu Problem 22 Scenarios Generator
6NM → 90m scale conversion for IR-SIM

Original: Own ship starts at (-6.0, 0.0) NM, heading 0° (North)
Scaled: Own ship starts at (-45, 0) m, heading 0° (North)
"""

import yaml
import math

# Scale factor: 6 NM → 90m
# 1 NM = 1852m, so 6 NM = 11112m
# Scale: 90 / 11112 = 0.0081
SCALE = 90.0 / 6.0

# Maritime heading to IR-SIM heading conversion
# Maritime: 0° = North, 90° = East, clockwise
# IR-SIM: 0° = East, 90° = North, counter-clockwise
def nautica_to_math(nautica_deg):
    """Convert maritime heading (0°=N, CW) to IR-SIM heading (0°=E, CCW)"""
    # Maritime 0° (North) → IR-SIM 90° (1.5708 rad)
    # Maritime 90° (East) → IR-SIM 0° (0 rad)
    # Maritime 180° (South) → IR-SIM 270° (-1.5708 rad)
    # Maritime -90° (West) → IR-SIM 180° (3.1416 rad)
    irsim_deg = 90 - nautica_deg
    irsim_rad = math.radians(irsim_deg)
    # Normalize to [-pi, pi]
    while irsim_rad > math.pi:
        irsim_rad -= 2 * math.pi
    while irsim_rad < -math.pi:
        irsim_rad += 2 * math.pi
    return irsim_rad

# Imazu Problem scenarios from Table 4
# Own ship: starts at (-6.0, 0.0) NM, heading 0° (North)
# Goal: (6.0, 0.0) NM (North along Y-axis)
imazu_scenarios = {
    1: [
        {"x": 6.000, "y": 0.000, "psi": 180.0, "u": 3.0}
    ],
    2: [
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0}
    ],
    3: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1}
    ],
    4: [
        {"x": -4.243, "y": -4.243, "psi": 45.0, "u": 2.121}
    ],
    5: [
        {"x": 6.000, "y": 0.000, "psi": 180.0, "u": 3.0},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0}
    ],
    6: [
        {"x": -5.909, "y": 1.042, "psi": -10.0, "u": 2.954},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    7: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    8: [
        {"x": 6.000, "y": 0.000, "psi": 180.0, "u": 3.0},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0}
    ],
    9: [
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.196, "y": 3.000, "psi": -30.0, "u": 2.598}
    ],
    10: [
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.796, "y": -1.553, "psi": 15.0, "u": 2.898}
    ],
    11: [
        {"x": 0.000, "y": -6.000, "psi": 90.0, "u": 3.0},
        {"x": -5.196, "y": 3.000, "psi": -30.0, "u": 2.598}
    ],
    12: [
        {"x": 6.000, "y": 0.000, "psi": 180.0, "u": 3.0},
        {"x": -5.909, "y": -1.042, "psi": 10.0, "u": 2.954},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    13: [
        {"x": 6.000, "y": 0.000, "psi": 180.0, "u": 3.0},
        {"x": -5.909, "y": -1.042, "psi": 10.0, "u": 2.954},
        {"x": -4.243, "y": -4.243, "psi": 45.0, "u": 2.121}
    ],
    14: [
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.909, "y": 1.042, "psi": -10.0, "u": 2.954},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    15: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    16: [
        {"x": 0.000, "y": -6.000, "psi": 90.0, "u": 3.0},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -2.970, "y": -2.970, "psi": 45.0, "u": 2.121}
    ],
    17: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1},
        {"x": -5.909, "y": -1.042, "psi": 10.0, "u": 2.954},
        {"x": -4.243, "y": 4.243, "psi": -45.0, "u": 2.121}
    ],
    18: [
        {"x": 4.243, "y": 4.243, "psi": -135.0, "u": 2.121},
        {"x": -5.796, "y": 1.553, "psi": -15.0, "u": 2.898},
        {"x": -5.196, "y": 3.000, "psi": -30.0, "u": 2.598}
    ],
    19: [
        {"x": 4.243, "y": 4.243, "psi": -135.0, "u": 2.121},
        {"x": -5.796, "y": -1.553, "psi": 15.0, "u": 2.898},
        {"x": -5.796, "y": 1.553, "psi": -15.0, "u": 2.898}
    ],
    20: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.796, "y": 1.553, "psi": -15.0, "u": 2.898}
    ],
    21: [
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.796, "y": -1.553, "psi": 15.0, "u": 2.898},
        {"x": -5.796, "y": 1.553, "psi": -15.0, "u": 2.898}
    ],
    22: [
        {"x": -4.200, "y": 0.000, "psi": 0.0, "u": 2.1},
        {"x": 0.000, "y": 6.000, "psi": -90.0, "u": 3.0},
        {"x": -5.196, "y": 3.000, "psi": -30.0, "u": 2.598}
    ]
}

# YAML template
yaml_template = """world:
  height: 200  # 200m maritime area (North-South)
  width: 200   # 100m maritime area (East-West)
  offset: [-100, -100]
  step_time: 0.1  # 20Hz simulation
  sample_time: 0.1  # 20Hz rendering
  collision_mode: 'reactive'  # Maritime collision avoidance

coordinate_system: maritime

robot:
    kinematics: {{name: 'otter_usv'}}
    shape: {{name: 'rectangle', length: 2.0, width: 1.08}}  # Otter USV dimensions
    # behavior: {{name: 'dash', reference_velocity: 3.0}}
    state: [-90, 0, 0, 0, 0, 0, 0, 0]  # Own ship starts at (-90, 0)m heading North
    goal: [0, 0, 0]  # Goal at (0, 90)m North
    vel_min: [-10.0, -10.0]
    vel_max: [10.0, 10.0]
    arrive_mode: position
    goal_threshold: 4.0

    sensors:
      - type: 'lidar2d'
        range_min: 0
        range_max: 100
        angle_range: 6.2832  # 360 degrees
        number: 360
        noise: True
        std: 0.08
        angle_std: 0.1
        offset: [0, 0, 3.14159]
        alpha: 0.3

    plot:
      show_arrow: true
      show_trajectory: true
      show_velocity_text: True

obstacle:
{obstacles}
  
  # World boundary
  - shape: {{name: 'linestring', vertices: [[-50, -100], [50, -100], [50, 100], [-50, 100], [-50, -100]]}}
    kinematics: {{name: 'static'}}
    coordinate_system: math
    unobstructed: False # 뚫고 나가는가? 아니요
    state: [0, 0, 0]
"""

def generate_obstacle_yaml(target_ships):
    """Generate obstacle YAML section for target ships"""
    obstacles = []
    for i, ship in enumerate(target_ships, 1):
        # Convert position
        x_m = ship["x"] * SCALE
        y_m = ship["y"] * SCALE
        psi_rad = math.radians(ship["psi"])
        
        # Calculate goal position (ship travels through collision point toward opposite direction)
        # Collision point is at origin (0, 0)
        # Goal is 2x distance beyond collision point
        goal_x = -x_m
        goal_y = -y_m
        
        obstacle_yaml = f"""  - number: {i}
    kinematics: {{name: 'otter_usv'}}
    behavior: {{name: 'dash', reference_velocity: {ship["u"]}}}
    vel_max: [10.0, 10.0]
    vel_min: [-10.0, -10.0]
    shape:
      - {{name: 'rectangle', length: 2.0, width: 1.08}}
    state: [{x_m:.3f}, {y_m:.3f}, {psi_rad:.4f}, 0, 0, 0, 0, 0]
    goal: [{goal_x:.3f}, {goal_y:.3f}, {psi_rad:.4f}]
    arrive_mode: position
    goal_threshold: 4.0
"""
        obstacles.append(obstacle_yaml)
    
    return "\n".join(obstacles)

def generate_all_scenarios():
    """Generate all 22 Imazu scenarios"""
    output_dir = "/home/hyo/DRL-otter-navigation/robot_nav/worlds/imazu_scenario"
    
    for case_num, target_ships in imazu_scenarios.items():
        # Generate obstacles
        obstacles_yaml = generate_obstacle_yaml(target_ships)
        
        # Fill template
        world_yaml = yaml_template.format(obstacles=obstacles_yaml)
        
        # Write to file
        filename = f"{output_dir}/imazu_case_{case_num:02d}.yaml"
        with open(filename, 'w') as f:
            f.write(world_yaml)
        
        print(f"✓ Generated Case {case_num:02d}: {len(target_ships)} target ship(s)")
        print(f"  File: {filename}")
        
        # Print target ship positions for verification
        for i, ship in enumerate(target_ships, 1):
            x_m = ship["x"] * SCALE
            y_m = ship["y"] * SCALE
            psi_rad = math.radians(ship["psi"])
            print(f"    Target {i}: ({x_m:.2f}, {y_m:.2f})m, heading {math.degrees(psi_rad):.1f}° IR-SIM")

if __name__ == "__main__":
    print("=" * 70)
    print("Imazu Problem 22 Scenarios Generator")
    print("=" * 70)
    print(f"Scale factor: 6 NM → 90m (ratio: {SCALE:.6f})")
    print(f"Own ship: (-90, 0)m → (0, 90)m")
    print("=" * 70)
    print()
    
    generate_all_scenarios()
    
    print()
    print("=" * 70)
    print("✓ All 22 scenarios generated successfully!")
    print("=" * 70)
