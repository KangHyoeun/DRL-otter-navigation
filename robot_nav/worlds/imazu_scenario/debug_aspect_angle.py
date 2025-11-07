import numpy as np
import math
from colregs_core.utils.utils import WrapTo180, WrapTo360, distance, dist_hypot
from colregs_core import (
    EncounterClassifier,
    RiskAssessment,
    heading_speed_to_velocity,
    EncounterType
)
import yaml

# Case 01 Data
os_position = (-90.0, 0.0)
ts_position = (90.0, 0.0)
ts_heading = 180.0

print("=" * 60)
print("Case 01: Aspect Angle Debug")
print("=" * 60)

# Step-by-step calculation
print(f"\n📍 Positions:")
print(f"   OS: {os_position}")
print(f"   TS: {ts_position}")
print(f"   TS Heading: {ts_heading}°")

# Calculate relative position (TS → OS)
dx = os_position[0] - ts_position[0]
dy = os_position[1] - ts_position[1]

print(f"\n🧮 Relative Position (TS → OS):")
print(f"   dx (North): {dx} m")
print(f"   dy (East): {dy} m")

# Calculate absolute bearing using atan2
angle_rad = np.arctan2(dy, dx)
angle_deg = np.degrees(angle_rad)

print(f"\n📐 Absolute Bearing (TS → OS):")
print(f"   Radians: {angle_rad:.10f}")
print(f"   Degrees: {angle_deg:.10f}")

# Calculate aspect angle
aspect_before_wrap = angle_deg - ts_heading
print(f"\n🔄 Aspect Angle Calculation:")
print(f"   absolute_bearing - ts_heading = {angle_deg:.10f} - {ts_heading}")
print(f"   aspect (before wrap) = {aspect_before_wrap:.10f}°")

# Apply WrapTo360
aspect_after_wrap = WrapTo360(aspect_before_wrap)
print(f"   aspect (after wrap) = {aspect_after_wrap:.10f}°")

# Test edge cases
print(f"\n🧪 WrapTo360 Tests:")
test_values = [0.0, -0.0, 360.0, -360.0, 0.00000001, -0.00000001, 359.99999999]
for val in test_values:
    result = WrapTo360(val)
    print(f"   WrapTo360({val:15.10f}) = {result:.10f}")

print("\n" + "=" * 60)
print("결론:")
print("=" * 60)

if abs(aspect_after_wrap) < 0.001:
    print("✅ Aspect Angle = 0° (정선수)")
    print("   TS에서 OS를 보면 정면에 있음")
elif abs(aspect_after_wrap - 360) < 0.001:
    print("✅ Aspect Angle = 360° (정선수)")
    print("   360° = 0° (수학적으로 동일)")
    print("   TS에서 OS를 보면 정면에 있음")
else:
    print(f"⚠️  예상치 못한 값: {aspect_after_wrap}°")

print("\n💡 Head-on 상황:")
print("   - Relative Bearing: 0° (OS → TS가 정면)")
print("   - Aspect Angle: 0° 또는 360° (TS → OS가 정면)")
print("   - Relative Course: -180° (반대 방향 항해)")
print("=" * 60)