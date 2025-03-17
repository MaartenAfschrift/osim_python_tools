
from datetime import datetime, timedelta

def compute_marathon_times(start_time, total_distance, total_time, keypoints):
    # Compute pace per km
    pace_per_km = total_time / total_distance  # in minutes per km

    # Convert start time to datetime object
    start_datetime = datetime.strptime(start_time, "%H:%M")

    # Compute and print times at keypoints
    print("\nMarathon Keypoint Times:")
    print("-----------------------")
    for km in keypoints:
        time_at_keypoint = start_datetime + timedelta(minutes=km * pace_per_km)
        print(f"At {km:.1f} km: {time_at_keypoint.strftime('%H:%M:%S')}")
    print(' pace per km', pace_per_km)

# Define inputs
start_time = "09:00"  # 9 AM start
total_distance = 42.2  # Marathon distance in km
total_time = 180  # Total time in minutes (3 hours)
keypoints = [0, 5, 11, 13.5, 20, 27.5, 32, 35.7, 37.5, 42.1]

# Compute and display times
compute_marathon_times(start_time, total_distance, total_time, keypoints)


