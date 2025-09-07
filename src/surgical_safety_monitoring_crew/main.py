#!/usr/bin/env python
import os
import sys
from surgical_safety_monitoring_crew.crew import SurgicalSafetyMonitoringCrewCrew
from dotenv import load_dotenv

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

# Load .env variables
load_dotenv()

def get_inputs():
    """
    Collect all inputs from environment variables
    """
    return {
        "video_feed_url": os.getenv("VIDEO_FEED_URL", "sample_value")
        
    }


def run():
    """
    Run the crew.
    """
    inputs = get_inputs()
    print(f"👉 Running crew with inputs:\n{inputs}\n")
    SurgicalSafetyMonitoringCrewCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = get_inputs()
    try:
        SurgicalSafetyMonitoringCrewCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        SurgicalSafetyMonitoringCrewCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = get_inputs()
    try:
        SurgicalSafetyMonitoringCrewCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
