import pipeline.htmlserver
import pipeline.ntables
from pipeline.visionmanager import VisionManager
import argparse
import platform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    args = parser.parse_args()

    config_file = args.config
    if config_file is None:
        if args.dev or platform.system() in ["Windows", "Darwin"]:
            config_file = "config_dev.json"
        else:
            config_file = "config.json"

    print(f"Using config: {config_file}")
    vision_manager = VisionManager(config_file)
    server = pipeline.htmlserver.DashboardServer(vision_manager)
    vision_manager.execute()
