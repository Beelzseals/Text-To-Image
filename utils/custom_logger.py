import logging
import datetime
import os


def create_logger():
    os.makedirs("logs", exist_ok=True)
    current_day = datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(os.path.join("logs", current_day), exist_ok=True)
    logger = logging.getLogger("logger")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{current_day}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"),
            logging.StreamHandler(),
        ],
    )
    return logger
