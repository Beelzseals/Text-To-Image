import logging
import datetime
import os

def create_logger():
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger('logger')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}.log'),
            ]
    )
    return logger