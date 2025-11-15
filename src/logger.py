import logging
import os
from datetime import datetime

log_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(log_directory, exist_ok=True)
filename = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
filepath = os.path.join(log_directory, filename)

logging.basicConfig(
    filename=filepath,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
