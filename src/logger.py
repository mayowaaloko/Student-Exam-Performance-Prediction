import logging
import os
from datetime import datetime


# 📂 1. Create a "logs" folder (if it doesn’t exist already)
# This keeps all your log files neatly in one place.
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 🕒 2. Generate a unique log filename based on current date & time
# Example: 09_23_2025_22_40_10.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# 📝 3. Configure logging settings
# - filename: where to save logs
# - format: how each log line looks
# - level: INFO means normal messages + warnings/errors will be captured
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# 🚀 4. When running this file directly, write a starter log
# This helps confirm logging is working right away
if __name__ == "__main__":
    logging.info("Logging has started")
