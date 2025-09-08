import torch
import logging

# Setup logging config to log to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/athenaik/GRPO_test/run_log.txt"),  # Log file
        logging.StreamHandler()               # Console output
    ]
)

def main():
    if torch.cuda.is_available():
        logging.info("CUDA is available! Using GPU.")
        device = torch.device("cuda")
    else:
        logging.info("CUDA NOT available. Using CPU.")
        device = torch.device("cpu")

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    logging.info(f"Tensor on device: {x}")

    model = torch.nn.Linear(3, 1).to(device)
    y = model(x)
    logging.info(f"Model output: {y}")

if __name__ == "__main__":
    main()

