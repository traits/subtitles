import torch

from processor import Processor

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA ON")
    else:
        print("CUDA OFF")

    processor = Processor()
    processor.run()
