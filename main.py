import os
from utils.arguments import get_args
import torch
import numpy as np
from episode import Episode
os.environ["OMP_NUM_THREADS"] = "1"

def setup_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = get_args()
    setup_seed(args)

    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    # Starting environments
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    # Start navigation
    navigation_episodes = Episode(args)
    navigation_episodes.start()