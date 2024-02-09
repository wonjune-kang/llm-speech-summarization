import random
import torch


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_batch_prompts(user_prompts):
    system_prompt = ""
    full_prompts = [
        f"{system_prompt}[|User|] {user_prompt.lower()}</s>[|Assistant|]"
        for user_prompt in user_prompts
    ]
    return full_prompts
