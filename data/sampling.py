import random
import torch

def sample_meta_task(class_dict, N_way, K_shot, Q_query, device):
    """
    Sample a meta-learning task (episode) from the class dictionary.
    
    Args:
        class_dict (dict):{class: [imágenes]}
        N_way (int): Number of classes per episode
        K_shot (int): Number of support examples per class
        Q_query (int): Number of query examples per class
        device (torch.device)
        
    Returns:
        tuple: ((sx, sy), (qx, qy)) where:
            - sx: Support images [N_way*K_shot, 3, H, W]
            - sy: Support labels [N_way*K_shot]
            - qx: Query images [N_way*Q_query, 3, H, W]
            - qy: Query labels [N_way*Q_query]
    """

    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    
    for i, c in enumerate(chosen):
        imgs = class_dict[c]
        idxs = random.sample(range(len(imgs)), K_shot+Q_query)
        
        for j in idxs[:K_shot]:
            sx.append(imgs[j])
            sy.append(i)
        
        for j in idxs[K_shot:]:
            qx.append(imgs[j])
            qy.append(i)
    
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy, device=device)
    qy = torch.tensor(qy, device=device)
    
    return (sx, sy), (qx, qy)


def create_episode_batch(class_dict, N_way, K_shot, Q_query, device, batch_size=4):
    """
    Creates a batch of episodes for meta-learning.
    
    Args:
        class_dict (dict): {class: [imágenes]}
        N_way (int): Number of classes per episode
        K_shot (int): Number of support examples per class
        Q_query (int): Number of query examples per class
        device (torch.device)
        batch_size (int): number of episodes in the batch
        
    Returns:
        list: batch of episodes, each episode is a tuple of ((sx, sy), (qx, qy))
    """
    episodes = []
    for _ in range(batch_size):
        episodes.append(sample_meta_task(class_dict, N_way, K_shot, Q_query, device))
    return episodes