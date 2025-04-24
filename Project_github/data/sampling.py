import random
import torch

def sample_meta_task(class_dict, N_way, K_shot, Q_query, device):
    """
    Muestrea una tarea meta-learning (episodio) del diccionario de clases.
    
    Args:
        class_dict (dict): Diccionario {clase: [imágenes]}
        N_way (int): Número de clases por episodio
        K_shot (int): Número de ejemplos de soporte por clase
        Q_query (int): Número de ejemplos de consulta por clase
        device (torch.device): Dispositivo para alojar los tensores
        
    Returns:
        tuple: ((sx, sy), (qx, qy)) donde:
            - sx: Imágenes de soporte [N_way*K_shot, 3, H, W]
            - sy: Etiquetas de soporte [N_way*K_shot]
            - qx: Imágenes de consulta [N_way*Q_query, 3, H, W]
            - qy: Etiquetas de consulta [N_way*Q_query]
    """
    # Seleccionar N_way clases aleatoriamente
    chosen = random.sample(list(class_dict), N_way)
    sx, sy, qx, qy = [], [], [], []
    
    # Para cada clase, seleccionar ejemplos de soporte y consulta
    for i, c in enumerate(chosen):
        imgs = class_dict[c]
        idxs = random.sample(range(len(imgs)), K_shot+Q_query)
        
        # Ejemplos de soporte
        for j in idxs[:K_shot]:
            sx.append(imgs[j])
            sy.append(i)
        
        # Ejemplos de consulta
        for j in idxs[K_shot:]:
            qx.append(imgs[j])
            qy.append(i)
    
    # Convertir a tensores y mover a device
    sx = torch.stack(sx).to(device)
    qx = torch.stack(qx).to(device)
    sy = torch.tensor(sy, device=device)
    qy = torch.tensor(qy, device=device)
    
    return (sx, sy), (qx, qy)


def create_episode_batch(class_dict, N_way, K_shot, Q_query, device, batch_size=4):
    """
    Crea un lote de episodios para entrenamiento más eficiente.
    
    Args:
        class_dict (dict): Diccionario {clase: [imágenes]}
        N_way (int): Número de clases por episodio
        K_shot (int): Número de ejemplos de soporte por clase
        Q_query (int): Número de ejemplos de consulta por clase
        device (torch.device): Dispositivo para alojar los tensores
        batch_size (int): Número de episodios en el lote
        
    Returns:
        list: Batch de episodios
    """
    episodes = []
    for _ in range(batch_size):
        episodes.append(sample_meta_task(class_dict, N_way, K_shot, Q_query, device))
    return episodes