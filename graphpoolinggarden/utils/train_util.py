import os
import torch
from typing import Any

def load_epoch(model_path: str, epoch: int) -> Any:
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(model_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')