import torch
import sys
sys.path.append('.')
from tests.test_csd_synthetic_benchmark import _make_targets

def test_targets():
    bif_times = torch.tensor([100.0, 150.0])
    lens = torch.tensor([200, 200])
    device = torch.device('cpu')
    
    targets = _make_targets(bif_times, lens, device)
    print(targets.shape)
    print(targets[0, 90:105])
    print("Soft targets generated successfully!")

if __name__ == '__main__':
    test_targets()
