import torch
from bgsl.models.csd_observer import CSDKalmanObserver

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSDKalmanObserver(input_dim=2, latent_dim=4, lstm_head=True, lstm_dim=8).to(device)
    
    # B=2, T=50, input_dim=2
    x = torch.randn(2, 50, 2, device=device)
    mask = torch.ones(2, 50, 2, device=device)
    
    y_preds, logits, zs, A, K, C = model(x, mask)
    print(f"y_preds shape: {y_preds.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"zs shape: {zs.shape}")
    print(f"A shape: {A.shape}, K shape: {K.shape}, C shape: {C.shape}")
    
    # Dummy loss
    loss = y_preds.sum() + logits.sum() + zs.sum() + A.sum()
    loss.backward()
    print("Backward pass successful!")

if __name__ == '__main__':
    test()
