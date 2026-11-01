import torch
from snake_detr import ContextAnchorAttention, GSConv, CAAGELAN, ADown, EFENet, PowerfulIoULoss

def test_modules():
    print("=" * 60)
    print("TESTING CUSTOM SNAKE-DETR MODULES IMPLEMENTATION")
    print("=" * 60)
    
    # 1. Test Context Anchor Attention (CAA)
    print("\n[1/6] Testing Context Anchor Attention (CAA)...")
    caa = ContextAnchorAttention(channels=64, kb=7)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    out = caa(x)
    assert out.shape == x.shape, f"CAA shape mismatch: {out.shape} vs {x.shape}"
    out.sum().backward()
    print(f"  [OK] CAA working! Input: {list(x.shape)} -> Output: {list(out.shape)}")

    # 2. Test Group-Shuffle Convolution (GSConv)
    print("\n[2/6] Testing Group-Shuffle Convolution (GSConv)...")
    gsconv = GSConv(c1=64, c2=128, k=1)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    out = gsconv(x)
    assert out.shape == (2, 128, 32, 32), f"GSConv shape mismatch: {out.shape}"
    out.sum().backward()
    print(f"  [OK] GSConv working! Input: {list(x.shape)} -> Output: {list(out.shape)}")

    # 3. Test CAA-GELAN
    print("\n[3/6] Testing CAA-GELAN (Context Anchor GELAN)...")
    caagelan = CAAGELAN(c1=64, c2=128)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    out = caagelan(x)
    assert out.shape == (2, 128, 32, 32), f"CAA-GELAN shape mismatch: {out.shape}"
    out.sum().backward()
    print(f"  [OK] CAA-GELAN working! Input: {list(x.shape)} -> Output: {list(out.shape)}")

    # 4. Test ADown (Asymmetric Downsampling)
    print("\n[4/6] Testing ADown (Asymmetric Downsampler)...")
    adown = ADown(c1=64, c2=128)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    out = adown(x)
    assert out.shape == (2, 128, 16, 16), f"ADown shape mismatch: {out.shape}"
    out.sum().backward()
    print(f"  [OK] ADown working! Input: {list(x.shape)} -> Output: {list(out.shape)} (Spatial halved)")

    # 5. Test EFENet Backbone
    print("\n[5/6] Testing EFENet Full Backbone Network...")
    efenet = EFENet(in_channels=3)
    img = torch.randn(1, 3, 640, 640)
    s3, s4, s5 = efenet(img)
    total_params = sum(p.numel() for p in efenet.parameters())
    print(f"  [OK] EFENet Hierarchical Outputs:")
    print(f"       - S3 stage: {list(s3.shape)} (128 channels)")
    print(f"       - S4 stage: {list(s4.shape)} (256 channels)")
    print(f"       - S5 stage: {list(s5.shape)} (512 channels)")
    print(f"       - Total Backbone Parameters: {total_params / 1e6:.2f} Million")

    # 6. Test Powerful-IoU (PIoU) Loss
    print("\n[6/6] Testing Powerful-IoU (PIoU) Loss Function...")
    piou_loss = PowerfulIoULoss()
    pred_boxes = torch.tensor([[50.0, 50.0, 150.0, 150.0], [20.0, 30.0, 80.0, 90.0]], requires_grad=True)
    target_boxes = torch.tensor([[55.0, 48.0, 152.0, 145.0], [25.0, 35.0, 75.0, 85.0]])
    loss = piou_loss(pred_boxes, target_boxes)
    loss.backward()
    assert loss.item() > 0, "Loss should be positive"
    print(f"  [OK] PIoU Loss calculated successfully: {loss.item():.4f}")
    print(f"  [OK] Gradients computed successfully for box regression backprop!")

    print("\n" + "=" * 60)
    print("ALL 5 COMPONENTS OF SNAKE-DETR VERIFIED AND WORKING!")
    print("=" * 60)

if __name__ == "__main__":
    test_modules()
