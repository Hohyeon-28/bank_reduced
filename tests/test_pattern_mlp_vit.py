import torch

from models.early_pattern_vit import PatternMLPViT, PatternMLPBlock
from models.early_vit import vit_small_patch16_224_depth12
from models.early_vit_partial_routing import router_vit_small_patch16_224_depth12_6_e256_partial_mlp


def tiny_model(**kwargs):
    pattern_bank = {
        "none": [0, 0, 0, 0],
        "half": [1, 0, 1, 0],
        "full": [1, 1, 1, 1],
    }
    return PatternMLPViT(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        depth=4,
        num_heads=4,
        num_classes=10,
        class_token=False,
        global_pool="avg",
        pattern_bank=pattern_bank,
        **kwargs,
    )


def test_all_zero_layer_skips_mlp_and_keeps_shape():
    block = PatternMLPBlock(dim=32, num_heads=4)
    calls = {"mlp": 0}
    original = block.mlp.forward

    def counted(x):
        calls["mlp"] += 1
        return original(x)

    block.mlp.forward = counted
    x = torch.randn(3, 5, 32, requires_grad=True)
    y = block(x, pattern_mask=torch.zeros(3))
    assert y.shape == x.shape
    assert calls["mlp"] == 0
    y.sum().backward()
    assert x.grad is not None


def test_attention_runs_even_when_mlp_skips():
    block = PatternMLPBlock(dim=32, num_heads=4)
    calls = {"attn": 0}
    original = block.attn.forward

    def counted(x):
        calls["attn"] += 1
        return original(x)

    block.attn.forward = counted
    x = torch.randn(2, 5, 32)
    block(x, pattern_mask=torch.zeros(2))
    assert calls["attn"] == 1


def test_fixed_pattern_routing_info_and_active_count():
    model = tiny_model(pattern_mode="fixed", fixed_pattern="half")
    logits, routing = model(torch.randn(2, 3, 32, 32), return_routing=True)
    assert logits.shape == (2, 10)
    assert routing["pattern_name"] == ["half", "half"]
    assert routing["active_mlp_count"] == [2.0, 2.0]
    assert routing["pattern_masks"] == [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]


def test_router_zero_logits_probabilities_are_uniform():
    model = tiny_model(pattern_mode="router", router_init="zero_logits")
    _, routing = model(torch.randn(2, 3, 32, 32), return_routing=True)
    probs = torch.tensor(routing["pattern_probs"])
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / 3.0), atol=1e-6)


def test_budget_loss_tracks_expected_active_mlp():
    model = tiny_model(pattern_mode="router", router_init="zero_logits")
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 2])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y) + model.get_budget_loss(2.0, 0.1)
    loss.backward()
    assert model.pattern_router.weight.grad is not None


def test_existing_models_still_forward():
    x = torch.randn(1, 3, 224, 224)
    vit = vit_small_patch16_224_depth12(num_classes=10).eval()
    partial = router_vit_small_patch16_224_depth12_6_e256_partial_mlp(num_classes=10).eval()
    with torch.no_grad():
        assert vit(x).shape == (1, 10)
        assert partial(x).shape == (1, 10)
