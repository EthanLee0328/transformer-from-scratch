#!/usr/bin/env python3
"""L02–L03 的教学不变量；TODO=NOT_IMPLEMENTED 且 exit 2，绝不算通过。"""

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from student.attention import scaled_dot_product_attention as attention
except ImportError as exc:
    print(f"ENVIRONMENT_ERROR: {exc}", file=sys.stderr)
    sys.exit(3)


class AttentionInvariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_known_softmax_weighted_example(self):
        # 分数为 [0, ln(3)]，权重应是 [1/4, 3/4]，结果为 5。
        q = torch.tensor([[[[1.0]]]])
        k = torch.tensor([[[[0.0], [math.log(3.0)]]]])
        v = torch.tensor([[[[2.0], [6.0]]]])
        out, weights = attention(q, k, v)
        torch.testing.assert_close(weights, torch.tensor([[[[0.25, 0.75]]]]))
        torch.testing.assert_close(out, torch.tensor([[[[5.0]]]]))

    def test_scale_is_sqrt_key_dimension(self):
        q = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]])
        k = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [math.log(3), 0.0, 0.0, 0.0]]]])
        v = torch.tensor([[[[2.0], [6.0]]]])
        out, weights = attention(q, k, v)
        torch.testing.assert_close(weights, torch.tensor([[[[0.25, 0.75]]]]))
        torch.testing.assert_close(out, torch.tensor([[[[5.0]]]]))

    def test_shape_and_row_sums_cross_attention(self):
        q, k, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 7, 5), torch.randn(2, 3, 7, 6)
        out, weights = attention(q, k, v)
        self.assertEqual(out.shape, (2, 3, 4, 6))
        self.assertEqual(weights.shape, (2, 3, 4, 7))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 4))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_causal_future_keys_and_values_cannot_change_past(self):
        q, k, v = (torch.randn(2, 2, 5, 4) for _ in range(3))
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        original, weights = attention(q, k, v, allowed_mask=mask)
        altered_k, altered_v = k.clone(), v.clone()
        altered_k[:, :, 3:] = 100.0 * torch.randn_like(altered_k[:, :, 3:])
        altered_v[:, :, 3:] = 100.0 * torch.randn_like(altered_v[:, :, 3:])
        altered, _ = attention(q, altered_k, altered_v, allowed_mask=mask)
        torch.testing.assert_close(original[:, :, :3], altered[:, :, :3])
        torch.testing.assert_close(weights[..., ~mask], torch.zeros_like(weights[..., ~mask]))

    def test_padding_keys_do_not_affect_output(self):
        q, k, v = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 6)
        original, _ = attention(q, k, v)
        padded_k = torch.cat([k, 100.0 * torch.randn(2, 2, 2, 4)], dim=-2)
        padded_v = torch.cat([v, 100.0 * torch.randn(2, 2, 2, 6)], dim=-2)
        mask = torch.tensor([True, True, True, True, False, False])[None, None, None, :]
        padded, weights = attention(q, padded_k, padded_v, allowed_mask=mask)
        torch.testing.assert_close(original, padded)
        torch.testing.assert_close(weights[..., -2:], torch.zeros_like(weights[..., -2:]))

    def test_gradients_flow_to_q_k_v(self):
        tensors = [torch.randn(2, 2, 3, 4, requires_grad=True) for _ in range(3)]
        out, _ = attention(*tensors)
        out.square().mean().backward()
        for tensor in tensors:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_fully_masked_query_is_rejected(self):
        q = torch.ones(1, 1, 1, 2)
        with self.assertRaises(ValueError):
            attention(q, q, q, allowed_mask=torch.zeros(1, 1, dtype=torch.bool))


def main():
    try:
        attention(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))
    except NotImplementedError as exc:
        print(f"NOT_IMPLEMENTED: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # 其他错误留给测试具体报告，不能误认成未开始。
        pass
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AttentionInvariants)
    )
    print("CHECKS_PASS（仍需本人解释与迁移练习）" if result.wasSuccessful() else "CHECKS_FAILED")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
