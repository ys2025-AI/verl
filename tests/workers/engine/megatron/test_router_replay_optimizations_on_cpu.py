# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the R3 optimization round (runbook section 14).

Covers:
- uint8 route storage dtype selection (capture + replay targets)
- native-rows-only topk in RouterReplay.get_replay_topk
- prepare_r3_router_replay_data single-pass align+mask (incl. unrecorded rows)
- multi-turn route merge
- TransferQueue field projection for the reference policy
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils.megatron import router_replay_patch as rr_patch  # noqa: E402
from verl.utils.megatron import router_replay_utils as rr_utils  # noqa: E402


# --------------------------------------------------------------------------- #
# uint8 storage dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_experts,expected",
    [(8, torch.uint8), (256, torch.uint8), (257, torch.int16), (32768, None), (None, None)],
)
def test_replay_storage_dtype(num_experts, expected):
    assert rr_patch._replay_storage_dtype(num_experts) is expected


def test_replay_index_dtype_prefers_uint8(monkeypatch):
    assert rr_utils._replay_index_dtype(SimpleNamespace(num_moe_experts=256)) is torch.uint8
    assert rr_utils._replay_index_dtype(SimpleNamespace(num_moe_experts=128)) is torch.uint8
    assert rr_utils._replay_index_dtype(SimpleNamespace(num_moe_experts=300)) is torch.int16
    assert rr_utils._replay_index_dtype(SimpleNamespace(num_moe_experts=40000)) is torch.int64
    assert rr_utils._replay_index_dtype(SimpleNamespace()) is torch.int64


def test_record_indices_uses_uint8_for_256_experts():
    router = rr_patch.RouterReplay()
    indices = torch.randint(0, 256, (4, 8), dtype=torch.int64)
    router.record_indices(indices, num_experts=256)
    assert router.recorded_topk_idx.dtype == torch.uint8
    assert torch.equal(router.recorded_topk_idx.long(), indices)


# --------------------------------------------------------------------------- #
# native-rows-only topk
# --------------------------------------------------------------------------- #
def _native_topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, k=topk, dim=1)


def _run_replay(router, scores, topk, mask):
    router.set_target_indices(router._test_indices, replay_mask=mask)
    router.set_router_replay_action(rr_patch.RouterReplayAction.REPLAY_FORWARD)
    probs, out = router.get_replay_topk(scores, topk, default_compute_topk=_native_topk)
    assert torch.equal(probs, scores.gather(1, out))
    return out


def test_get_replay_topk_subset_native_matches_full_reference():
    torch.manual_seed(0)
    rows, experts, topk = 6, 8, 2
    scores = torch.randn(rows, experts)
    replay_targets = torch.randint(0, experts, (rows, topk))
    ref = torch.topk(scores, k=topk, dim=1).indices

    # mask True => replay; False => native. Only rows 0 and 3 stay native.
    mask = torch.tensor([False, True, True, False, True, True])
    router = rr_patch.RouterReplay()
    router._test_indices = replay_targets
    out = _run_replay(router, scores, topk, mask)

    assert torch.equal(out[0], ref[0])
    assert torch.equal(out[3], ref[3])
    assert torch.equal(out[1], replay_targets[1].long())
    assert torch.equal(out[2], replay_targets[2].long())


def test_get_replay_topk_all_native_and_all_replay():
    torch.manual_seed(1)
    rows, experts, topk = 4, 8, 2
    scores = torch.randn(rows, experts)
    replay_targets = torch.randint(0, experts, (rows, topk))
    ref = torch.topk(scores, k=topk, dim=1).indices

    router = rr_patch.RouterReplay()
    router._test_indices = replay_targets
    out_all_native = _run_replay(router, scores, topk, torch.zeros(rows, dtype=torch.bool))
    assert torch.equal(out_all_native, ref)

    router2 = rr_patch.RouterReplay()
    router2._test_indices = replay_targets
    out_all_replay = _run_replay(router2, scores, topk, torch.ones(rows, dtype=torch.bool))
    assert torch.equal(out_all_replay, replay_targets.long())


def test_get_replay_topk_group_routing_falls_back_to_full_scores():
    """group_limited_topk depends on the original token count, so keep the full path."""
    torch.manual_seed(2)
    rows, experts, topk = 4, 8, 2
    scores = torch.randn(rows, experts)
    replay_targets = torch.randint(0, experts, (rows, topk))
    mask = torch.tensor([False, True, True, True])

    calls = {"n": 0}

    def fake_group_topk(scores, topk, num_groups=None, group_topk=None):
        calls["n"] += 1
        assert scores.shape[0] == rows, "group routing must see the full score matrix"
        return torch.topk(scores, k=topk, dim=1)

    router = rr_patch.RouterReplay()
    router.set_target_indices(replay_targets, replay_mask=mask)
    router.set_router_replay_action(rr_patch.RouterReplayAction.REPLAY_FORWARD)
    router.get_replay_topk(scores, topk, num_groups=4, group_topk=1, default_compute_topk=fake_group_topk)
    assert calls["n"] == 1


def test_get_replay_topk_misaligned_rows_raise():
    router = rr_patch.RouterReplay()
    router.set_target_indices(torch.zeros(5, 2, dtype=torch.int64), replay_mask=torch.ones(5, dtype=torch.bool))
    router.set_router_replay_action(rr_patch.RouterReplayAction.REPLAY_FORWARD)
    with pytest.raises(RuntimeError, match="not aligned"):
        router.get_replay_topk(torch.randn(4, 8), 2, default_compute_topk=_native_topk)


# --------------------------------------------------------------------------- #
# prepare_r3_router_replay_data (single pass align + mask)
# --------------------------------------------------------------------------- #
def _nested_routes(parts):
    lengths = torch.tensor([part.shape[0] for part in parts], dtype=torch.long)
    offsets = torch.zeros(len(parts) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    return torch.nested.nested_tensor_from_jagged(torch.cat(parts, dim=0), offsets=offsets)


def test_prepare_r3_router_replay_data_align_and_mask():
    # seq0: prompt 2 + response 3 = 5 rows, routes recorded for 4 rows (one missing tail)
    # seq1: 4 rows, all recorded
    lens = [5, 4]
    route0 = torch.randint(1, 16, (4, 2, 2), dtype=torch.int16)  # non-zero => recorded
    route1 = torch.randint(1, 16, (4, 2, 2), dtype=torch.int16)
    routes = _nested_routes([route0, route1])
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(sum(lens), dtype=torch.int64), offsets=torch.tensor([0, lens[0], sum(lens)])
    )
    response_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])  # dense response masks

    aligned, mask = rr_utils.prepare_r3_router_replay_data(routes, input_ids, response_mask)

    assert aligned.offsets().diff().tolist() == lens
    aligned_parts = aligned.unbind()
    assert torch.equal(aligned_parts[0][:4], route0)
    assert torch.equal(aligned_parts[0][4], torch.zeros_like(route0[0]))
    assert torch.equal(aligned_parts[1], route1)

    mask_parts = mask.unbind()
    # seq0: rows 0..3 replay (incl. placeholder? no: placeholder row is all-zero -> native),
    # last row native. Row 4 is the zero placeholder -> masked native.
    assert mask_parts[0].tolist() == [True, True, True, True, False]
    # seq1: rows 0..2 replay, last row native
    assert mask_parts[1].tolist() == [True, True, True, False]


def test_prepare_r3_router_replay_data_unrecorded_rows_go_native():
    """Multi-turn / prefix-cached rows come back all-zero and must not replay expert 0."""
    lens = [6]
    routes_part = torch.randint(1, 16, (6, 2, 2), dtype=torch.int16)
    routes_part[1] = 0  # row 1 not recorded (prefix cached)
    routes_part[4] = 0  # row 4 not recorded (tool turn)
    routes = _nested_routes([routes_part])
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(6, dtype=torch.int64), offsets=torch.tensor([0, 6])
    )
    response_mask = torch.ones(1, 6, dtype=torch.int64)

    _, mask = rr_utils.prepare_r3_router_replay_data(routes, input_ids, response_mask)
    assert mask.unbind()[0].tolist() == [True, False, True, True, False, False]


def test_prepare_r3_router_replay_data_rejects_bad_route_len():
    routes = _nested_routes([torch.ones(2, 2, 2, dtype=torch.int16)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.zeros(6, dtype=torch.int64), offsets=torch.tensor([0, 6]))
    with pytest.raises(RuntimeError, match="route rows"):
        rr_utils.prepare_r3_router_replay_data(routes, input_ids, torch.ones(1, 6, dtype=torch.int64))


def test_build_r3_replay_mask_without_routes_keeps_legacy_semantics():
    lens = [5, 4]
    input_ids = torch.nested.nested_tensor_from_jagged(
        torch.zeros(sum(lens), dtype=torch.int64), offsets=torch.tensor([0, lens[0], sum(lens)])
    )
    response_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    mask = rr_utils.build_r3_replay_mask(input_ids, response_mask)
    parts = mask.unbind()
    assert parts[0].tolist() == [True, True, True, True, False]
    assert parts[1].tolist() == [True, True, True, False]


# --------------------------------------------------------------------------- #
# multi-turn route merge
# --------------------------------------------------------------------------- #
def test_merge_turn_routed_experts_keeps_previous_rows():
    from verl.experimental.agent_loop.tool_agent_loop import merge_turn_routed_experts

    turn0 = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.uint8)  # 3 rows, row2 unrecorded
    turn1 = torch.tensor([[0, 0], [0, 0], [5, 6], [7, 8]], dtype=torch.uint8)  # rows 0-1 cached
    merged = merge_turn_routed_experts(turn0, turn1)
    assert merged.tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_merge_turn_routed_experts_first_turn():
    from verl.experimental.agent_loop.tool_agent_loop import merge_turn_routed_experts

    turn0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    assert merge_turn_routed_experts(None, turn0) is turn0


# --------------------------------------------------------------------------- #
# TransferQueue field projection for the reference policy
# --------------------------------------------------------------------------- #
def test_project_batch_fields_excludes_routes():
    from transfer_queue import KVBatchMeta

    from verl.trainer.ppo.v1.trainer_base import _project_batch_fields

    meta = KVBatchMeta(
        keys=["k0", "k1"],
        tags=[{}, {}],
        partition_id="train",
        fields=["input_ids", "routed_experts", "responses"],
        extra_info={"global_steps": 1},
    )
    projected = _project_batch_fields(meta, exclude={"routed_experts"})
    assert projected.fields == ["input_ids", "responses"]
    assert projected.keys == meta.keys and projected.extra_info == meta.extra_info
    # untouched when the field is absent
    assert _project_batch_fields(projected, exclude={"routed_experts"}) is projected
    # cannot project when fields are unset (= fetch everything)
    meta_all = KVBatchMeta(keys=["k"], tags=[{}], partition_id="train", fields=None, extra_info={})
    with patch("verl.trainer.ppo.v1.trainer_base.KVBatchMeta", KVBatchMeta):
        assert _project_batch_fields(meta_all, exclude={"routed_experts"}) is meta_all
