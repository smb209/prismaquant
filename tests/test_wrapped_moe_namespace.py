"""Wrapped-MoE three-namespace fix — CPU-only, tiny synthetic, compile-off.

Pins the fix merged in PR #86 (commit 91139f4, "streaming export: bridge the
wrapped-MoE source's live namespace") against `export_nvfp4_cb_streaming`. A
wrapped-VLM (Qwen3.5/3.6-class) checkpoint has THREE spellings of the same
module in play at once — the checkpoint spelling stored in safetensors
(``model.language_model.layers.N.*``), the live/vLLM-internal spelling that
``profile.to_vllm_internal_name()`` produces and the per-expert regex matches
against (``language_model.model.layers.N.*``), and the recipe/canonical
spelling the allocator's assignment dict is keyed by
(``model.layers.N.*``). Before PR #86, a wrapped source's expert-group key,
its packed-stack tensor names, and its delegated quant-config targets each
leaked the wrong one of these three spellings; this module locks in that the
exporter now resolves each to the spelling its downstream consumer actually
needs. No GPU, no torch.compile (PRISMAQUANT_CB_ENCODE_COMPILE=0).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

os.environ["PRISMAQUANT_CB_ENCODE_COMPILE"] = "0"

import prismaquant.export_nvfp4_cb_streaming as streaming_mod  # noqa: E402
from prismaquant.export_nvfp4_cb_streaming import (  # noqa: E402
    _LazySkeleton,
    _plan_expert_stacks,
    export_nvfp4_cb_streaming as _export_nvfp4_cb_streaming,
)
from prismaquant.model_profiles import detect_profile  # noqa: E402

# Gridbook 0.9.1 / runtime-contract v12 (#113) added the CB route-status gate
# (campaign rule R3): the pinned lane-eligibility table names no CB cell at
# all on sm_121 (`serving_profile_specs/nvfp4_cb.json`'s declared platform),
# so this fixture's synthetic NVFP4_CB_K16 units resolve unattested there and
# the gate refuses. This module's export is about NAMESPACE bridging, not
# route attestation, and is CPU-only/never served, so it uses the same
# sanctioned declaration every other CB test fixture uses for this exact
# situation (`tests/cb_synthetic_target.py`): `PQ_CB_NON_NATIVE_TARGET`, "I do
# not target a native route here", not the route-status override (that one is
# a decision to ship past a real gap, which a test fixture is not making).
pytestmark = pytest.mark.usefixtures("synthetic_cb_target")


def export_nvfp4_cb_streaming(*args, **kwargs):
    """This module's synthetic direct calls are explicit research renders."""
    kwargs.setdefault("allow_unstamped_research", True)
    return _export_nvfp4_cb_streaming(*args, **kwargs)


@pytest.fixture
def workdir(tmp_path: Path):
    """Keep synthetic exports isolated and portable across CI runners."""
    return tmp_path


# --- synthetic wrapped-VLM skeleton ------------------------------------------
#
# hid=256, 2 experts, layer 0 — the smallest shape that still satisfies CB's
# in_features % 256 == 0 requirement. Every expert Linear is plain float32
# (no scale plane needed: CB does not require a native low-precision source),
# mirroring how `_dsv4_source_model` keeps its filler tensors simple.

_WRAPPED_HID = 256
_WRAPPED_EXPERTS = 2
_CB_RECIPE = {"data_type": "nvfp4_cb", "cb_k": 16}
_DELEGATED_RECIPE = {"data_type": "fp8_e4m3", "bits": 8}


def _wrapped_moe_source_model(mdl: Path, *, seed: int = 5) -> dict:
    """A Qwen3.5-class wrapped-VLM checkpoint at 1/1000 scale.

    Three module spellings are in play simultaneously for the very same
    expert group: this fixture writes the CHECKPOINT spelling
    (`model.language_model.layers.0.mlp.experts.*`) — the only one that
    exists on disk. The other two spellings (live / recipe) are never
    written anywhere; they are derived by the profile at export time, which
    is exactly the bridging PR #86 fixed.
    """
    mdl.mkdir(parents=True, exist_ok=True)
    hid = _WRAPPED_HID
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(_WRAPPED_EXPERTS):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.language_model.layers.0.mlp.experts.{expert}.{leaf}"
            tensors[base + ".weight"] = (
                torch.randn(hid, hid, generator=generator) * 0.05
            )
    # A delegated (non-expert) dense Linear, for closure (3).
    tensors["model.language_model.layers.0.self_attn.o_proj.weight"] = (
        torch.randn(hid, hid, generator=generator) * 0.05
    )
    # Harmless filler, uneventful outside the two units under test.
    tensors["model.language_model.norm.weight"] = torch.ones(
        hid, dtype=torch.bfloat16
    )
    save_file(tensors, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "hidden_size": hid,
        "intermediate_size": hid,
    }))
    return tensors


def _wrapped_moe_recipe():
    """RECIPE-spelled (canonical) per-tensor assignment, as the allocator
    writes it — `model.layers.0.*`, never the checkpoint or live spelling."""
    assignment: dict[str, object] = {}
    col_weights: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(11)
    for expert in range(_WRAPPED_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            qname = f"model.layers.0.mlp.experts.{expert}.{proj}"
            assignment[qname] = _CB_RECIPE
            col_weights[qname] = (
                torch.rand(_WRAPPED_HID, generator=generator) + 0.05
            )
    # A stock/delegated (non-"scheme") format — produces a config_groups
    # entry without a "scheme" key.
    assignment["model.layers.0.self_attn.o_proj"] = _DELEGATED_RECIPE
    return assignment, col_weights


def _export_wrapped_moe(workdir, monkeypatch, out_name="out"):
    """Build the synthetic source + recipe, run the export once, and capture
    the `expert_groups` dict actually handed to `_collapse_per_expert_assignment`
    (closure 1's own view of the world, not a value we recompute ourselves)."""
    mdl = workdir / "src"
    tensors = _wrapped_moe_source_model(mdl)
    assignment, col_weights = _wrapped_moe_recipe()
    path = workdir / f"{out_name}.json"
    path.write_text(json.dumps(assignment))
    out = workdir / out_name

    captured: dict[str, list[str]] = {}
    original = streaming_mod._collapse_per_expert_assignment

    def _spy(assignment_, expert_groups_, profile_):
        # The 2nd positional arg IS what fix (1) rekeyed (or failed to).
        captured["expert_groups_keys"] = list(expert_groups_.keys())
        return original(assignment_, expert_groups_, profile_)

    monkeypatch.setattr(streaming_mod, "_collapse_per_expert_assignment", _spy)

    counts = export_nvfp4_cb_streaming(
        mdl, path, out, col_weights, device="cpu",
        allow_route_pending_passthrough=True,
    )
    return mdl, out, tensors, dict(counts), captured


def test_wrapped_moe_three_namespaces_are_bridged(workdir, monkeypatch):
    """One export, three independently-commented regression guards — one per
    closure PR #86 (commit 91139f4) added to
    `prismaquant/export_nvfp4_cb_streaming.py`."""
    mdl, out, tensors, counts, captured = _export_wrapped_moe(workdir, monkeypatch)

    # --- documentation/contrast: the raw, un-rekeyed planner output --------
    # `_plan_expert_stacks` (export_nvfp4_cb_streaming.py:1479) keys each
    # expert group by whichever spelling matched
    # `profile.per_expert_moe_regex()` — for a wrapped Qwen3.5 source that is
    # the LIVE spelling, which nothing downstream (the allocator's
    # recipe-keyed assignment dict) can resolve. This is what would leak
    # through, uncorrected, without fix (1).
    profile = detect_profile(str(mdl))
    skeleton = _LazySkeleton(mdl)
    raw_groups = _plan_expert_stacks(skeleton, profile)
    assert list(raw_groups.keys()) == [
        "language_model.model.layers.0.mlp.experts"
    ]

    # --- closure (1): group-key normalization -------------------------------
    # export_nvfp4_cb_streaming.py:2939-3001 (the rekey block right after
    # `expert_groups = _plan_expert_stacks(...)`) rewrites any LIVE-spelled
    # group key to the RECIPE spelling before it is handed to
    # `_collapse_per_expert_assignment` (call site at line 3027). Guard: the
    # dict the exporter actually passed in is keyed by the RECIPE spelling,
    # not the raw live spelling captured above.
    assert captured["expert_groups_keys"] == ["model.layers.0.mlp.experts"]

    # --- closure (2): packed-stack tensor naming ----------------------------
    # The `_base_name` closure (export_nvfp4_cb_streaming.py:3184-3223) names
    # a packed-stack parent (not itself a checkpoint leaf) by the CHECKPOINT
    # prefix its own expert group carries (via `_canon_to_ckpt_prefix`,
    # built at 2961-2976), not the bare recipe or live spelling. Guard: the
    # emitted packed CB tensors are checkpoint-prefixed
    # (`model.language_model.layers.0.mlp.experts.*`), and no per-expert or
    # live-prefixed leaves exist for this group.
    emitted = set(load_file(str(out / "model.safetensors")))
    assert {
        "model.language_model.layers.0.mlp.experts.gate_up_proj.cb_qweight",
        "model.language_model.layers.0.mlp.experts.down_proj.cb_qweight",
    } <= emitted
    assert not any(
        name.startswith("language_model.model.layers.0.mlp.experts")
        for name in emitted
    )
    assert not any(
        name.startswith("model.layers.0.mlp.experts")
        for name in emitted
    )
    assert counts["NVFP4_CB_K16"] == 2          # gate_up + down, collapsed
                                                 # (6 per-expert leaves would
                                                 # mean the collapse never ran)

    # --- closure (3): delegated target namespace ----------------------------
    # `_delegated_target_name` (export_nvfp4_cb_streaming.py:5055-5088) takes
    # `profile.to_vllm_internal_name`'s output and mirrors two of the pinned
    # codebook runtime's own `gridbook/config.py::_candidate_bases` /
    # `_canonical_prefix` rewrite rules (`language_model.model.` -> `model.`,
    # bare `language_model.<rest>` -> `model.<rest>`) — NOT a bare
    # `language_model.` strip; #89 rewrote the closure from that bare strip
    # to this canonicalization after it was found to mis-handle names with no
    # `.model.` segment (e.g. `language_model.lm_head`). Deliberately a
    # narrower mirror than gridbook's own `_canonical_prefix`: that function
    # also lifts a bare `layers.` prefix (DSv4-class), which this closure
    # leaves alone on purpose (see the closure's own comment). The result
    # feeds `delegated_target_name=`/`source_target_name=` in the
    # `build_quant_config(...)` call (line 5393, kwargs at 5422-5423). Guard:
    # the delegated group's `targets` regex is tower-relative
    # (`model.layers.0.self_attn.o_proj`), never `language_model.`-prefixed.
    quant_config = json.loads((out / "quant_config.json").read_text())
    delegated_groups = [
        g for g in quant_config["config_groups"].values() if "scheme" not in g
    ]
    assert len(delegated_groups) == 1
    assert delegated_groups[0]["targets"] == [
        "re:^model[.]layers[.]0[.]self_attn[.]o_proj$"
    ]
    assert not any(
        t.startswith("re:^language_model") for t in delegated_groups[0]["targets"]
    )

    # Same fact as closure (2), from the quant_config side: the CB group's
    # targets are checkpoint-prefixed on purpose.
    cb_groups = [
        g for g in quant_config["config_groups"].values() if "scheme" in g
    ]
    assert len(cb_groups) == 1
    assert sorted(cb_groups[0]["targets"]) == [
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
    ]

    # Nothing silently dropped.
    assert quant_config["ignore"] == []
