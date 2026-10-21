"""CPU-only accounting for the speculative Mamba verify scratch.

Adapted from Pennyroyal bbd9451b56: ``MambaPool`` allocates two per-draft-token
buffers beside the committed state (see ``memory_pool.py``):
``intermediate_ssm_state_cache``
``[num_layers, spec_slots + 1, draft_tokens, *temporal]`` and
``intermediate_conv_window_cache``, in either the deduplicated sliding-window or
the dense layout. The memory solver used to stand the whole
``mamba_cache_per_req`` (conv + SSM) in for the snapshots -- where it charged
them at all. It never charged the conv rollback windows that EVERY spec mode
allocates, and would also have mispriced them at ``draft * (K-1)`` columns
instead of the ``draft + K - 2`` the dedup layout physically stores. Where the
old solver did drop the scratch (ReplaySSM spec) it dropped all of it, windows
included -- under-reserving instead of over-reserving.

These tests pin the reserve to the buffers the pool really allocates: the SSM
term only for the modes that allocate it (dropped under ReplaySSM, where the
verify ring + cursors own rollback), the conv rollback windows (the accepted
conv-state history, which stays in EVERY spec mode) always, sized from the
real dtype / TP-sharded / PP-stage geometry plus the pool's padding slot.
``TestPoolAllocationMatchesHelpers`` builds actual ``MambaPool``s and reads the
allocated bytes, closing the loop between helper arithmetic and allocation. No
model names: everything comes from the cache params and the published spec/mamba
config, so an EAGLE-family draft and a DFLASH draft (both linear chains, topk 1)
or a KDA pool (dense windows) are just configurations.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.mem_cache import kv_cache_configurator as kcc_module
from sglang.srt.mem_cache import memory_pool as mp_module
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.runtime_context import get_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GB = 1 << 30
# Qwen3-Next-shaped linear layer: conv state (3072, K-1), SSM state
# (32, 128, 128), bf16 conv / fp32 SSM.
INTERMEDIATE_SIZE = 2048
N_GROUPS = 4
NUM_HEADS = 32
HEAD_DIM = 128
STATE_SIZE = 128
CONV_KERNEL = 4
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * STATE_SIZE  # 3072
WIN = CONV_KERNEL - 1  # the conv state's sliding-window axis is (dim, K-1)
SSM_ELEMS = (NUM_HEADS, HEAD_DIM, STATE_SIZE)
MAMBA_LAYERS = [0, 1, 2, 3, 8, 9, 10, 11]
N_LAYERS = len(MAMBA_LAYERS)
DRAFT = 5
BUDGET_GB = 24.0
MAMBA_MEM_RATIO = 0.5
MAX_RUNNING = 8
SPEC_SLOTS = 12  # spec_state_size handed to MambaPool in the allocation tests
RECORD_LEN = 16  # linear_replayssm_cache_len pinned in solve()/_pool()


# --- per-spec-slot byte expectations, written out from the pool's shapes -------
def ssm_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=4, elems=SSM_ELEMS):
    return elems[0] * elems[1] * elems[2] * itemsize * draft * layers


def conv_dedup_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=2, conv_dim=CONV_DIM):
    return conv_dim * (draft + WIN - 1) * itemsize * layers


def conv_dense_charge(draft=DRAFT, *, layers=N_LAYERS, itemsize=2, conv_dim=CONV_DIM):
    return conv_dim * WIN * draft * itemsize * layers


def _params(layers=MAMBA_LAYERS, *, tp=1, ssm_dtype=torch.float32):
    shape = Mamba2StateShape.create(
        tp_world_size=tp,
        intermediate_size=INTERMEDIATE_SIZE,
        n_groups=N_GROUPS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE_SIZE,
        conv_kernel=CONV_KERNEL,
    )
    return Mamba2CacheParams(shape=shape, dtype=_dtype(ssm_dtype), layers=list(layers))


def _dtype(ssm_dtype=torch.float32):
    return Mamba2StateDType(conv=torch.bfloat16, temporal=ssm_dtype)


def _kda_params(layers=MAMBA_LAYERS):
    shape = KimiLinearStateShape.create(
        tp_world_size=1,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_k_heads=8,
        head_k_dim=STATE_SIZE,
        conv_kernel_size=CONV_KERNEL,
    )
    return KimiLinearCacheParams(shape=shape, dtype=_dtype(), layers=list(layers))


def _kda_conv_dense_charge(params, draft=DRAFT):
    return (
        sum(c[0] * c[1] for c in params.shape.conv)
        * draft
        * params.dtype.conv.itemsize
        * len(params.layers)
    )


def _autofit(
    budget_gb,
    resident,
    intermediate,
    ratio,
    max_running,
    dp_size=1,
    scratch_slots=1,
    fixed=0,
):
    """Mirror of the solver's joint equation:
      (K + 1) * resident + (K / ratio + scratch_slots) * intermediate
        + fixed = mamba budget
    ``scratch_slots`` is the pool's padding slot (1) plus any PD-decode
    pre-alloc slots; ``fixed`` is the GDN ReplaySSM request-lifetime ring.
    Returns the solved slot count and the memory left for the KV pool."""
    budget_bytes = budget_gb * MAMBA_MEM_RATIO / (1 + MAMBA_MEM_RATIO) * GB
    slots = int(
        (budget_bytes - fixed - resident - scratch_slots * intermediate)
        // (resident + intermediate / ratio)
    )
    capped = min(max_running // dp_size, slots // ratio)
    rest = (
        budget_gb
        - (slots + 1) * resident / GB
        - fixed / GB
        - (capped + scratch_slots) * intermediate / GB
    )
    return slots, rest


class _Solver:
    """Run ``_handle_max_mamba_cache`` over a published configuration."""

    def __init__(self, params, *, pp_size=1, pp_layers=None):
        self.params = params
        self.ps_pp_size = pp_size
        # Layer ids the PP split is computed over (params.layers by default, i.e.
        # every mamba layer of the model -- the rank then charges its own slice).
        self.pp_layers = pp_layers

    def solve(
        self,
        *,
        spec=None,
        topk=None,
        adaptive=False,
        replayssm_spec=False,
        unified=False,
        disagg="null",
        extra_slots=None,
        max_mamba_cache_size=None,
        max_running_requests=MAX_RUNNING,
        disable_radix=False,
        budget_gb=BUDGET_GB,
        dp_size=1,
        cuda=True,
    ):
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=self.params),
            extra_mamba_cache_bytes_per_req=0,
            fused_entry_bytes=lambda sub_pool_name: None,
            spec_algorithm=SimpleNamespace(is_none=lambda: spec is None),
            attn_dp_size=dp_size,
            pp_size=self.ps_pp_size,
            # A GDN (non-KDA) linear-attn target: the shape's own flags, not a
            # model name, decide the KDA/unified differences under test.
            hybrid_gdn_config=object(),
            hybrid_kda_config=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(),
                num_hidden_layers=max(self.pp_layers or self.params.layers) + 1,
            ),
        )
        # The real ratio calculation (it reads only the published config bags);
        # SimpleNamespace does not inherit the class method.
        fake._calculate_mamba_ratio = lambda: (
            KVCacheConfigurator._calculate_mamba_ratio(fake)
        )
        with (
            rc.get_context().override_server_args(
                disable_radix_cache=disable_radix,
                max_mamba_cache_size=max_mamba_cache_size,
                max_running_requests=max_running_requests,
                mamba_full_memory_ratio=MAMBA_MEM_RATIO,
                enable_linear_replayssm_spec=replayssm_spec,
                enable_linear_replayssm=False,
                linear_replayssm_cache_len=RECORD_LEN,
                speculative_num_draft_tokens=spec,
                speculative_eagle_topk=topk,
                speculative_adaptive=adaptive,
                enable_unified_memory=unified,
                disaggregation_mode=disagg,
                disaggregation_decode_extra_slots=extra_slots,
                mamba_radix_cache_strategy="no_buffer",
                disable_overlap_schedule=False,
            ),
            # The dedup/dense layout choice must follow the platform flags
            # under test, not the machine the suite runs on.
            patch.object(kcc_module, "_is_cpu", not cuda),
            patch.object(kcc_module, "_is_npu", False),
        ):
            rest = KVCacheConfigurator._handle_max_mamba_cache(fake, budget_gb)
            ratio = KVCacheConfigurator._calculate_mamba_ratio(fake)
            return (
                rest,
                get_schedule().max_mamba_cache_size,
                ratio,
            )


class TestIntermediateBufferSizing(CustomTestCase):
    """The pure sizing helpers mirror the pool's allocation shapes."""

    def setUp(self):
        self.params = _params()

    def test_ssm_snapshots(self):
        self.assertEqual(
            self.params.intermediate_ssm_bytes_per_slot(DRAFT), ssm_charge()
        )
        # no draft window -> no snapshots (nothing to price in the solver)
        self.assertEqual(self.params.intermediate_ssm_bytes_per_slot(0), 0)

    def test_ssm_snapshots_follow_the_real_dtypes(self):
        bf16 = _params(ssm_dtype=torch.bfloat16)
        self.assertEqual(
            bf16.intermediate_ssm_bytes_per_slot(DRAFT), ssm_charge(itemsize=2)
        )

    def test_ssm_snapshots_follow_the_tp_shard(self):
        self.assertEqual(
            _params(tp=2).intermediate_ssm_bytes_per_slot(DRAFT),
            ssm_charge(elems=(NUM_HEADS // 2, HEAD_DIM, STATE_SIZE)),
        )

    def test_conv_windows_dedup_vs_dense(self):
        self.assertEqual(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            conv_dedup_charge(),
        )
        self.assertEqual(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
            conv_dense_charge(),
        )
        # the whole point of the dedup layout
        self.assertLess(
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            self.params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
        )
        for dedup in (True, False):
            self.assertEqual(
                self.params.intermediate_conv_window_bytes_per_slot(0, dedup=dedup), 0
            )

    def test_conv_windows_honor_layer_count_and_tp_shard(self):
        self.assertEqual(
            _params(tp=2).intermediate_conv_window_bytes_per_slot(DRAFT, dedup=True),
            conv_dedup_charge(conv_dim=CONV_DIM // 2),
        )
        self.assertEqual(
            _params(layers=[0]).intermediate_conv_window_bytes_per_slot(
                DRAFT, dedup=True
            ),
            conv_dedup_charge(layers=1),
        )

    def test_kda_windows_are_dense(self):
        # KDA stores conv state as (K-1, dim), so the overlapping view stays off.
        params = _kda_params()
        self.assertTrue(params.shape.disable_conv_window_dedup)
        self.assertTrue(params.is_kda)
        self.assertEqual(
            params.intermediate_conv_window_bytes_per_slot(DRAFT, dedup=False),
            _kda_conv_dense_charge(params),
        )


class TestPoolAllocationMatchesHelpers(CustomTestCase):
    """The helpers' arithmetic against real MambaPool allocations: bytes the
    pool really allocates for the verify scratch, read back off the tensors
    (the dedup conv windows report their PHYSICAL buffers, not the overlapping
    view's logical size)."""

    def _pool(self, *, replayssm, dedup, draft=DRAFT, params=None, topk=None):
        params = params or _params()
        with patch.object(mp_module, "_is_cpu", not dedup), patch.object(
            mp_module, "_is_npu", False
        ):
            return MambaPool(
                size=32,
                spec_state_size=SPEC_SLOTS,
                cache_params=params,
                mamba_layer_ids=list(params.layers),
                device="cpu",
                speculative_num_draft_tokens=draft,
                speculative_eagle_topk=topk,
                linear_replayssm_cache_len=RECORD_LEN,
                enable_linear_replayssm_spec=replayssm,
            )

    def _intermediate_bytes(self, pool):
        ssm = pool.mamba_cache.intermediate_ssm
        return (
            mp_module.get_tensor_size_bytes(ssm) if ssm is not None else 0
        ) + mp_module.get_tensor_size_bytes(pool._intermediate_conv_window_phys)

    def test_plain_spec_allocates_snapshots_and_dedup_windows(self):
        pool = self._pool(replayssm=False, dedup=True)
        self.assertEqual(
            self._intermediate_bytes(pool),
            (SPEC_SLOTS + 1) * (ssm_charge() + conv_dedup_charge()),
        )

    def test_replayssm_keeps_windows_only(self):
        pool = self._pool(replayssm=True, dedup=True)
        self.assertIsNone(pool.mamba_cache.intermediate_ssm)
        self.assertEqual(
            self._intermediate_bytes(pool),
            (SPEC_SLOTS + 1) * conv_dedup_charge(),
        )

    def test_dense_layout_allocates_dense_windows(self):
        pool = self._pool(replayssm=False, dedup=False)
        self.assertEqual(
            self._intermediate_bytes(pool),
            (SPEC_SLOTS + 1) * (ssm_charge() + conv_dense_charge()),
        )

    def test_kda_pool_allocates_dense_windows(self):
        params = _kda_params()
        pool = self._pool(replayssm=False, dedup=False, params=params)
        self.assertEqual(
            self._intermediate_bytes(pool),
            (SPEC_SLOTS + 1) * (ssm_charge() + _kda_conv_dense_charge(params)),
        )

    def test_tree_verify_pool_allocates_dense_windows(self):
        # topk > 1 forces the dense layout even with the CUDA platform flags.
        pool = self._pool(replayssm=False, dedup=True, topk=2)
        self.assertEqual(
            self._intermediate_bytes(pool),
            (SPEC_SLOTS + 1) * (ssm_charge() + conv_dense_charge()),
        )


class TestAutoFitSpecReserve(CustomTestCase):
    """The auto-fit branch: joint solve over resident state + verify scratch."""

    def setUp(self):
        self.params = _params()
        self.solver = _Solver(self.params)
        self.resident = self.params.mamba_cache_per_req

    def charged_intermediate(self, rest, slots, fixed=0):
        """Bytes the solver left for the per-draft-token scratch."""
        return round(
            (BUDGET_GB - rest - (slots + 1) * self.resident / GB - fixed / GB) * GB
        )

    def test_plain_spec_charges_snapshots_and_windows(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dedup_charge() + ssm_charge(),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_replayssm_charges_conv_windows_and_the_ring(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT, replayssm_spec=True)
        fixed = self._ring(fixed_only=True)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dedup_charge(),
            ratio,
            MAX_RUNNING,
            fixed=fixed,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def _ring(self, *, fixed_only, layers=N_LAYERS, scale=1):
        ring = self.params.replayssm_ring_bytes_per_req(record_len=RECORD_LEN)
        if fixed_only:
            # GDN replay is request-lifetime scratch: charged once at
            # (max_running + 1) requests, not per persistent slot.
            return int(ring * scale) * (MAX_RUNNING + 1)
        return int(ring * scale)

    def test_replayssm_frees_exactly_the_dropped_snapshots(self):
        """The base bug: ReplaySSM kept paying for snapshots its pool never holds.

        The reserve the auto-fit branch actually spends on the verify scratch is
        read back out of the returned budget; under ReplaySSM it must be exactly
        the conv rollback buffers (which stay allocated in every spec mode), and
        the freed bytes must come back as mamba state slots.
        """
        rep_rest, rep_slots, ratio = self.solver.solve(spec=DRAFT, replayssm_spec=True)
        fixed = self._ring(fixed_only=True)
        self.assertEqual(
            self.charged_intermediate(rep_rest, rep_slots, fixed=fixed),
            (min(MAX_RUNNING, rep_slots // ratio) + 1) * conv_dedup_charge(),
        )
        _full_rest, full_slots, _ = self.solver.solve(spec=DRAFT)
        self.assertGreater(rep_slots, full_slots)  # the saving goes to the pool

    def test_base_stand_in_over_reserved(self):
        """What the pre-fix reserve charged: mamba_cache_per_req * slots * draft."""
        legacy = self.resident * (MAX_RUNNING + 1) * DRAFT
        # Under ReplaySSM the pool only holds the conv rollback windows.
        correct = (MAX_RUNNING + 1) * conv_dedup_charge()
        self.assertGreater(legacy, 4 * correct)

    def test_tree_verify_falls_back_to_dense_windows(self):
        rest, slots, ratio = self.solver.solve(spec=DRAFT, topk=2)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dense_charge() + ssm_charge(),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_platform_without_dedup_charges_dense_windows(self):
        """The conv-window layout the charge uses follows the pool's platform flags."""
        # ReplaySSM drops the SSM snapshots so the reserve is purely the windows.
        cuda_rest, cuda_slots, ratio = self.solver.solve(
            spec=DRAFT, replayssm_spec=True
        )
        fixed = self._ring(fixed_only=True)
        self.assertEqual(
            self.charged_intermediate(cuda_rest, cuda_slots, fixed=fixed),
            (min(MAX_RUNNING, cuda_slots // ratio) + 1) * conv_dedup_charge(),
        )
        cpu_rest, cpu_slots, _ = self.solver.solve(
            spec=DRAFT, replayssm_spec=True, cuda=False
        )
        self.assertEqual(
            self.charged_intermediate(cpu_rest, cpu_slots, fixed=fixed),
            (min(MAX_RUNNING, cpu_slots // ratio) + 1) * conv_dense_charge(),
        )
        self.assertGreater(
            self.charged_intermediate(cpu_rest, cpu_slots, fixed=fixed),
            self.charged_intermediate(cuda_rest, cuda_slots, fixed=fixed),
        )

    def test_adaptive_draft_window_is_priced_at_its_bound(self):
        """Adaptive algorithms switch draft width at runtime, so the buffers --
        and the reserve -- are sized to the widest candidate window, not the
        flat draft-token count."""
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        _flat_rest, flat_slots, ratio = self.solver.solve(spec=2)
        adapt_rest, adapt_slots, _ = self.solver.solve(spec=2, adaptive=True)
        bound = max(resolve_candidate_steps_from_config(None)) + 1
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            conv_dedup_charge(bound) + ssm_charge(bound),
            ratio,
            MAX_RUNNING,
        )
        self.assertEqual(adapt_slots, want_slots)
        self.assertAlmostEqual(adapt_rest, want_rest, delta=1e-9)
        self.assertLess(adapt_slots, flat_slots)

    def test_no_spec_allocates_no_scratch(self):
        rest, slots, ratio = self.solver.solve(spec=None)
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, 0, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)
        # Without a draft window ReplaySSM has no snapshots to drop, but its
        # request-lifetime ring still eats budget as fixed bytes.
        rep_rest, rep_slots, _ = self.solver.solve(spec=None, replayssm_spec=True)
        fixed = self._ring(fixed_only=True)
        want_rep_slots, want_rep_rest = _autofit(
            BUDGET_GB, self.resident, 0, ratio, MAX_RUNNING, fixed=fixed
        )
        self.assertEqual(rep_slots, want_rep_slots)
        self.assertAlmostEqual(rep_rest, want_rep_rest, delta=1e-9)

    def test_prefill_only_server_allocates_no_scratch(self):
        """A PD prefill server builds its pool with no draft window at all."""
        rest, slots, ratio = self.solver.solve(spec=DRAFT, disagg="prefill")
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, 0, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_unified_pool_holds_snapshots_in_dense_layout(self):
        """UnifiedMambaPool has neither the ReplaySSM skip nor the dedup view."""
        rest, slots, ratio = self.solver.solve(spec=DRAFT, unified=True)
        intermediate = conv_dense_charge() + ssm_charge()
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, intermediate, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_unified_pool_prefill_server_still_holds_the_scratch(self):
        """PD prefill + --enable-unified-memory: the reserve must follow THAT pool.

        _init_unified_mamba_pools passes the draft count through unchanged (no
        prefill clamp) and the unified pool allocates both dense scratch buffers
        whenever it is set, so the shared-pool path keeps its reserve on a
        prefill server where the standalone pool (built with
        speculative_num_draft_tokens=None there, see _build_hybrid_req_pool)
        legitimately holds none.
        """
        intermediate = conv_dense_charge() + ssm_charge()
        rest_u, slots_u, ratio = self.solver.solve(
            spec=DRAFT, disagg="prefill", unified=True
        )
        want_slots, want_rest = _autofit(
            BUDGET_GB, self.resident, intermediate, ratio, MAX_RUNNING
        )
        self.assertEqual(slots_u, want_slots)
        self.assertAlmostEqual(rest_u, want_rest, delta=1e-9)
        padded = (min(MAX_RUNNING, slots_u // ratio) + 1) * intermediate
        self.assertEqual(self.charged_intermediate(rest_u, slots_u), padded)
        # The same PD role on the standalone pool reserves nothing at all, so the
        # whole padded amount is the exact difference the unified path owes.
        plain_rest, plain_slots, _ = self.solver.solve(spec=DRAFT, disagg="prefill")
        self.assertEqual(self.charged_intermediate(plain_rest, plain_slots), 0)
        self.assertEqual(
            self.charged_intermediate(rest_u, slots_u)
            - self.charged_intermediate(plain_rest, plain_slots),
            padded,
        )
        # And the unified path ignores the PD role entirely.
        null_rest, null_slots, _ = self.solver.solve(spec=DRAFT, unified=True)
        self.assertEqual((rest_u, slots_u), (null_rest, null_slots))
        # Same for the adaptive bound: the unified pool is built from the flat
        # field, so that -- not the candidate-table maximum -- is what gets priced.
        flat_rest, flat_slots, _ = self.solver.solve(spec=2, unified=True)
        adapt_rest, adapt_slots, _ = self.solver.solve(
            spec=2, adaptive=True, unified=True
        )
        self.assertEqual((adapt_rest, adapt_slots), (flat_rest, flat_slots))

    def test_pd_decode_charges_its_prealloc_scratch_slots(self):
        """HybridMambaDecodeReqToTokenPool keys the scratch by size + pre_alloc_size
        (see disaggregation/decode.py), so the in-transfer slots must be reserved
        too -- capped_reqs plus the padding slot alone underbudgets them.
        """
        extra = 16
        intermediate = conv_dedup_charge()  # ReplaySSM: windows only
        fixed = self._ring(fixed_only=True)
        rest, slots, ratio = self.solver.solve(
            spec=DRAFT, replayssm_spec=True, disagg="decode", extra_slots=extra
        )
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            self.resident,
            intermediate,
            ratio,
            MAX_RUNNING,
            scratch_slots=1 + extra,
            fixed=fixed,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)
        self.assertEqual(
            self.charged_intermediate(rest, slots, fixed=fixed),
            (min(MAX_RUNNING, slots // ratio) + 1 + extra) * intermediate,
        )

    def test_pd_decode_prealloc_slots_are_gate_isolated(self):
        extra = 16
        # Only the ordinary pool: the unified one is built with
        # mamba_spec_state_size=max_num_reqs (unified_memory_pool.py).
        uni_rest, uni_slots, uni_ratio = self.solver.solve(
            spec=DRAFT, disagg="decode", extra_slots=extra, unified=True
        )
        base_uni_rest, base_uni_slots, _ = self.solver.solve(spec=DRAFT, unified=True)
        self.assertEqual((uni_rest, uni_slots), (base_uni_rest, base_uni_slots))
        self.assertEqual(
            self.charged_intermediate(uni_rest, uni_slots),
            (min(MAX_RUNNING, uni_slots // uni_ratio) + 1)
            * (conv_dense_charge() + ssm_charge()),
        )
        # Only the decode role: published extra_slots stay inert otherwise ...
        other_rest, other_slots, _ = self.solver.solve(spec=DRAFT, extra_slots=extra)
        base_rest, base_slots, _ = self.solver.solve(spec=DRAFT)
        self.assertEqual((other_rest, other_slots), (base_rest, base_slots))
        # ... and there is no scratch to pad without spec decoding.
        zero_rest, zero_slots, _ = self.solver.solve(
            spec=None, disagg="decode", extra_slots=extra
        )
        self.assertEqual(self.charged_intermediate(zero_rest, zero_slots), 0)

    def test_replayssm_spec_drops_snapshots_and_charges_the_ring(self):
        """Our GDN spec ring is request-lifetime scratch (fixed bytes, not
        per-slot), so the solve charges it once while the conv rollback windows
        join the per-slot terms."""
        params = _params(layers=list(range(16)))
        solver = _Solver(params)
        rest, slots, ratio = solver.solve(spec=DRAFT, replayssm_spec=True)
        ring = params.replayssm_ring_bytes_per_req(record_len=RECORD_LEN)
        fixed = ring * (MAX_RUNNING + 1)
        intermediate = conv_dedup_charge(DRAFT, layers=16)
        want_slots, want_rest = _autofit(
            BUDGET_GB,
            params.mamba_cache_per_req,
            intermediate,
            ratio,
            MAX_RUNNING,
            fixed=fixed,
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)

    def test_kda_pool_charges_dense_windows_and_snapshots(self):
        params = _kda_params()
        solver = _Solver(params)
        rest, slots, ratio = solver.solve(spec=DRAFT)
        intermediate = _kda_conv_dense_charge(params) + ssm_charge()
        want_slots, want_rest = _autofit(
            BUDGET_GB, params.mamba_cache_per_req, intermediate, ratio, MAX_RUNNING
        )
        self.assertEqual(slots, want_slots)
        self.assertAlmostEqual(rest, want_rest, delta=1e-9)


class TestExplicitSizeReserve(CustomTestCase):
    """The explicit-max_mamba_cache_size and disabled-radix branches."""

    def setUp(self):
        self.params = _params()
        self.solver = _Solver(self.params)
        self.resident = self.params.mamba_cache_per_req
        self.ring = self.params.replayssm_ring_bytes_per_req(record_len=RECORD_LEN)

    def _rest(self, *, replayssm=False, size=64, spec=DRAFT, **kwargs):
        rest, slots, ratio = self.solver.solve(
            spec=spec, replayssm_spec=replayssm, max_mamba_cache_size=size, **kwargs
        )
        return rest, slots, ratio, size

    def test_explicit_size_charges_the_mode_scratch(self):
        fixed = self.ring * (MAX_RUNNING + 1)
        for replayssm, intermediate in (
            (True, conv_dedup_charge()),
            (False, conv_dedup_charge() + ssm_charge()),
        ):
            rest, _, ratio, size = self._rest(replayssm=replayssm)
            capped = min(MAX_RUNNING, size // ratio)
            self.assertAlmostEqual(
                rest,
                BUDGET_GB
                - (size + 1) * self.resident / GB
                - (fixed if replayssm else 0) / GB
                - (capped + 1) * intermediate / GB,
                delta=1e-9,
                msg=f"replayssm={replayssm}",
            )

    def test_explicit_size_replayssm_saving_is_the_snapshots(self):
        """The replayssm rest is the full-mode rest plus the freed snapshot
        bytes, net of the request-lifetime ring the mode adds."""
        rest_rep, _, ratio, size = self._rest(replayssm=True)
        rest_full, _, _, _ = self._rest(replayssm=False)
        fixed = self.ring * (MAX_RUNNING + 1)
        capped = min(MAX_RUNNING, size // ratio)
        self.assertAlmostEqual(
            rest_rep - rest_full,
            (capped + 1) * ssm_charge() / GB - fixed / GB,
            delta=1e-9,
        )

    def test_explicit_size_without_spec_charges_nothing_extra(self):
        rest, _, _, size = self._rest(spec=None)
        self.assertAlmostEqual(
            rest,
            BUDGET_GB - (size + 1) * self.resident / GB,
            delta=1e-9,
        )

    def test_explicit_size_pd_decode_prealloc_exact_bytes(self):
        """PD decode charges the pre-alloc scratch slots at their exact size."""
        extra, size = 16, 64
        per_slot = conv_dedup_charge()  # ReplaySSM: conv rollback windows only
        rest_dec, slots_dec, _, _ = self._rest(
            replayssm=True, size=size, disagg="decode", extra_slots=extra
        )
        rest_plain, slots_plain, _, _ = self._rest(replayssm=True, size=size)
        self.assertEqual(slots_dec, slots_plain)  # explicit size is not solved
        self.assertAlmostEqual(rest_plain - rest_dec, extra * per_slot / GB, delta=1e-9)

    def test_disabled_radix_pd_decode_prealloc_exact_bytes(self):
        extra, max_running = 16, 12
        per_slot = conv_dedup_charge() + ssm_charge()
        rest_dec, slots_dec, _ = self.solver.solve(
            spec=DRAFT,
            disable_radix=True,
            max_running_requests=max_running,
            disagg="decode",
            extra_slots=extra,
        )
        rest_plain, slots_plain, _ = self.solver.solve(
            spec=DRAFT, disable_radix=True, max_running_requests=max_running
        )
        self.assertEqual((slots_dec, slots_plain), (max_running, max_running))
        self.assertAlmostEqual(rest_plain - rest_dec, extra * per_slot / GB, delta=1e-9)

    def test_dp_shard_divides_the_explicit_pool(self):
        _, slots, _ = self.solver.solve(
            spec=DRAFT, max_mamba_cache_size=64, dp_size=2, max_running_requests=8
        )
        self.assertEqual(slots, 32)

    def test_disabled_radix_uses_the_request_sized_pool(self):
        rest, slots, _ = self.solver.solve(
            spec=DRAFT, replayssm_spec=True, disable_radix=True, max_running_requests=12
        )
        self.assertEqual(slots, 12)  # ratio is 1 with the radix cache off
        intermediate = conv_dedup_charge()
        fixed = self.ring * (12 + 1)
        self.assertAlmostEqual(
            rest,
            BUDGET_GB
            - (slots + 1) * self.resident / GB
            - fixed / GB
            - (slots + 1) * intermediate / GB,
            delta=1e-9,
        )
        # ReplaySSM still frees the snapshots here too
        full_rest, _, _ = self.solver.solve(
            spec=DRAFT, disable_radix=True, max_running_requests=12
        )
        self.assertAlmostEqual(
            rest - full_rest, (slots + 1) * ssm_charge() / GB - fixed / GB, delta=1e-9
        )


class TestPPStageScratch(CustomTestCase):
    """A PP rank holds its own layer slice, so it charges only that slice -- for
    the verify scratch too, not just the committed state."""

    # 24 layers, mamba on every other one: 12 state layers, 6 per stage.
    TOTAL_LAYERS: int = 24
    ALL_MAMBA_LAYERS: list = [i for i in range(TOTAL_LAYERS) if i % 2 == 0]

    def _solve(self, pp_rank, replayssm):
        params = _params(layers=self.ALL_MAMBA_LAYERS)
        solver = _Solver(params, pp_size=2, pp_layers=list(range(self.TOTAL_LAYERS)))
        rest, slots, ratio = solver.solve(
            spec=DRAFT, replayssm_spec=replayssm, max_running_requests=4
        )
        stage = sum(
            1
            for i in self.ALL_MAMBA_LAYERS
            if i in range(*get_pp_indices(self.TOTAL_LAYERS, pp_rank, 2))
        )
        scale = stage / len(self.ALL_MAMBA_LAYERS)
        resident = int(params.mamba_cache_per_req * scale)
        layers = len(self.ALL_MAMBA_LAYERS)
        intermediate = int(
            (
                conv_dedup_charge(layers=layers)
                + (0 if replayssm else ssm_charge(layers=layers))
            )
            * scale
        )
        fixed = (
            int(params.replayssm_ring_bytes_per_req(record_len=RECORD_LEN) * scale)
            * (4 + 1)
            if replayssm
            else 0
        )
        want_slots, want_rest = _autofit(
            BUDGET_GB, resident, intermediate, ratio, 4, fixed=fixed
        )
        return rest, slots, want_rest, want_slots

    def test_scratch_scales_with_the_stage(self):
        for replayssm in (False, True):
            rest, slots, want_rest, want_slots = self._solve(0, replayssm)
            self.assertEqual(slots, want_slots, f"replayssm={replayssm}")
            self.assertAlmostEqual(
                rest, want_rest, delta=1e-9, msg=f"replayssm={replayssm}"
            )

    def test_replayssm_grows_the_stage_pool(self):
        _, rep_slots, _, _ = self._solve(0, True)
        _, full_slots, _, _ = self._solve(0, False)
        self.assertGreater(rep_slots, full_slots)

    def test_both_ranks_agree(self):
        _, rank0_slots, _, _ = self._solve(0, False)
        _, rank1_slots, _, _ = self._solve(1, False)
        self.assertEqual(rank0_slots, rank1_slots)


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main([__file__, "-v"]))
