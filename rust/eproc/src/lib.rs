//! Zero-dependency Rust bindings for `eproc` (ABI v1).
//!
//! Compiles directly against the canonical zero-allocation C11 core (`csrc/`)
//! with deterministic IEEE-754 binary64 arithmetic (`-ffp-contract=off`) and
//! automatic runtime AVX2 dispatch on x86_64.

use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
use std::cell::UnsafeCell;
use std::ffi::c_int;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Raw C11 ABI v1 FFI declarations matching `csrc/eproc.h` byte-for-byte.
#[allow(non_camel_case_types)]
pub mod ffi {
    use std::ffi::c_int;

    pub const EPROC_ABI_VERSION: u32 = 1;

    pub const EPROC_OK: c_int = 0;
    pub const EPROC_ERR_INVALID: c_int = -1;
    pub const EPROC_ERR_CAPACITY: c_int = -2;
    pub const EPROC_ERR_EMPTY: c_int = -3;
    pub const EPROC_ERR_MISMATCH: c_int = -4;

    pub const EPROC_VERDICT_CONTINUE: c_int = 0;
    pub const EPROC_VERDICT_REJECT: c_int = 1;
    pub const EPROC_VERDICT_PROMOTE: c_int = 1;
    pub const EPROC_VERDICT_ROLLBACK: c_int = -1;

    pub const EPROC_SIDE_UPPER: i32 = 0;
    pub const EPROC_SIDE_LOWER: i32 = 1;
    pub const EPROC_SIDE_TWO_SIDED: i32 = 2;

    pub const EPROC_CS_EMPIRICAL_BERNSTEIN: i32 = 0;
    pub const EPROC_CS_NORMAL_MIXTURE: i32 = 1;

    pub const EPROC_MAX_ARMS: usize = 64;
    pub const EPROC_MAX_RATES: usize = 5;
    pub const EPROC_AGACI_EXPERTS: usize = 4;
    pub const EPROC_CONFORMAL_MAX_WIN: usize = 256;
    pub const EPROC_CMONOID_BINS: usize = 64;
    pub const EPROC_CMONOID_WIRE_SIZE: usize = 640;
    pub const EPROC_DELAY_MAX_PENDING: usize = 4096;
    pub const EPROC_BOA_MAX_EXPERTS: usize = 64;
    pub const EPROC_ECHO_MAX_SOURCES: usize = 256;
    pub const EPROC_ECHO_MAX_CLAIMS: usize = 256;
    pub const EPROC_ECHO_TABLE_CAP: usize = 4096;
    pub const EPROC_VAROPT_MAX_K: usize = 4096;
    pub const EPROC_RLS_MAX_D: usize = 32;
    pub const EPROC_SIFT_MAX_SITES: usize = 1024;
    pub const EPROC_SIFT_MAX_PER_SITE: usize = 16;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_bet_t {
        pub lower: f64,
        pub upper: f64,
        pub inv_span: f64,
        pub mu0_norm: f64,
        pub alpha: f64,
        pub log_thresh: f64,
        pub c_max: f64,
        pub side: i32,
        pub _pad0: u32,
        pub lam_pos: f64,
        pub a_pos: f64,
        pub log_e_pos: f64,
        pub lam_neg: f64,
        pub a_neg: f64,
        pub log_e_neg: f64,
        pub log_e: f64,
        pub max_log_e: f64,
        pub sum_x: f64,
        pub sum_x_comp: f64,
        pub sum_sq: f64,
        pub n: u64,
        pub rejected_at: u64,
        pub n_invalid: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_bet_result_t {
        pub n: u64,
        pub e_value: f64,
        pub log_e: f64,
        pub max_log_e: f64,
        pub p_value: f64,
        pub mean: f64,
        pub rejected: i32,
        pub _pad0: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_cs_t {
        pub lower: f64,
        pub upper: f64,
        pub span: f64,
        pub inv_span: f64,
        pub alpha: f64,
        pub log_2_over_alpha: f64,
        pub sigma: f64,
        pub rho: f64,
        pub mode: i32,
        pub _pad0: u32,
        pub sum_y: f64,
        pub sum_y_comp: f64,
        pub sum_sq_dev: f64,
        pub sum_lam: f64,
        pub sum_lam_y: f64,
        pub sum_psi_v: f64,
        pub cs_low: f64,
        pub cs_high: f64,
        pub sum_x: f64,
        pub sum_x_comp: f64,
        pub n: u64,
        pub n_invalid: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_cs_result_t {
        pub n: u64,
        pub mean: f64,
        pub low: f64,
        pub high: f64,
        pub width: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_edetect_t {
        pub inner: eproc_bet_t,
        pub log_cusum: f64,
        pub max_log_cusum: f64,
        pub log_thresh: f64,
        pub alarm_at: u64,
        pub n_alarms: u64,
        pub n: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_edetect_result_t {
        pub n: u64,
        pub log_stat: f64,
        pub stat: f64,
        pub alarm: i32,
        pub _pad0: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_race_t {
        pub n_arms: u32,
        pub n_active: u32,
        pub alpha: f64,
        pub lower: f64,
        pub upper: f64,
        pub margin: f64,
        pub winner: i32,
        pub _pad0: u32,
        pub total_obs: u64,
        pub active: [u8; EPROC_MAX_ARMS],
        pub eliminated_at: [u64; EPROC_MAX_ARMS],
        pub arms: [eproc_cs_t; EPROC_MAX_ARMS],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_race_status_t {
        pub n_arms: u32,
        pub n_active: u32,
        pub winner: i32,
        pub leader: i32,
        pub total_obs: u64,
        pub leader_low: f64,
        pub leader_high: f64,
        pub runner_up_high: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_conformal_t {
        pub n_rates: u32,
        pub win_cap: u32,
        pub win_len: u32,
        pub ring_pos: u32,
        pub pid_ki: f64,
        pub pid_csat: f64,
        pub n_updates: u64,
        pub target_alphas: [f64; EPROC_MAX_RATES],
        pub gammas: [f64; EPROC_AGACI_EXPERTS],
        pub alpha_t: [[f64; EPROC_AGACI_EXPERTS]; EPROC_MAX_RATES],
        pub exp_log_w: [[f64; EPROC_AGACI_EXPERTS]; EPROC_MAX_RATES],
        pub exp_var: [[f64; EPROC_AGACI_EXPERTS]; EPROC_MAX_RATES],
        pub pid_err_sum: [f64; EPROC_MAX_RATES],
        pub eff_alpha: [f64; EPROC_MAX_RATES],
        pub current_q: [f64; EPROC_MAX_RATES],
        pub emp_cov_ema: [f64; EPROC_MAX_RATES],
        pub hit_counts: [u64; EPROC_MAX_RATES],
        pub ring: [f64; EPROC_CONFORMAL_MAX_WIN],
        pub sorted: [f64; EPROC_CONFORMAL_MAX_WIN],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_conformal_out_t {
        pub n_rates: u32,
        pub _pad0: u32,
        pub quantiles: [f64; EPROC_MAX_RATES],
        pub eff_alphas: [f64; EPROC_MAX_RATES],
        pub emp_coverage: [f64; EPROC_MAX_RATES],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_delay_slot_t {
        pub ticket_id: u64,
        pub pred: f64,
        pub scale: f64,
        pub occupied: u8,
        pub _pad: [u8; 7],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_delayed_conformal_t {
        pub core: eproc_conformal_t,
        pub pending_count: u32,
        pub _pad0: u32,
        pub next_ticket: u64,
        pub slots: [eproc_delay_slot_t; EPROC_DELAY_MAX_PENDING],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_budget_conformal_t {
        pub core: eproc_conformal_t,
        pub censor_penalty: f64,
        pub n_completed: u64,
        pub n_censored: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_cmonoid_t {
        pub count: u64,
        pub bin_counts: [u64; EPROC_CMONOID_BINS],
        pub sum: f64,
        pub sum_comp: f64,
        pub sum_sq: f64,
        pub sum_sq_comp: f64,
        pub min_val: f64,
        pub max_val: f64,
        pub log_wealth: f64,
        pub scale_hint: f64,
        pub flags: u64,
        pub _reserved: [u64; 6],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_boa_t {
        pub n_experts: u32,
        pub anchor_idx: u32,
        pub fallback_active: i32,
        pub _pad0: u32,
        pub scale_ema: f64,
        pub clip_mult: f64,
        pub n_updates: u64,
        pub n_poison_clipped: u64,
        pub n_fallback_trips: u64,
        pub weights: [f64; EPROC_BOA_MAX_EXPERTS],
        pub log_w: [f64; EPROC_BOA_MAX_EXPERTS],
        pub grad_sq_sum: [f64; EPROC_BOA_MAX_EXPERTS],
        pub cum_loss: [f64; EPROC_BOA_MAX_EXPERTS],
        pub quarantined: [u8; EPROC_BOA_MAX_EXPERTS],
        pub poison_strikes: [u32; EPROC_BOA_MAX_EXPERTS],
        pub sup_e: eproc_bet_t,
        pub deg_e: eproc_bet_t,
        pub rec_e: eproc_bet_t,
        pub seqlock_seq: u64,
        pub seqlock_words: [u64; EPROC_BOA_MAX_EXPERTS + 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_boa_step_t {
        pub fused_pred: f64,
        pub anchor_pred: f64,
        pub fallback_active: i32,
        pub n_quarantined: u32,
        pub sup_log_e: f64,
        pub deg_log_e: f64,
        pub n_updates: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_echo_slot_t {
        pub key_hash: u64,
        pub evidence_id: u64,
        pub claim_id: u32,
        pub source_id: u32,
        pub weighted_llr: f64,
        pub used: u8,
        pub _pad: [u8; 7],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_echopool_t {
        pub n_sources: u32,
        pub n_claims: u32,
        pub n_accepted: u64,
        pub n_echo_dropped: u64,
        pub n_table_full: u64,
        pub src_alpha: [f64; EPROC_ECHO_MAX_SOURCES],
        pub src_beta: [f64; EPROC_ECHO_MAX_SOURCES],
        pub src_reliability: [f64; EPROC_ECHO_MAX_SOURCES],
        pub claim_prior_lo: [f64; EPROC_ECHO_MAX_CLAIMS],
        pub claim_lo: [f64; EPROC_ECHO_MAX_CLAIMS],
        pub claim_lo_comp: [f64; EPROC_ECHO_MAX_CLAIMS],
        pub claim_unique: [u64; EPROC_ECHO_MAX_CLAIMS],
        pub table: [eproc_echo_slot_t; EPROC_ECHO_TABLE_CAP],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_echo_out_t {
        pub accepted: i32,
        pub _pad0: u32,
        pub posterior_prob: f64,
        pub log_odds: f64,
        pub source_reliability: f64,
        pub unique_count: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_varopt_item_t {
        pub item_id: u64,
        pub orig_weight: f64,
        pub adj_weight: f64,
        pub value: f64,
        pub rand_u: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_varopt_t {
        pub k: u32,
        pub count: u32,
        pub n_heavy: u32,
        pub _pad0: u32,
        pub tau: f64,
        pub total_weight: f64,
        pub total_weight_comp: f64,
        pub n_seen: u64,
        pub rng_state: [u64; 2],
        pub items: [eproc_varopt_item_t; EPROC_VAROPT_MAX_K],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_rls_t {
        pub d: u32,
        pub _pad0: u32,
        pub lambda: f64,
        pub oversample: f64,
        pub min_prob: f64,
        pub n_seen: u64,
        pub n_kept: u64,
        pub rng_state: [u64; 2],
        pub inv_a: [f64; EPROC_RLS_MAX_D * EPROC_RLS_MAX_D],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_rls_out_t {
        pub keep: i32,
        pub _pad0: u32,
        pub leverage: f64,
        pub prob: f64,
        pub ht_weight: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct eproc_sift_t {
        pub n_sites: u32,
        pub per_site: u32,
        pub prior_hot_site: f64,
        pub prior_active_cold: f64,
        pub prior_active_hot: f64,
        pub signal_mean: f64,
        pub noise_sd: f64,
        pub site_log_prob: [f64; EPROC_SIFT_MAX_SITES],
        pub chan_prob: [f64; EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE],
        pub chan_noise_scale: [f64; EPROC_SIFT_MAX_SITES * EPROC_SIFT_MAX_PER_SITE],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct eproc_sift_action_t {
        pub action_type: i32,
        pub target_idx: u32,
        pub evoi_per_cost: f64,
    }

    // Compile-time ABI v1 struct size invariants (verified against C11 core).
    const _: () = {
        use std::mem::size_of;
        assert!(size_of::<eproc_bet_t>() == 176);
        assert!(size_of::<eproc_bet_result_t>() == 56);
        assert!(size_of::<eproc_cs_t>() == 168);
        assert!(size_of::<eproc_cs_result_t>() == 40);
        assert!(size_of::<eproc_edetect_t>() == 224);
        assert!(size_of::<eproc_edetect_result_t>() == 32);
        assert!(size_of::<eproc_race_t>() == 11_384);
        assert!(size_of::<eproc_race_status_t>() == 48);
        assert!(size_of::<eproc_conformal_t>() == 4_888);
        assert!(size_of::<eproc_conformal_out_t>() == 128);
        assert!(size_of::<eproc_delay_slot_t>() == 32);
        assert!(size_of::<eproc_delayed_conformal_t>() == 135_976);
        assert!(size_of::<eproc_budget_conformal_t>() == 4_912);
        assert!(size_of::<eproc_cmonoid_t>() == 640);
        assert!(size_of::<eproc_boa_t>() == 3_488);
        assert!(size_of::<eproc_boa_step_t>() == 48);
        assert!(size_of::<eproc_echo_slot_t>() == 40);
        assert!(size_of::<eproc_echopool_t>() == 178_208);
        assert!(size_of::<eproc_echo_out_t>() == 40);
        assert!(size_of::<eproc_varopt_item_t>() == 40);
        assert!(size_of::<eproc_varopt_t>() == 163_904);
        assert!(size_of::<eproc_rls_t>() == 8_256);
        assert!(size_of::<eproc_rls_out_t>() == 32);
        assert!(size_of::<eproc_sift_t>() == 270_384);
        assert!(size_of::<eproc_sift_action_t>() == 16);
    };

    extern "C" {
        pub fn eproc_bet_init(
            ep: *mut eproc_bet_t,
            mu0: f64,
            alpha: f64,
            lower: f64,
            upper: f64,
            side: i32,
        ) -> c_int;
        pub fn eproc_bet_update(
            ep: *mut eproc_bet_t,
            x: f64,
            out: *mut eproc_bet_result_t,
        ) -> c_int;
        pub fn eproc_bet_update_batch(
            ep: *mut eproc_bet_t,
            xs: *const f64,
            n: usize,
            out: *mut eproc_bet_result_t,
        ) -> c_int;
        pub fn eproc_bet_query(ep: *const eproc_bet_t, out: *mut eproc_bet_result_t);
        pub fn eproc_combine_independent_log(log_e_a: f64, log_e_b: f64) -> f64;
        pub fn eproc_combine_dependent_log(log_es: *const f64, k: usize) -> f64;

        pub fn eproc_cs_init_eb(
            cs: *mut eproc_cs_t,
            alpha: f64,
            lower: f64,
            upper: f64,
        ) -> c_int;
        pub fn eproc_cs_init_normal(
            cs: *mut eproc_cs_t,
            alpha: f64,
            sigma: f64,
            v_opt: f64,
        ) -> c_int;
        pub fn eproc_cs_update(
            cs: *mut eproc_cs_t,
            x: f64,
            out: *mut eproc_cs_result_t,
        ) -> c_int;
        pub fn eproc_cs_update_batch(
            cs: *mut eproc_cs_t,
            xs: *const f64,
            n: usize,
            out: *mut eproc_cs_result_t,
        ) -> c_int;
        pub fn eproc_cs_query(cs: *const eproc_cs_t, out: *mut eproc_cs_result_t);

        pub fn eproc_edetect_init(
            ed: *mut eproc_edetect_t,
            mu0: f64,
            alpha: f64,
            lower: f64,
            upper: f64,
            side: i32,
        ) -> c_int;
        pub fn eproc_edetect_update(
            ed: *mut eproc_edetect_t,
            x: f64,
            out: *mut eproc_edetect_result_t,
        ) -> c_int;
        pub fn eproc_edetect_reset_alarm(ed: *mut eproc_edetect_t);

        pub fn eproc_race_init(
            race: *mut eproc_race_t,
            n_arms: u32,
            alpha: f64,
            lower: f64,
            upper: f64,
            margin: f64,
        ) -> c_int;
        pub fn eproc_race_update(
            race: *mut eproc_race_t,
            arm: u32,
            reward: f64,
            out: *mut eproc_race_status_t,
        ) -> c_int;
        pub fn eproc_race_update_batch(
            race: *mut eproc_race_t,
            arms: *const u32,
            rewards: *const f64,
            n: usize,
            out: *mut eproc_race_status_t,
        ) -> c_int;
        pub fn eproc_race_status(race: *const eproc_race_t, out: *mut eproc_race_status_t);
        pub fn eproc_race_arm_state(
            race: *const eproc_race_t,
            arm: u32,
            out: *mut eproc_cs_result_t,
            is_active: *mut i32,
        ) -> c_int;
        pub fn eproc_race_suggest_arm(race: *const eproc_race_t) -> i32;

        pub fn eproc_conformal_init(
            cp: *mut eproc_conformal_t,
            alphas: *const f64,
            n_rates: u32,
            window_size: u32,
            pid_ki: f64,
        ) -> c_int;
        pub fn eproc_conformal_predict(
            cp: *const eproc_conformal_t,
            point_pred: f64,
            scale: f64,
            lows_out: *mut f64,
            highs_out: *mut f64,
            meta_out: *mut eproc_conformal_out_t,
        );
        pub fn eproc_conformal_update(
            cp: *mut eproc_conformal_t,
            nonconformity_score: f64,
            meta_out: *mut eproc_conformal_out_t,
        ) -> c_int;
        pub fn eproc_conformal_update_batch(
            cp: *mut eproc_conformal_t,
            scores: *const f64,
            n: usize,
            meta_out: *mut eproc_conformal_out_t,
        ) -> c_int;

        pub fn eproc_delayed_conformal_init(
            dc: *mut eproc_delayed_conformal_t,
            alphas: *const f64,
            n_rates: u32,
            window_size: u32,
            pid_ki: f64,
        ) -> c_int;
        pub fn eproc_delayed_conformal_issue(
            dc: *mut eproc_delayed_conformal_t,
            pred: f64,
            scale: f64,
            ticket_out: *mut u64,
            lows_out: *mut f64,
            highs_out: *mut f64,
        ) -> c_int;
        pub fn eproc_delayed_conformal_reveal(
            dc: *mut eproc_delayed_conformal_t,
            ticket_id: u64,
            y_true: f64,
            meta_out: *mut eproc_conformal_out_t,
        ) -> c_int;

        pub fn eproc_budget_conformal_init(
            bc: *mut eproc_budget_conformal_t,
            alpha: f64,
            window_size: u32,
            censor_penalty: f64,
        ) -> c_int;
        pub fn eproc_budget_conformal_bound(
            bc: *const eproc_budget_conformal_t,
            base_estimate: f64,
        ) -> f64;
        pub fn eproc_budget_conformal_update(
            bc: *mut eproc_budget_conformal_t,
            base_estimate: f64,
            observed_cost: f64,
            censored: i32,
        ) -> c_int;

        pub fn eproc_cmonoid_init(m: *mut eproc_cmonoid_t, scale_hint: f64);
        pub fn eproc_cmonoid_add(m: *mut eproc_cmonoid_t, score: f64);
        pub fn eproc_cmonoid_add_batch(m: *mut eproc_cmonoid_t, scores: *const f64, n: usize);
        pub fn eproc_cmonoid_merge(
            a: *const eproc_cmonoid_t,
            b: *const eproc_cmonoid_t,
            out: *mut eproc_cmonoid_t,
        );
        pub fn eproc_cmonoid_subtract(
            global_m: *const eproc_cmonoid_t,
            local_m: *const eproc_cmonoid_t,
            loo_out: *mut eproc_cmonoid_t,
        ) -> c_int;
        pub fn eproc_cmonoid_quantile(m: *const eproc_cmonoid_t, q: f64) -> f64;
        pub fn eproc_cmonoid_wire_size() -> usize;
        pub fn eproc_cmonoid_serialize(
            m: *const eproc_cmonoid_t,
            buf: *mut u8,
            cap: usize,
        ) -> c_int;
        pub fn eproc_cmonoid_deserialize(
            m: *mut eproc_cmonoid_t,
            buf: *const u8,
            len: usize,
        ) -> c_int;

        pub fn eproc_boa_init(
            boa: *mut eproc_boa_t,
            n_experts: u32,
            anchor_idx: u32,
            alpha_deg: f64,
        ) -> c_int;
        pub fn eproc_boa_predict(boa: *const eproc_boa_t, expert_preds: *const f64) -> f64;
        pub fn eproc_boa_update(
            boa: *mut eproc_boa_t,
            expert_preds: *const f64,
            y_true: f64,
            out: *mut eproc_boa_step_t,
        ) -> c_int;
        pub fn eproc_boa_snapshot_weights(
            boa: *const eproc_boa_t,
            weights_out: *mut f64,
            max_experts: u32,
            fallback_out: *mut i32,
        ) -> c_int;

        pub fn eproc_echopool_init(
            pool: *mut eproc_echopool_t,
            n_sources: u32,
            n_claims: u32,
            prior_prob: f64,
        ) -> c_int;
        pub fn eproc_echopool_submit(
            pool: *mut eproc_echopool_t,
            source_id: u32,
            claim_id: u32,
            evidence_id: u64,
            llr: f64,
            out: *mut eproc_echo_out_t,
        ) -> c_int;
        pub fn eproc_echopool_resolve(
            pool: *mut eproc_echopool_t,
            claim_id: u32,
            outcome_positive: i32,
        ) -> c_int;
        pub fn eproc_echopool_posterior(pool: *const eproc_echopool_t, claim_id: u32) -> f64;
        pub fn eproc_echopool_source_reliability(
            pool: *const eproc_echopool_t,
            source_id: u32,
        ) -> f64;

        pub fn eproc_varopt_init(vo: *mut eproc_varopt_t, k: u32, seed: u64) -> c_int;
        pub fn eproc_varopt_push(
            vo: *mut eproc_varopt_t,
            item_id: u64,
            weight: f64,
            value: f64,
        ) -> c_int;
        pub fn eproc_varopt_estimate_sum(vo: *const eproc_varopt_t) -> f64;
        pub fn eproc_varopt_estimate_mean(vo: *const eproc_varopt_t) -> f64;

        pub fn eproc_rls_init(
            rls: *mut eproc_rls_t,
            d: u32,
            lambda: f64,
            oversample: f64,
            min_prob: f64,
            seed: u64,
        ) -> c_int;
        pub fn eproc_rls_step(
            rls: *mut eproc_rls_t,
            x: *const f64,
            out: *mut eproc_rls_out_t,
        ) -> c_int;

        pub fn eproc_sift_init(
            sf: *mut eproc_sift_t,
            n_sites: u32,
            per_site: u32,
            signal_mean: f64,
            noise_sd: f64,
        ) -> c_int;
        pub fn eproc_sift_set_noise_scale(
            sf: *mut eproc_sift_t,
            channel_idx: u32,
            noise_scale: f64,
        ) -> c_int;
        pub fn eproc_sift_observe_channel(
            sf: *mut eproc_sift_t,
            channel_idx: u32,
            obs_mean: f64,
            n_ticks: u32,
        ) -> c_int;
        pub fn eproc_sift_observe_site(
            sf: *mut eproc_sift_t,
            site_idx: u32,
            site_mean: f64,
        ) -> c_int;
        pub fn eproc_sift_select_next(
            sf: *const eproc_sift_t,
            cost_site: f64,
            cost_channel: f64,
            cost_ticks: f64,
            ticks_per_read: u32,
            out: *mut eproc_sift_action_t,
        ) -> c_int;

        pub fn eproc_abi_version() -> u32;
        pub fn eproc_simd_flags() -> u32;
    }
}

/// SIMD capability bit: AVX2 active (compile-time or runtime CPUID dispatched).
pub const SIMD_AVX2: u32 = 1 << 0;
/// SIMD capability bit: FMA active at compile time.
pub const SIMD_FMA: u32 = 1 << 1;
/// SIMD capability bit: ARM NEON active.
pub const SIMD_NEON: u32 = 1 << 2;

/// Returns the compiled C11 core ABI version (`1`).
#[inline]
pub fn abi_version() -> u32 {
    unsafe { ffi::eproc_abi_version() }
}

/// Returns active runtime SIMD capability flags (bit 0: AVX2, bit 1: FMA, bit 2: NEON).
#[inline]
pub fn simd_flags() -> u32 {
    unsafe { ffi::eproc_simd_flags() }
}

/// Combine two independent e-processes in log-space (`log(E_a * E_b)`).
#[inline]
pub fn combine_independent_log(log_e_a: f64, log_e_b: f64) -> f64 {
    unsafe { ffi::eproc_combine_independent_log(log_e_a, log_e_b) }
}

/// Combine arbitrarily dependent e-processes via arithmetic averaging (`log((1/K) sum_k E_k)`).
#[inline]
pub fn combine_dependent_log(log_es: &[f64]) -> f64 {
    if log_es.is_empty() {
        return 0.0;
    }
    unsafe { ffi::eproc_combine_dependent_log(log_es.as_ptr(), log_es.len()) }
}

/// Error codes returned by the C11 core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Invalid,
    Capacity,
    Empty,
    Mismatch,
    Unknown(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid => f.write_str("invalid argument or state"),
            Error::Capacity => f.write_str("capacity exceeded"),
            Error::Empty => f.write_str("collection or stream is empty"),
            Error::Mismatch => f.write_str("checksum or format mismatch"),
            Error::Unknown(code) => write!(f, "unknown eproc status code {code}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[inline]
fn check_rc(rc: c_int) -> Result<()> {
    match rc {
        ffi::EPROC_OK => Ok(()),
        ffi::EPROC_ERR_INVALID => Err(Error::Invalid),
        ffi::EPROC_ERR_CAPACITY => Err(Error::Capacity),
        ffi::EPROC_ERR_EMPTY => Err(Error::Empty),
        ffi::EPROC_ERR_MISMATCH => Err(Error::Mismatch),
        other => Err(Error::Unknown(other)),
    }
}

fn alloc_zeroed_box<T>() -> Box<T> {
    let layout = Layout::new::<T>();
    assert!(layout.size() > 0);
    // SAFETY: Caller only uses this for #[repr(C)] POD C structs (or UnsafeCell
    // wrapping one) immediately initialized by the corresponding eproc_*_init.
    let ptr = unsafe { alloc_zeroed(layout) as *mut T };
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    unsafe { Box::from_raw(ptr) }
}

/// Hypothesis test direction for betting e-processes and e-detectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// `H0: E[X] <= mu0` vs `H1: E[X] > mu0`.
    Upper,
    /// `H0: E[X] >= mu0` vs `H1: E[X] < mu0`.
    Lower,
    /// `H0: E[X] == mu0` vs `H1: E[X] != mu0`.
    TwoSided,
}

impl Side {
    #[inline]
    fn to_raw(self) -> i32 {
        match self {
            Side::Upper => ffi::EPROC_SIDE_UPPER,
            Side::Lower => ffi::EPROC_SIDE_LOWER,
            Side::TwoSided => ffi::EPROC_SIDE_TWO_SIDED,
        }
    }
}

// ============================================================================
// 1. EProcess, ConfidenceSequence, EDetector, AgentRace
// ============================================================================

/// Summary returned by [`EProcess`] queries and updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EProcessResult {
    pub n: u64,
    pub e_value: f64,
    pub log_e: f64,
    pub max_log_e: f64,
    pub p_value: f64,
    pub mean: f64,
    pub rejected: bool,
}

impl From<ffi::eproc_bet_result_t> for EProcessResult {
    #[inline]
    fn from(r: ffi::eproc_bet_result_t) -> Self {
        Self {
            n: r.n,
            e_value: r.e_value,
            log_e: r.log_e,
            max_log_e: r.max_log_e,
            p_value: r.p_value,
            mean: r.mean,
            rejected: r.rejected != 0,
        }
    }
}

/// Anytime-valid ONS betting supermartingale e-process for bounded streams.
#[derive(Clone)]
pub struct EProcess {
    inner: ffi::eproc_bet_t,
}

impl EProcess {
    /// Initializes an anytime-valid e-process testing `mu0` at level `alpha` on `[lower, upper]`.
    pub fn new(mu0: f64, alpha: f64, lower: f64, upper: f64, side: Side) -> Result<Self> {
        let mut inner = MaybeUninit::<ffi::eproc_bet_t>::zeroed();
        let rc = unsafe {
            ffi::eproc_bet_init(inner.as_mut_ptr(), mu0, alpha, lower, upper, side.to_raw())
        };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { inner.assume_init() },
        })
    }

    /// Updates the e-process with a single observation `x`.
    #[inline]
    pub fn update(&mut self, x: f64) -> Result<EProcessResult> {
        let mut out = ffi::eproc_bet_result_t::default();
        let rc = unsafe { ffi::eproc_bet_update(&mut self.inner, x, &mut out) };
        check_rc(rc)?;
        Ok(out.into())
    }

    /// Updates the e-process with a contiguous slice of observations (bit-identical to scalar loop).
    #[inline]
    pub fn update_batch(&mut self, xs: &[f64]) -> Result<EProcessResult> {
        let mut out = ffi::eproc_bet_result_t::default();
        let rc = unsafe {
            ffi::eproc_bet_update_batch(&mut self.inner, xs.as_ptr(), xs.len(), &mut out)
        };
        check_rc(rc)?;
        Ok(out.into())
    }

    /// Returns the current anytime-valid test summary without mutating state.
    #[inline]
    pub fn query(&self) -> EProcessResult {
        let mut out = ffi::eproc_bet_result_t::default();
        unsafe { ffi::eproc_bet_query(&self.inner, &mut out) };
        out.into()
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.inner.n
    }

    #[inline]
    pub fn n_invalid(&self) -> u64 {
        self.inner.n_invalid
    }

    #[inline]
    pub fn rejected_at(&self) -> Option<u64> {
        if self.inner.rejected_at == 0 {
            None
        } else {
            Some(self.inner.rejected_at)
        }
    }
}

/// Summary returned by [`ConfidenceSequence`] queries and updates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceSequenceResult {
    pub n: u64,
    pub mean: f64,
    pub low: f64,
    pub high: f64,
    pub width: f64,
}

impl From<ffi::eproc_cs_result_t> for ConfidenceSequenceResult {
    #[inline]
    fn from(r: ffi::eproc_cs_result_t) -> Self {
        Self {
            n: r.n,
            mean: r.mean,
            low: r.low,
            high: r.high,
            width: r.width,
        }
    }
}

/// Anytime-valid confidence sequence (Empirical-Bernstein or sub-Gaussian Normal-Mixture).
#[derive(Clone)]
pub struct ConfidenceSequence {
    inner: ffi::eproc_cs_t,
}

impl ConfidenceSequence {
    /// Initializes an Empirical-Bernstein confidence sequence for bounded streams in `[lower, upper]`.
    pub fn empirical_bernstein(alpha: f64, lower: f64, upper: f64) -> Result<Self> {
        let mut inner = MaybeUninit::<ffi::eproc_cs_t>::zeroed();
        let rc = unsafe { ffi::eproc_cs_init_eb(inner.as_mut_ptr(), alpha, lower, upper) };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { inner.assume_init() },
        })
    }

    /// Initializes a sub-Gaussian Normal-Mixture confidence sequence with proxy `sigma` and target variance `v_opt`.
    pub fn normal_mixture(alpha: f64, sigma: f64, v_opt: f64) -> Result<Self> {
        let mut inner = MaybeUninit::<ffi::eproc_cs_t>::zeroed();
        let rc = unsafe { ffi::eproc_cs_init_normal(inner.as_mut_ptr(), alpha, sigma, v_opt) };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { inner.assume_init() },
        })
    }

    #[inline]
    pub fn update(&mut self, x: f64) -> Result<ConfidenceSequenceResult> {
        let mut out = ffi::eproc_cs_result_t::default();
        let rc = unsafe { ffi::eproc_cs_update(&mut self.inner, x, &mut out) };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn update_batch(&mut self, xs: &[f64]) -> Result<ConfidenceSequenceResult> {
        let mut out = ffi::eproc_cs_result_t::default();
        let rc = unsafe {
            ffi::eproc_cs_update_batch(&mut self.inner, xs.as_ptr(), xs.len(), &mut out)
        };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn query(&self) -> ConfidenceSequenceResult {
        let mut out = ffi::eproc_cs_result_t::default();
        unsafe { ffi::eproc_cs_query(&self.inner, &mut out) };
        out.into()
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.inner.n
    }

    #[inline]
    pub fn n_invalid(&self) -> u64 {
        self.inner.n_invalid
    }
}

/// Summary returned by [`EDetector::update`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EDetectorResult {
    pub n: u64,
    pub log_stat: f64,
    pub stat: f64,
    pub alarm: bool,
}

impl From<ffi::eproc_edetect_result_t> for EDetectorResult {
    #[inline]
    fn from(r: ffi::eproc_edetect_result_t) -> Self {
        Self {
            n: r.n,
            log_stat: r.log_stat,
            stat: r.stat,
            alarm: r.alarm != 0,
        }
    }
}

/// Anytime-valid Page-CUSUM resettable e-detector for streaming drift detection.
#[derive(Clone)]
pub struct EDetector {
    inner: ffi::eproc_edetect_t,
}

impl EDetector {
    pub fn new(mu0: f64, alpha: f64, lower: f64, upper: f64, side: Side) -> Result<Self> {
        let mut inner = MaybeUninit::<ffi::eproc_edetect_t>::zeroed();
        let rc = unsafe {
            ffi::eproc_edetect_init(inner.as_mut_ptr(), mu0, alpha, lower, upper, side.to_raw())
        };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { inner.assume_init() },
        })
    }

    #[inline]
    pub fn update(&mut self, x: f64) -> Result<EDetectorResult> {
        let mut out = ffi::eproc_edetect_result_t::default();
        let rc = unsafe { ffi::eproc_edetect_update(&mut self.inner, x, &mut out) };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn reset_alarm(&mut self) {
        unsafe { ffi::eproc_edetect_reset_alarm(&mut self.inner) };
    }

    #[inline]
    pub fn n(&self) -> u64 {
        self.inner.n
    }

    #[inline]
    pub fn n_alarms(&self) -> u64 {
        self.inner.n_alarms
    }

    #[inline]
    pub fn log_cusum(&self) -> f64 {
        self.inner.log_cusum
    }

    #[inline]
    pub fn alarm_at(&self) -> Option<u64> {
        if self.inner.alarm_at == 0 {
            None
        } else {
            Some(self.inner.alarm_at)
        }
    }
}

/// Snapshot of an [`AgentRace`] best-arm identification session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaceStatus {
    pub n_arms: usize,
    pub n_active: usize,
    pub winner: Option<usize>,
    pub leader: Option<usize>,
    pub total_obs: u64,
    pub leader_low: f64,
    pub leader_high: f64,
    pub runner_up_high: f64,
}

impl From<ffi::eproc_race_status_t> for RaceStatus {
    #[inline]
    fn from(s: ffi::eproc_race_status_t) -> Self {
        Self {
            n_arms: s.n_arms as usize,
            n_active: s.n_active as usize,
            winner: if s.winner >= 0 {
                Some(s.winner as usize)
            } else {
                None
            },
            leader: if s.leader >= 0 {
                Some(s.leader as usize)
            } else {
                None
            },
            total_obs: s.total_obs,
            leader_low: s.leader_low,
            leader_high: s.leader_high,
            runner_up_high: s.runner_up_high,
        }
    }
}

/// Anytime-valid multi-arm best-arm racer with automatic elimination and FWER control.
pub struct AgentRace {
    inner: Box<ffi::eproc_race_t>,
}

impl AgentRace {
    pub fn new(
        n_arms: usize,
        alpha: f64,
        lower: f64,
        upper: f64,
        margin: f64,
    ) -> Result<Self> {
        let n_arms_u32 = u32::try_from(n_arms).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_race_t>();
        let rc = unsafe {
            ffi::eproc_race_init(inner.as_mut(), n_arms_u32, alpha, lower, upper, margin)
        };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<RaceStatus> {
        let arm_u32 = u32::try_from(arm).map_err(|_| Error::Invalid)?;
        let mut out = ffi::eproc_race_status_t::default();
        let rc = unsafe { ffi::eproc_race_update(self.inner.as_mut(), arm_u32, reward, &mut out) };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn update_batch(&mut self, arms: &[u32], rewards: &[f64]) -> Result<RaceStatus> {
        if arms.len() != rewards.len() {
            return Err(Error::Invalid);
        }
        let mut out = ffi::eproc_race_status_t::default();
        let rc = unsafe {
            ffi::eproc_race_update_batch(
                self.inner.as_mut(),
                arms.as_ptr(),
                rewards.as_ptr(),
                arms.len(),
                &mut out,
            )
        };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn status(&self) -> RaceStatus {
        let mut out = ffi::eproc_race_status_t::default();
        unsafe { ffi::eproc_race_status(self.inner.as_ref(), &mut out) };
        out.into()
    }

    #[inline]
    pub fn arm_state(&self, arm: usize) -> Result<(ConfidenceSequenceResult, bool)> {
        let arm_u32 = u32::try_from(arm).map_err(|_| Error::Invalid)?;
        let mut out = ffi::eproc_cs_result_t::default();
        let mut active: i32 = 0;
        let rc = unsafe {
            ffi::eproc_race_arm_state(self.inner.as_ref(), arm_u32, &mut out, &mut active)
        };
        check_rc(rc)?;
        Ok((out.into(), active != 0))
    }

    #[inline]
    pub fn suggest_arm(&self) -> Option<usize> {
        let idx = unsafe { ffi::eproc_race_suggest_arm(self.inner.as_ref()) };
        if idx >= 0 {
            Some(idx as usize)
        } else {
            None
        }
    }
}

// ============================================================================
// 2. MultiRateConformal, DelayedConformal, BudgetConformal, ConformalMonoid
// ============================================================================

/// Multi-rate conformal calibration metadata (fixed-size, zero heap allocation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalOutput {
    n_rates: usize,
    quantiles: [f64; ffi::EPROC_MAX_RATES],
    eff_alphas: [f64; ffi::EPROC_MAX_RATES],
    emp_coverage: [f64; ffi::EPROC_MAX_RATES],
}

impl From<ffi::eproc_conformal_out_t> for ConformalOutput {
    #[inline]
    fn from(o: ffi::eproc_conformal_out_t) -> Self {
        Self {
            n_rates: (o.n_rates as usize).min(ffi::EPROC_MAX_RATES),
            quantiles: o.quantiles,
            eff_alphas: o.eff_alphas,
            emp_coverage: o.emp_coverage,
        }
    }
}

impl ConformalOutput {
    #[inline]
    pub fn n_rates(&self) -> usize {
        self.n_rates
    }

    #[inline]
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles[..self.n_rates]
    }

    #[inline]
    pub fn eff_alphas(&self) -> &[f64] {
        &self.eff_alphas[..self.n_rates]
    }

    #[inline]
    pub fn emp_coverage(&self) -> &[f64] {
        &self.emp_coverage[..self.n_rates]
    }
}

/// Nested multi-rate conformal prediction intervals (fixed-size, zero heap allocation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformalIntervals {
    n_rates: usize,
    lows: [f64; ffi::EPROC_MAX_RATES],
    highs: [f64; ffi::EPROC_MAX_RATES],
    meta: ConformalOutput,
}

impl ConformalIntervals {
    #[inline]
    pub fn n_rates(&self) -> usize {
        self.n_rates
    }

    #[inline]
    pub fn lows(&self) -> &[f64] {
        &self.lows[..self.n_rates]
    }

    #[inline]
    pub fn highs(&self) -> &[f64] {
        &self.highs[..self.n_rates]
    }

    #[inline]
    pub fn meta(&self) -> &ConformalOutput {
        &self.meta
    }
}

/// Multi-rate crossing-free AgACI + Conformal PID calibrator (up to 5 rates).
pub struct MultiRateConformal {
    inner: Box<ffi::eproc_conformal_t>,
}

impl MultiRateConformal {
    pub fn new(alphas: &[f64], window_size: usize, pid_ki: f64) -> Result<Self> {
        let n_rates = u32::try_from(alphas.len()).map_err(|_| Error::Invalid)?;
        let win = u32::try_from(window_size).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_conformal_t>();
        let rc = unsafe {
            ffi::eproc_conformal_init(inner.as_mut(), alphas.as_ptr(), n_rates, win, pid_ki)
        };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn n_rates(&self) -> usize {
        self.inner.n_rates as usize
    }

    #[inline]
    pub fn target_alphas(&self) -> &[f64] {
        &self.inner.target_alphas[..self.n_rates()]
    }

    #[inline]
    pub fn predict(&self, point_pred: f64, scale: f64) -> ConformalIntervals {
        let mut lows = [0.0f64; ffi::EPROC_MAX_RATES];
        let mut highs = [0.0f64; ffi::EPROC_MAX_RATES];
        let mut meta = ffi::eproc_conformal_out_t::default();
        unsafe {
            ffi::eproc_conformal_predict(
                self.inner.as_ref(),
                point_pred,
                scale,
                lows.as_mut_ptr(),
                highs.as_mut_ptr(),
                &mut meta,
            );
        }
        ConformalIntervals {
            n_rates: self.n_rates(),
            lows,
            highs,
            meta: meta.into(),
        }
    }

    #[inline]
    pub fn update(&mut self, nonconformity_score: f64) -> Result<ConformalOutput> {
        let mut meta = ffi::eproc_conformal_out_t::default();
        let rc = unsafe {
            ffi::eproc_conformal_update(self.inner.as_mut(), nonconformity_score, &mut meta)
        };
        check_rc(rc)?;
        Ok(meta.into())
    }

    #[inline]
    pub fn update_batch(&mut self, scores: &[f64]) -> Result<ConformalOutput> {
        let mut meta = ffi::eproc_conformal_out_t::default();
        let rc = unsafe {
            ffi::eproc_conformal_update_batch(
                self.inner.as_mut(),
                scores.as_ptr(),
                scores.len(),
                &mut meta,
            )
        };
        check_rc(rc)?;
        Ok(meta.into())
    }
}

/// Strictly causal delayed-feedback multi-rate conformal wrapper.
pub struct DelayedConformal {
    inner: Box<ffi::eproc_delayed_conformal_t>,
}

/// Issued delayed-feedback ticket and causal prediction intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayedTicket {
    pub ticket_id: u64,
    n_rates: usize,
    lows: [f64; ffi::EPROC_MAX_RATES],
    highs: [f64; ffi::EPROC_MAX_RATES],
}

impl DelayedTicket {
    #[inline]
    pub fn lows(&self) -> &[f64] {
        &self.lows[..self.n_rates]
    }

    #[inline]
    pub fn highs(&self) -> &[f64] {
        &self.highs[..self.n_rates]
    }
}

impl DelayedConformal {
    pub fn new(alphas: &[f64], window_size: usize, pid_ki: f64) -> Result<Self> {
        let n_rates = u32::try_from(alphas.len()).map_err(|_| Error::Invalid)?;
        let win = u32::try_from(window_size).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_delayed_conformal_t>();
        let rc = unsafe {
            ffi::eproc_delayed_conformal_init(inner.as_mut(), alphas.as_ptr(), n_rates, win, pid_ki)
        };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn issue(&mut self, pred: f64, scale: f64) -> Result<DelayedTicket> {
        let mut ticket_id = 0u64;
        let mut lows = [0.0f64; ffi::EPROC_MAX_RATES];
        let mut highs = [0.0f64; ffi::EPROC_MAX_RATES];
        let rc = unsafe {
            ffi::eproc_delayed_conformal_issue(
                self.inner.as_mut(),
                pred,
                scale,
                &mut ticket_id,
                lows.as_mut_ptr(),
                highs.as_mut_ptr(),
            )
        };
        check_rc(rc)?;
        Ok(DelayedTicket {
            ticket_id,
            n_rates: self.inner.core.n_rates as usize,
            lows,
            highs,
        })
    }

    #[inline]
    pub fn reveal(&mut self, ticket_id: u64, y_true: f64) -> Result<ConformalOutput> {
        let mut meta = ffi::eproc_conformal_out_t::default();
        let rc = unsafe {
            ffi::eproc_delayed_conformal_reveal(self.inner.as_mut(), ticket_id, y_true, &mut meta)
        };
        check_rc(rc)?;
        Ok(meta.into())
    }

    #[inline]
    pub fn pending_count(&self) -> u32 {
        self.inner.pending_count
    }
}

/// Right-censored agent budget conformal calibrator (upper predictive bound).
pub struct BudgetConformal {
    inner: Box<ffi::eproc_budget_conformal_t>,
}

impl BudgetConformal {
    pub fn new(alpha: f64, window_size: usize, censor_penalty: f64) -> Result<Self> {
        let win = u32::try_from(window_size).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_budget_conformal_t>();
        let rc = unsafe {
            ffi::eproc_budget_conformal_init(inner.as_mut(), alpha, win, censor_penalty)
        };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn bound(&self, base_estimate: f64) -> f64 {
        unsafe { ffi::eproc_budget_conformal_bound(self.inner.as_ref(), base_estimate) }
    }

    #[inline]
    pub fn update(&mut self, base_estimate: f64, observed_cost: f64, censored: bool) -> Result<()> {
        let rc = unsafe {
            ffi::eproc_budget_conformal_update(
                self.inner.as_mut(),
                base_estimate,
                observed_cost,
                if censored { 1 } else { 0 },
            )
        };
        check_rc(rc)
    }

    #[inline]
    pub fn n_completed(&self) -> u64 {
        self.inner.n_completed
    }

    #[inline]
    pub fn n_censored(&self) -> u64 {
        self.inner.n_censored
    }
}

/// Exact associative & commutative 640-byte distributed conformal monoid.
#[derive(Clone, Copy)]
pub struct ConformalMonoid {
    inner: ffi::eproc_cmonoid_t,
}

impl Default for ConformalMonoid {
    #[inline]
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ConformalMonoid {
    /// Fixed wire size in bytes (`640`).
    pub const WIRE_SIZE: usize = ffi::EPROC_CMONOID_WIRE_SIZE;

    /// Returns the runtime wire size reported by the C11 core (`640`).
    #[inline]
    pub fn wire_size() -> usize {
        unsafe { ffi::eproc_cmonoid_wire_size() }
    }

    #[inline]
    pub fn new(scale_hint: f64) -> Self {
        let mut inner = MaybeUninit::<ffi::eproc_cmonoid_t>::zeroed();
        unsafe { ffi::eproc_cmonoid_init(inner.as_mut_ptr(), scale_hint) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    #[inline]
    pub fn add(&mut self, score: f64) {
        unsafe { ffi::eproc_cmonoid_add(&mut self.inner, score) };
    }

    #[inline]
    pub fn add_batch(&mut self, scores: &[f64]) {
        unsafe { ffi::eproc_cmonoid_add_batch(&mut self.inner, scores.as_ptr(), scores.len()) };
    }

    /// Bit-exact associative and commutative merge `self + other`.
    #[inline]
    pub fn merge(&self, other: &Self) -> Self {
        let mut out = MaybeUninit::<ffi::eproc_cmonoid_t>::zeroed();
        unsafe { ffi::eproc_cmonoid_merge(&self.inner, &other.inner, out.as_mut_ptr()) };
        Self {
            inner: unsafe { out.assume_init() },
        }
    }

    /// Exact leave-one-out abelian subtraction `self - local`.
    #[inline]
    pub fn subtract(&self, local: &Self) -> Result<Self> {
        let mut out = MaybeUninit::<ffi::eproc_cmonoid_t>::zeroed();
        let rc =
            unsafe { ffi::eproc_cmonoid_subtract(&self.inner, &local.inner, out.as_mut_ptr()) };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { out.assume_init() },
        })
    }

    #[inline]
    pub fn quantile(&self, q: f64) -> f64 {
        unsafe { ffi::eproc_cmonoid_quantile(&self.inner, q) }
    }

    #[inline]
    pub fn count(&self) -> u64 {
        self.inner.count
    }

    #[inline]
    pub fn bin_counts(&self) -> &[u64; ffi::EPROC_CMONOID_BINS] {
        &self.inner.bin_counts
    }

    /// Serializes into a checksummed little-endian 640-byte wire frame.
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::WIRE_SIZE] {
        let mut buf = [0u8; Self::WIRE_SIZE];
        let rc =
            unsafe { ffi::eproc_cmonoid_serialize(&self.inner, buf.as_mut_ptr(), buf.len()) };
        debug_assert_eq!(rc, ffi::EPROC_OK);
        buf
    }

    /// Deserializes and FNV-1a checksum-verifies a 640-byte wire frame.
    #[inline]
    pub fn from_bytes(buf: &[u8; Self::WIRE_SIZE]) -> Result<Self> {
        Self::deserialize(buf)
    }

    /// Deserializes from an arbitrary byte slice (requires exact length 640 and valid checksum).
    #[inline]
    pub fn deserialize(buf: &[u8]) -> Result<Self> {
        let mut inner = MaybeUninit::<ffi::eproc_cmonoid_t>::zeroed();
        let rc =
            unsafe { ffi::eproc_cmonoid_deserialize(inner.as_mut_ptr(), buf.as_ptr(), buf.len()) };
        check_rc(rc)?;
        Ok(Self {
            inner: unsafe { inner.assume_init() },
        })
    }
}

// ============================================================================
// 3. SafeFuser (Boehm-2012 Atomic Seqlock) & EchoPool
// ============================================================================

/// Result of a single [`SafeFuser::update`] step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoaStep {
    pub fused_pred: f64,
    pub anchor_pred: f64,
    pub fallback_active: bool,
    pub n_quarantined: u32,
    pub sup_log_e: f64,
    pub deg_log_e: f64,
    pub n_updates: u64,
}

impl From<ffi::eproc_boa_step_t> for BoaStep {
    #[inline]
    fn from(s: ffi::eproc_boa_step_t) -> Self {
        Self {
            fused_pred: s.fused_pred,
            anchor_pred: s.anchor_pred,
            fallback_active: s.fallback_active != 0,
            n_quarantined: s.n_quarantined,
            sup_log_e: s.sup_log_e,
            deg_log_e: s.deg_log_e,
            n_updates: s.n_updates,
        }
    }
}

/// Lock-free Boehm-2012 atomic seqlock weight snapshot (zero heap allocation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoaSnapshot {
    n_experts: usize,
    weights: [f64; ffi::EPROC_BOA_MAX_EXPERTS],
    pub fallback_active: bool,
}

impl BoaSnapshot {
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.weights[..self.n_experts]
    }

    #[inline]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
}

struct SafeFuserInner {
    // Immutable after construction so concurrent readers never touch non-atomic C fields.
    n_experts: u32,
    anchor_idx: u32,
    // UnsafeCell opts out of Rust's `noalias` assumptions for the atomic seqlock fields
    // read concurrently by `SafeFuserReader::snapshot`.
    boa: Box<UnsafeCell<ffi::eproc_boa_t>>,
}

// SAFETY:
// - `SafeFuser` is the unique non-Clone owner allowed to invoke mutating C calls (`update`),
//   requiring `&mut self` (single-writer guarantee).
// - `SafeFuserReader` only ever invokes `eproc_boa_snapshot_weights`, which exclusively accesses
//   `seqlock_seq` and `seqlock_words` through C11 atomic operations with release/acquire fences.
// - `eproc_boa_t` contains no thread-local state or raw pointers.
unsafe impl Send for SafeFuserInner {}
unsafe impl Sync for SafeFuserInner {}

/// Wait-free concurrent reader handle for [`SafeFuser`]'s Boehm-2012 atomic seqlock.
#[derive(Clone)]
pub struct SafeFuserReader {
    inner: Arc<SafeFuserInner>,
}

impl SafeFuserReader {
    /// Reads a consistent, tear-free weight snapshot concurrently with [`SafeFuser::update`].
    #[inline]
    pub fn snapshot(&self) -> BoaSnapshot {
        let mut weights = [0.0f64; ffi::EPROC_BOA_MAX_EXPERTS];
        let mut fallback: i32 = 0;
        let n = self.inner.n_experts;
        let rc = unsafe {
            ffi::eproc_boa_snapshot_weights(
                self.inner.boa.get() as *const ffi::eproc_boa_t,
                weights.as_mut_ptr(),
                n,
                &mut fallback,
            )
        };
        debug_assert_eq!(rc, ffi::EPROC_OK);
        BoaSnapshot {
            n_experts: n as usize,
            weights,
            fallback_active: fallback != 0,
        }
    }

    #[inline]
    pub fn n_experts(&self) -> usize {
        self.inner.n_experts as usize
    }

    #[inline]
    pub fn anchor_idx(&self) -> usize {
        self.inner.anchor_idx as usize
    }
}

/// Byzantine-safe multi-agent BOA fuser with 3-gate Ville fallback and lock-free atomic seqlock.
///
/// `SafeFuser` is `Send + Sync`. Call [`SafeFuser::reader`] to obtain `Clone + Send + Sync`
/// wait-free snapshot handles that may read weights concurrently while [`SafeFuser::update`] runs.
pub struct SafeFuser {
    inner: Arc<SafeFuserInner>,
}

// SAFETY: `SafeFuserInner` is `Send + Sync`, `SafeFuser` is not `Clone`, `&self` methods never
// mutate state, and `&mut self` methods have exclusive writer access while `SafeFuserReader`
// only touches the C11 atomic seqlock fields.
unsafe impl Send for SafeFuser {}
unsafe impl Sync for SafeFuser {}

impl SafeFuser {
    pub fn new(n_experts: usize, anchor_idx: usize, alpha_deg: f64) -> Result<Self> {
        let n_u32 = u32::try_from(n_experts).map_err(|_| Error::Invalid)?;
        let anc_u32 = u32::try_from(anchor_idx).map_err(|_| Error::Invalid)?;
        let boa = alloc_zeroed_box::<UnsafeCell<ffi::eproc_boa_t>>();
        let rc = unsafe { ffi::eproc_boa_init(boa.get(), n_u32, anc_u32, alpha_deg) };
        check_rc(rc)?;
        Ok(Self {
            inner: Arc::new(SafeFuserInner {
                n_experts: n_u32,
                anchor_idx: anc_u32,
                boa,
            }),
        })
    }

    /// Creates a `Clone + Send + Sync` lock-free seqlock snapshot reader.
    #[inline]
    pub fn reader(&self) -> SafeFuserReader {
        SafeFuserReader {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Reads an atomic, tear-free weight snapshot via the Boehm-2012 seqlock.
    #[inline]
    pub fn snapshot(&self) -> BoaSnapshot {
        let mut weights = [0.0f64; ffi::EPROC_BOA_MAX_EXPERTS];
        let mut fallback: i32 = 0;
        let n = self.inner.n_experts;
        let rc = unsafe {
            ffi::eproc_boa_snapshot_weights(
                self.inner.boa.get() as *const ffi::eproc_boa_t,
                weights.as_mut_ptr(),
                n,
                &mut fallback,
            )
        };
        debug_assert_eq!(rc, ffi::EPROC_OK);
        BoaSnapshot {
            n_experts: n as usize,
            weights,
            fallback_active: fallback != 0,
        }
    }

    /// Computes the Byzantine-safe fused prediction without mutating state.
    #[inline]
    pub fn predict(&self, expert_preds: &[f64]) -> Result<f64> {
        if expert_preds.len() != self.n_experts() {
            return Err(Error::Invalid);
        }
        Ok(unsafe {
            ffi::eproc_boa_predict(
                self.inner.boa.get() as *const ffi::eproc_boa_t,
                expert_preds.as_ptr(),
            )
        })
    }

    /// Updates expert weights, quarantine flags, Ville gates, and publishes the seqlock snapshot.
    #[inline]
    pub fn update(&mut self, expert_preds: &[f64], y_true: f64) -> Result<BoaStep> {
        if expert_preds.len() != self.n_experts() {
            return Err(Error::Invalid);
        }
        let mut out = ffi::eproc_boa_step_t::default();
        let rc = unsafe {
            ffi::eproc_boa_update(self.inner.boa.get(), expert_preds.as_ptr(), y_true, &mut out)
        };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn n_experts(&self) -> usize {
        self.inner.n_experts as usize
    }

    #[inline]
    pub fn anchor_idx(&self) -> usize {
        self.inner.anchor_idx as usize
    }

    #[inline]
    pub fn fallback_active(&self) -> bool {
        let ptr = self.inner.boa.get() as *const ffi::eproc_boa_t;
        unsafe { std::ptr::addr_of!((*ptr).fallback_active).read() != 0 }
    }

    #[inline]
    pub fn is_quarantined(&self, expert: usize) -> bool {
        if expert >= self.n_experts() {
            return false;
        }
        let ptr = self.inner.boa.get() as *const ffi::eproc_boa_t;
        unsafe {
            let base = std::ptr::addr_of!((*ptr).quarantined) as *const u8;
            base.add(expert).read() != 0
        }
    }

    #[inline]
    pub fn weights(&self) -> &[f64] {
        let n = self.n_experts();
        let ptr = self.inner.boa.get() as *const ffi::eproc_boa_t;
        unsafe {
            let base = std::ptr::addr_of!((*ptr).weights) as *const f64;
            std::slice::from_raw_parts(base, n)
        }
    }

    #[inline]
    pub fn n_poison_clipped(&self) -> u64 {
        let ptr = self.inner.boa.get() as *const ffi::eproc_boa_t;
        unsafe { std::ptr::addr_of!((*ptr).n_poison_clipped).read() }
    }

    #[inline]
    pub fn n_fallback_trips(&self) -> u64 {
        let ptr = self.inner.boa.get() as *const ffi::eproc_boa_t;
        unsafe { std::ptr::addr_of!((*ptr).n_fallback_trips).read() }
    }
}

/// Result returned by [`EchoPool::submit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoOutcome {
    pub accepted: bool,
    pub posterior_prob: f64,
    pub log_odds: f64,
    pub source_reliability: f64,
    pub unique_count: u64,
}

impl From<ffi::eproc_echo_out_t> for EchoOutcome {
    #[inline]
    fn from(o: ffi::eproc_echo_out_t) -> Self {
        Self {
            accepted: o.accepted != 0,
            posterior_prob: o.posterior_prob,
            log_odds: o.log_odds,
            source_reliability: o.source_reliability,
            unique_count: o.unique_count,
        }
    }
}

/// Provenance-deduplicated multi-agent evidence pool with Beta-Bernoulli peer reliability.
pub struct EchoPool {
    inner: Box<ffi::eproc_echopool_t>,
}

impl EchoPool {
    pub fn new(n_sources: usize, n_claims: usize, prior_prob: f64) -> Result<Self> {
        let ns = u32::try_from(n_sources).map_err(|_| Error::Invalid)?;
        let nc = u32::try_from(n_claims).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_echopool_t>();
        let rc = unsafe { ffi::eproc_echopool_init(inner.as_mut(), ns, nc, prior_prob) };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn submit(
        &mut self,
        source_id: usize,
        claim_id: usize,
        evidence_id: u64,
        llr: f64,
    ) -> Result<EchoOutcome> {
        let src = u32::try_from(source_id).map_err(|_| Error::Invalid)?;
        let clm = u32::try_from(claim_id).map_err(|_| Error::Invalid)?;
        let mut out = ffi::eproc_echo_out_t::default();
        let rc = unsafe {
            ffi::eproc_echopool_submit(self.inner.as_mut(), src, clm, evidence_id, llr, &mut out)
        };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn resolve(&mut self, claim_id: usize, outcome_positive: bool) -> Result<()> {
        let clm = u32::try_from(claim_id).map_err(|_| Error::Invalid)?;
        let rc = unsafe {
            ffi::eproc_echopool_resolve(
                self.inner.as_mut(),
                clm,
                if outcome_positive { 1 } else { 0 },
            )
        };
        check_rc(rc)
    }

    #[inline]
    pub fn posterior(&self, claim_id: usize) -> f64 {
        match u32::try_from(claim_id) {
            Ok(c) => unsafe { ffi::eproc_echopool_posterior(self.inner.as_ref(), c) },
            Err(_) => f64::NAN,
        }
    }

    #[inline]
    pub fn source_reliability(&self, source_id: usize) -> f64 {
        match u32::try_from(source_id) {
            Ok(s) => unsafe { ffi::eproc_echopool_source_reliability(self.inner.as_ref(), s) },
            Err(_) => f64::NAN,
        }
    }

    #[inline]
    pub fn n_accepted(&self) -> u64 {
        self.inner.n_accepted
    }

    #[inline]
    pub fn n_echo_dropped(&self) -> u64 {
        self.inner.n_echo_dropped
    }
}

// ============================================================================
// 4. VarOptSampler, OnlineRLSSampler, SiftVOISelector
// ============================================================================

/// Cohen-2011 variance-optimal weighted reservoir sampler with exact Horvitz-Thompson weights.
pub struct VarOptSampler {
    inner: Box<ffi::eproc_varopt_t>,
}

impl VarOptSampler {
    pub fn new(k: usize, seed: u64) -> Result<Self> {
        let k_u32 = u32::try_from(k).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_varopt_t>();
        let rc = unsafe { ffi::eproc_varopt_init(inner.as_mut(), k_u32, seed) };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn push(&mut self, item_id: u64, weight: f64, value: f64) -> Result<()> {
        let rc = unsafe { ffi::eproc_varopt_push(self.inner.as_mut(), item_id, weight, value) };
        check_rc(rc)
    }

    #[inline]
    pub fn estimate_sum(&self) -> f64 {
        unsafe { ffi::eproc_varopt_estimate_sum(self.inner.as_ref()) }
    }

    #[inline]
    pub fn estimate_mean(&self) -> f64 {
        unsafe { ffi::eproc_varopt_estimate_mean(self.inner.as_ref()) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.count as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.count == 0
    }

    #[inline]
    pub fn n_seen(&self) -> u64 {
        self.inner.n_seen
    }

    #[inline]
    pub fn items(&self) -> &[ffi::eproc_varopt_item_t] {
        &self.inner.items[..self.len()]
    }
}

/// Decision returned by [`OnlineRLSSampler::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RLSOutcome {
    pub keep: bool,
    pub leverage: f64,
    pub prob: f64,
    pub ht_weight: f64,
}

impl From<ffi::eproc_rls_out_t> for RLSOutcome {
    #[inline]
    fn from(o: ffi::eproc_rls_out_t) -> Self {
        Self {
            keep: o.keep != 0,
            leverage: o.leverage,
            prob: o.prob,
            ht_weight: o.ht_weight,
        }
    }
}

/// Online Ridge Leverage Score (RLS) causal subsampler with Sherman-Morrison updates.
pub struct OnlineRLSSampler {
    inner: Box<ffi::eproc_rls_t>,
}

impl OnlineRLSSampler {
    pub fn new(
        d: usize,
        lambda: f64,
        oversample: f64,
        min_prob: f64,
        seed: u64,
    ) -> Result<Self> {
        let d_u32 = u32::try_from(d).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_rls_t>();
        let rc = unsafe {
            ffi::eproc_rls_init(inner.as_mut(), d_u32, lambda, oversample, min_prob, seed)
        };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn step(&mut self, x: &[f64]) -> Result<RLSOutcome> {
        if x.len() != self.dim() {
            return Err(Error::Invalid);
        }
        let mut out = ffi::eproc_rls_out_t::default();
        let rc = unsafe { ffi::eproc_rls_step(self.inner.as_mut(), x.as_ptr(), &mut out) };
        check_rc(rc)?;
        Ok(out.into())
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.d as usize
    }

    #[inline]
    pub fn n_seen(&self) -> u64 {
        self.inner.n_seen
    }

    #[inline]
    pub fn n_kept(&self) -> u64 {
        self.inner.n_kept
    }

    #[inline]
    pub fn inv_a(&self) -> &[f64] {
        let d = self.dim();
        &self.inner.inv_a[..d * d]
    }
}

/// Resolution level recommended by [`SiftVOISelector::select_next`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiftActionType {
    ReadSiteMean,
    ReadChannelMean,
    ReadRawTicks,
}

/// Recommended next sensing action from [`SiftVOISelector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftAction {
    pub action_type: SiftActionType,
    pub target_idx: usize,
    pub evoi_per_cost: f64,
}

/// Hierarchical multi-resolution Bayes Value-of-Information (VOI) selector.
pub struct SiftVOISelector {
    inner: Box<ffi::eproc_sift_t>,
}

impl SiftVOISelector {
    pub fn new(
        n_sites: usize,
        per_site: usize,
        signal_mean: f64,
        noise_sd: f64,
    ) -> Result<Self> {
        let ns = u32::try_from(n_sites).map_err(|_| Error::Invalid)?;
        let ps = u32::try_from(per_site).map_err(|_| Error::Invalid)?;
        let mut inner = alloc_zeroed_box::<ffi::eproc_sift_t>();
        let rc =
            unsafe { ffi::eproc_sift_init(inner.as_mut(), ns, ps, signal_mean, noise_sd) };
        check_rc(rc)?;
        Ok(Self { inner })
    }

    #[inline]
    pub fn set_noise_scale(&mut self, channel_idx: usize, noise_scale: f64) -> Result<()> {
        let idx = u32::try_from(channel_idx).map_err(|_| Error::Invalid)?;
        let rc = unsafe { ffi::eproc_sift_set_noise_scale(self.inner.as_mut(), idx, noise_scale) };
        check_rc(rc)
    }

    #[inline]
    pub fn observe_channel(
        &mut self,
        channel_idx: usize,
        obs_mean: f64,
        n_ticks: u32,
    ) -> Result<()> {
        let idx = u32::try_from(channel_idx).map_err(|_| Error::Invalid)?;
        let rc = unsafe {
            ffi::eproc_sift_observe_channel(self.inner.as_mut(), idx, obs_mean, n_ticks)
        };
        check_rc(rc)
    }

    #[inline]
    pub fn observe_site(&mut self, site_idx: usize, site_mean: f64) -> Result<()> {
        let idx = u32::try_from(site_idx).map_err(|_| Error::Invalid)?;
        let rc = unsafe { ffi::eproc_sift_observe_site(self.inner.as_mut(), idx, site_mean) };
        check_rc(rc)
    }

    #[inline]
    pub fn select_next(
        &self,
        cost_site: f64,
        cost_channel: f64,
        cost_ticks: f64,
        ticks_per_read: u32,
    ) -> Result<SiftAction> {
        let mut act = ffi::eproc_sift_action_t::default();
        let rc = unsafe {
            ffi::eproc_sift_select_next(
                self.inner.as_ref(),
                cost_site,
                cost_channel,
                cost_ticks,
                ticks_per_read,
                &mut act,
            )
        };
        check_rc(rc)?;
        let action_type = match act.action_type {
            0 => SiftActionType::ReadSiteMean,
            1 => SiftActionType::ReadChannelMean,
            2 => SiftActionType::ReadRawTicks,
            other => return Err(Error::Unknown(other)),
        };
        Ok(SiftAction {
            action_type,
            target_idx: act.target_idx as usize,
            evoi_per_cost: act.evoi_per_cost,
        })
    }

    #[inline]
    pub fn site_prob(&self, site_idx: usize) -> Option<f64> {
        if site_idx >= self.inner.n_sites as usize {
            return None;
        }
        Some(self.inner.site_log_prob[site_idx].exp())
    }

    #[inline]
    pub fn channel_prob(&self, channel_idx: usize) -> Option<f64> {
        let total = (self.inner.n_sites as usize) * (self.inner.per_site as usize);
        if channel_idx >= total {
            return None;
        }
        Some(self.inner.chan_prob[channel_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn u01(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        }

        fn norm(&mut self) -> f64 {
            let u1 = self.u01().max(1e-15);
            let u2 = self.u01();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn test_abi_version_simd_and_struct_sizes() {
        assert_eq!(abi_version(), ffi::EPROC_ABI_VERSION);
        assert_eq!(ConformalMonoid::wire_size(), 640);
        assert_eq!(size_of::<ffi::eproc_bet_t>(), 176);
        assert_eq!(size_of::<ffi::eproc_bet_result_t>(), 56);
        assert_eq!(size_of::<ffi::eproc_cs_t>(), 168);
        assert_eq!(size_of::<ffi::eproc_cs_result_t>(), 40);
        assert_eq!(size_of::<ffi::eproc_edetect_t>(), 224);
        assert_eq!(size_of::<ffi::eproc_edetect_result_t>(), 32);
        assert_eq!(size_of::<ffi::eproc_race_t>(), 11_384);
        assert_eq!(size_of::<ffi::eproc_race_status_t>(), 48);
        assert_eq!(size_of::<ffi::eproc_conformal_t>(), 4_888);
        assert_eq!(size_of::<ffi::eproc_conformal_out_t>(), 128);
        assert_eq!(size_of::<ffi::eproc_delay_slot_t>(), 32);
        assert_eq!(size_of::<ffi::eproc_delayed_conformal_t>(), 135_976);
        assert_eq!(size_of::<ffi::eproc_budget_conformal_t>(), 4_912);
        assert_eq!(size_of::<ffi::eproc_cmonoid_t>(), 640);
        assert_eq!(size_of::<ffi::eproc_boa_t>(), 3_488);
        assert_eq!(size_of::<ffi::eproc_boa_step_t>(), 48);
        assert_eq!(size_of::<ffi::eproc_echo_slot_t>(), 40);
        assert_eq!(size_of::<ffi::eproc_echopool_t>(), 178_208);
        assert_eq!(size_of::<ffi::eproc_echo_out_t>(), 40);
        assert_eq!(size_of::<ffi::eproc_varopt_item_t>(), 40);
        assert_eq!(size_of::<ffi::eproc_varopt_t>(), 163_904);
        assert_eq!(size_of::<ffi::eproc_rls_t>(), 8_256);
        assert_eq!(size_of::<ffi::eproc_rls_out_t>(), 32);
        assert_eq!(size_of::<ffi::eproc_sift_t>(), 270_384);
        assert_eq!(size_of::<ffi::eproc_sift_action_t>(), 16);

        let flags = simd_flags();
        assert_eq!(flags & !(SIMD_AVX2 | SIMD_FMA | SIMD_NEON), 0);
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                assert_ne!(
                    flags & SIMD_AVX2,
                    0,
                    "runtime AVX2 dispatch must activate on AVX2-capable x86_64 CPU"
                );
            }
        }
    }

    #[test]
    fn test_eprocess_ville_rejection_and_combiners() {
        let mut rng = SplitMix64(12_345);
        // Anytime-valid Type-I error under continuous peeking.
        let mut false_rejects = 0usize;
        let trials = 300;
        for _ in 0..trials {
            let mut ep = EProcess::new(0.5, 0.05, 0.0, 1.0, Side::Upper).unwrap();
            let mut last = ep.query();
            for _ in 0..300 {
                let x = if rng.u01() < 0.5 { 1.0 } else { 0.0 };
                last = ep.update(x).unwrap();
            }
            if last.rejected {
                false_rejects += 1;
            }
        }
        assert!((false_rejects as f64) / (trials as f64) <= 0.06);

        // Power under H1 + NaN/Inf sanitization + scalar/batch bit-identity.
        let mut xs = Vec::with_capacity(260);
        for _ in 0..250 {
            xs.push(if rng.u01() < 0.72 { 1.0 } else { 0.0 });
        }
        xs.extend_from_slice(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-315, 1e15, -1e15]);
        xs.extend_from_slice(&[0.8, 0.8, 0.8, 0.8]);

        let mut ep_s = EProcess::new(0.5, 0.01, 0.0, 1.0, Side::Upper).unwrap();
        let mut ep_b = EProcess::new(0.5, 0.01, 0.0, 1.0, Side::Upper).unwrap();
        let mut rs = ep_s.query();
        for &x in &xs {
            rs = ep_s.update(x).unwrap();
        }
        let rb = ep_b.update_batch(&xs).unwrap();
        assert!(rs.rejected && rs.p_value < 0.01);
        assert_eq!(ep_s.n(), 257);
        assert_eq!(ep_s.n_invalid(), 3);
        assert!(ep_s.rejected_at().is_some());
        assert_eq!(rs.log_e.to_bits(), rb.log_e.to_bits());
        assert_eq!(rs.p_value.to_bits(), rb.p_value.to_bits());

        assert!((combine_independent_log(2.5, 3.5) - 6.0).abs() < 1e-12);
        let dep = combine_dependent_log(&[0.0, 3.0_f64.ln(), 5.0_f64.ln()]);
        assert!((dep - 3.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_confidence_sequence_eb_and_normal() {
        let mut rng = SplitMix64(54_321);
        let mut eb = ConfidenceSequence::empirical_bernstein(0.05, 0.0, 1.0).unwrap();
        let mut nm = ConfidenceSequence::normal_mixture(0.05, 1.0, 100.0).unwrap();
        let mut prev_w = f64::INFINITY;
        for _ in 0..300 {
            let x = if rng.u01() < 0.35 { 1.0 } else { 0.0 };
            let r = eb.update(x).unwrap();
            assert!(r.width <= prev_w + 1e-15);
            prev_w = r.width;
            let _ = nm.update(2.5 + rng.norm()).unwrap();
        }
        let reb = eb.query();
        let rnm = nm.query();
        assert!(reb.low <= 0.35 && reb.high >= 0.35);
        assert!(rnm.low <= 2.5 && rnm.high >= 2.5);
    }

    #[test]
    fn test_edetector_and_agent_race() {
        let mut rng = SplitMix64(99_999);
        let mut ed = EDetector::new(0.5, 0.01, 0.0, 1.0, Side::Upper).unwrap();
        for _ in 0..200 {
            let x = if rng.u01() < 0.5 { 1.0 } else { 0.0 };
            let r = ed.update(x).unwrap();
            assert!(!r.alarm);
        }
        let mut detected = None;
        for t in 0..200 {
            let x = if rng.u01() < 0.85 { 1.0 } else { 0.0 };
            if ed.update(x).unwrap().alarm && detected.is_none() {
                detected = Some(t);
            }
        }
        assert!(detected.expect("drift alarm") < 80);
        ed.reset_alarm();
        assert_eq!(ed.log_cusum(), 0.0);
        assert_eq!(ed.alarm_at(), None);

        let means = [0.25, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.88];
        let mut race = AgentRace::new(8, 0.05, 0.0, 1.0, 0.0).unwrap();
        let mut st = race.status();
        for _ in 0..5000 {
            let arm = race.suggest_arm().unwrap();
            let rew = if rng.u01() < means[arm] { 1.0 } else { 0.0 };
            st = race.update(arm, rew).unwrap();
            if st.winner.is_some() {
                break;
            }
        }
        assert_eq!(st.winner, Some(7));
        assert_eq!(st.n_active, 1);
        let (arm7, active7) = race.arm_state(7).unwrap();
        assert!(active7 && arm7.mean > 0.75);
    }

    #[test]
    fn test_five_rate_crossing_free_conformal_and_variants() {
        let mut rng = SplitMix64(77_777);
        let unsorted = [0.05, 0.50, 0.01, 0.20, 0.10];
        let mut cp = MultiRateConformal::new(&unsorted, 128, 0.05).unwrap();
        assert_eq!(cp.target_alphas(), &[0.50, 0.20, 0.10, 0.05, 0.01]);

        let mut last = cp.update(0.5).unwrap();
        for t in 0..800 {
            let scale = if t < 400 { 1.0 } else { 3.0 };
            let s = match t {
                123 => f64::NAN,
                456 => f64::INFINITY,
                _ => rng.norm().abs() * scale,
            };
            last = cp.update(s).unwrap();
            let qs = last.quantiles();
            for r in 1..5 {
                assert!(qs[r] >= qs[r - 1], "quantiles must never cross");
            }
            let iv = cp.predict(10.0, 1.0);
            for r in 1..5 {
                assert!(iv.lows()[r] <= iv.lows()[r - 1]);
                assert!(iv.highs()[r] >= iv.highs()[r - 1]);
            }
        }
        let cov90 = last.emp_coverage()[2];
        assert!(cov90 > 0.75 && cov90 < 0.98);

        // Delayed conformal causality check.
        let mut dc = DelayedConformal::new(&[0.20, 0.10], 64, 0.02).unwrap();
        let t0 = dc.issue(5.0, 1.0).unwrap();
        let t1 = dc.issue(5.0, 1.0).unwrap();
        assert_eq!(t0.lows(), t1.lows());
        assert_eq!(t0.highs(), t1.highs());
        assert_eq!(dc.pending_count(), 2);
        dc.reveal(t1.ticket_id, 5.8).unwrap();
        dc.reveal(t0.ticket_id, 5.4).unwrap();
        assert_eq!(dc.pending_count(), 0);

        // Right-censored budget conformal.
        let mut bc = BudgetConformal::new(0.10, 64, 1.5).unwrap();
        for i in 0..40 {
            let cens = i % 5 == 0;
            bc.update(100.0, if cens { 120.0 } else { 105.0 }, cens)
                .unwrap();
        }
        assert!(bc.bound(100.0) > 100.0);
        assert_eq!(bc.n_censored(), 8);
    }

    #[test]
    fn test_conformal_monoid_abelian_640b_wire() {
        let mut rng = SplitMix64(88_888);
        let mut a = ConformalMonoid::new(1.0);
        let mut b = ConformalMonoid::new(1.0);
        let mut c = ConformalMonoid::new(1.0);
        for _ in 0..500 {
            a.add(rng.u01() * 10.0);
            b.add(rng.u01() * 10.0);
            c.add(rng.u01() * 20.0 - 5.0);
        }
        a.add(f64::NAN);
        a.add(f64::INFINITY);
        assert_eq!(a.count(), 500);

        // Associativity: (A + B) + C == A + (B + C)
        let ab = a.merge(&b);
        let abc1 = ab.merge(&c);
        let abc2 = a.merge(&b.merge(&c));
        assert_eq!(abc1.count(), abc2.count());
        assert_eq!(abc1.bin_counts(), abc2.bin_counts());

        // Commutativity: A + B == B + A
        let ba = b.merge(&a);
        assert_eq!(ab.bin_counts(), ba.bin_counts());

        // Exact abelian leave-one-out subtraction: (A + B) - A == B
        let loo = ab.subtract(&a).unwrap();
        assert_eq!(loo.count(), b.count());
        assert_eq!(loo.bin_counts(), b.bin_counts());

        // Underflow protection
        assert_eq!(a.subtract(&ab).err(), Some(Error::Invalid));

        // Quantile accuracy on Uniform(0, 10)
        assert!((ab.quantile(0.50) - 5.0).abs() < 0.7);
        assert!((ab.quantile(0.90) - 9.0).abs() < 0.9);

        // 640-byte wire roundtrip & bit-flip corruption rejection
        let mut wire: [u8; 640] = abc1.to_bytes();
        let decoded = ConformalMonoid::from_bytes(&wire).unwrap();
        assert_eq!(decoded.count(), abc1.count());
        assert_eq!(decoded.bin_counts(), abc1.bin_counts());

        wire[100] ^= 0x01;
        assert_eq!(ConformalMonoid::from_bytes(&wire).err(), Some(Error::Mismatch));
        assert_eq!(ConformalMonoid::deserialize(&wire[..639]).err(), Some(Error::Invalid));
    }

    #[test]
    fn test_safe_fuser_byzantine_quarantine_and_seqlock() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SafeFuser>();
        assert_send_sync::<SafeFuserReader>();

        let mut rng = SplitMix64(314_159);
        let mut fuser = SafeFuser::new(8, 0, 0.05).unwrap();
        let mut preds = [0.0f64; 8];

        for t in 0..100 {
            let y = 10.0 + (t as f64 * 0.1).sin();
            preds[0] = y + 0.5 * rng.norm();
            preds[1] = y + 0.4 * rng.norm();
            preds[2] = y + 0.3 * rng.norm();
            preds[3] = y + 0.05 * rng.norm(); // best expert
            preds[4] = y + 0.8 * rng.norm();
            preds[5] = if (70..=74).contains(&t) {
                y + 50.0
            } else {
                y + 0.5 * rng.norm()
            };
            preds[6] = if t == 60 { f64::NAN } else { y + 0.5 * rng.norm() };
            preds[7] = if t == 50 { 1e6 } else { y + 0.5 * rng.norm() };

            let p_pred = fuser.predict(&preds).unwrap();
            if t == 50 {
                assert!(
                    (p_pred - y).abs() < 5.0,
                    "1e6 Byzantine spike must have ~0 blast radius"
                );
            }
            let st = fuser.update(&preds, y).unwrap();
            assert!(st.fused_pred.is_finite());
        }
        assert!(fuser.is_quarantined(7));
        assert!(fuser.is_quarantined(6));
        assert!(fuser.is_quarantined(5));
        assert!(fuser.weights()[3] > 0.35);

        // Multi-threaded Boehm-2012 atomic seqlock stress test (4 readers + 1 writer).
        let mut live = SafeFuser::new(8, 0, 0.05).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let torn = Arc::new(AtomicU64::new(0));

        let mut readers = Vec::new();
        for _ in 0..4 {
            let reader = live.reader();
            let torn_c = Arc::clone(&torn);
            readers.push(thread::spawn(move || {
                let mut local_torn = 0u64;
                for _ in 0..50_000 {
                    let snap = reader.snapshot();
                    let mut sum = 0.0;
                    let mut bad = false;
                    for (i, &w) in snap.weights().iter().enumerate() {
                        if !w.is_finite() || w < 0.0 || w > 1.0 + 1e-12 {
                            bad = true;
                            break;
                        }
                        if snap.fallback_active {
                            let expected = if i == 0 { 1.0 } else { 0.0 };
                            if w != expected {
                                bad = true;
                                break;
                            }
                        }
                        sum += w;
                    }
                    if bad || (sum - 1.0).abs() > 1e-9 {
                        local_torn += 1;
                    }
                }
                torn_c.fetch_add(local_torn, Ordering::Relaxed);
            }));
        }

        let stop_w = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            let mut wrng = SplitMix64(0xCAFE_BABE);
            let mut p = [0.0f64; 8];
            let mut n = 0u64;
            while !stop_w.load(Ordering::Acquire) {
                let y = (n as f64 * 0.01).sin() * 5.0;
                for (i, slot) in p.iter_mut().enumerate() {
                    *slot = y + (0.2 + 0.3 * i as f64) * wrng.norm();
                }
                live.update(&p, y).unwrap();
                n += 1;
            }
            live
        });

        for h in readers {
            h.join().unwrap();
        }
        stop.store(true, Ordering::Release);
        let _live = writer.join().unwrap();
        assert_eq!(torn.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_echopool_dedup_and_reliability() {
        let mut pool = EchoPool::new(16, 32, 0.5).unwrap();
        let o1 = pool.submit(0, 0, 1001, 2.0).unwrap();
        assert!(o1.accepted && o1.unique_count == 1);

        for s in 1..10 {
            let dup = pool.submit(s, 0, 1001, 2.0).unwrap();
            assert!(!dup.accepted);
            assert_eq!(dup.posterior_prob.to_bits(), o1.posterior_prob.to_bits());
        }
        assert_eq!(pool.n_echo_dropped(), 9);

        for c in 1..=10 {
            pool.submit(0, c, 2000 + c as u64, 2.0).unwrap();
            pool.submit(1, c, 3000 + c as u64, -2.0).unwrap();
            pool.resolve(c, true).unwrap();
            pool.resolve(c, true).unwrap(); // idempotent
        }
        assert!(pool.source_reliability(0) > 0.99);
        assert_eq!(pool.source_reliability(1), 0.0);
        assert!(pool.posterior(0) > 0.5);
    }

    #[test]
    fn test_varopt_rls_and_sift_voi() {
        let mut rng = SplitMix64(2026);
        let mut weights = [0.0f64; 200];
        let mut values = [0.0f64; 200];
        let mut true_w = 0.0;
        for i in 0..200 {
            weights[i] = if i % 20 == 0 {
                50.0
            } else {
                0.5 + 3.0 * rng.u01()
            };
            values[i] = 1.0 + 0.1 * i as f64;
            true_w += weights[i];
        }
        let mut vo = VarOptSampler::new(32, 42).unwrap();
        for i in 0..200 {
            vo.push(i as u64, weights[i], values[i]).unwrap();
        }
        assert_eq!(vo.len(), 32);
        let adj_sum: f64 = vo.items().iter().map(|it| it.adj_weight).sum();
        assert!((adj_sum - true_w).abs() < 1e-8 * true_w);
        assert!(vo.estimate_sum().is_finite() && vo.estimate_mean().is_finite());

        // Online RLS
        let mut rls = OnlineRLSSampler::new(16, 1.0, 2.0, 0.05, 1337).unwrap();
        let mut x = [0.0f64; 16];
        for _ in 0..100 {
            for v in &mut x {
                *v = rng.norm();
            }
            let out = rls.step(&x).unwrap();
            assert!(out.leverage >= 0.0 && out.prob >= 0.05 && out.prob <= 1.0);
            if out.keep {
                assert!((out.ht_weight - 1.0 / out.prob).abs() < 1e-12);
            } else {
                assert_eq!(out.ht_weight, 0.0);
            }
        }
        assert_eq!(rls.n_seen(), 100);
        assert!(rls.n_kept() > 0);

        // SiftVOI hierarchical zoom + 3x loud distractor immunity
        let mut sf = SiftVOISelector::new(16, 8, 2.0, 1.0).unwrap();
        let act0 = sf.select_next(1.0, 1.0, 4.0, 16).unwrap();
        assert_eq!(act0.action_type, SiftActionType::ReadSiteMean);

        let loud_ch = 5 * 8;
        let clean_ch = 5 * 8 + 1;
        sf.set_noise_scale(loud_ch, 3.0).unwrap();
        sf.observe_site(5, 1.6).unwrap();
        assert!(sf.site_prob(5).unwrap() > 0.5);

        let act1 = sf.select_next(1.0, 1.0, 4.0, 16).unwrap();
        assert_eq!(act1.action_type, SiftActionType::ReadChannelMean);
        assert_eq!(act1.target_idx, clean_ch);

        sf.observe_channel(clean_ch, 2.5, 4).unwrap();
        assert!(sf.channel_prob(clean_ch).unwrap() > 0.95);
    }

    #[test]
    fn test_safe_fuser_ville_fallback() {
        let mut fuser = SafeFuser::new(4, 0, 0.05).unwrap();
        for _ in 0..40 {
            fuser.update(&[1.0, 0.0, 0.5, 0.5], 0.0).unwrap();
        }
        let mut tripped = false;
        for _ in 0..150 {
            let st = fuser.update(&[0.0, 3.0, 3.0, 3.0], 0.0).unwrap();
            if st.fallback_active {
                tripped = true;
                break;
            }
        }
        assert!(tripped, "Ville deg_e must trip fallback on degradation");
        assert!(fuser.fallback_active());
        let snap = fuser.snapshot();
        assert!(snap.fallback_active);
        assert_eq!(snap.weights(), &[1.0, 0.0, 0.0, 0.0]);
    }
}

