/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed-capacity graph execution over two pinned transport frames.

mod manifest;
mod parallel;
mod rank_stream;
mod ranks;
mod readiness;
pub use readiness::{MAX_READINESS_INTERVALS, ReadinessIntervals};
pub mod rendezvous;
#[cfg(feature = "roce")]
pub mod roce;
pub mod roce_fabric;
mod threaded;
pub use rank_stream::RankStream;
pub use ranks::RankPeers;
mod transport;
pub(crate) use transport::{clear_pages, validate_speculative_survival};
mod worker;
pub use parallel::{TensorParallelExecutor, serve_rank};
pub(crate) use ranks::RankExecutor;
pub use transport::{
    CompletionRecord, ExecutionTiming, FrameCompletion, GraphExecutor, PromptLogprobs, TokenLogprobs,
    TokenVector, Upload,
};
pub use worker::{ExecutionWorker, OwnedUpload};

#[cfg(test)]
mod tests;

pub use manifest::{
    ABI_VERSION, BATCH_CAPACITIES, BufferDescriptor, CollectiveHealth, Dtype, ExecutionManifest,
    GraphDescriptor, TuningGraphDescriptor, TunedBinding, GraphKind, PrefixAllocation, InputBinding, LOGPROB_FIELDS, LOGPROB_PAIR_FIELDS,
    DEFAULT_COMPLETION_TOKEN_CAPACITY, COMPLETION_METADATA_FIELDS, resolve_completion_token_capacity, MAX_TOP_LOGPROBS, MAX_LOGPROB_TOKEN_IDS, PROMPT_FIELDS, REQUEST_LANES,
    SamplingMode,
};

/// Debug A/B (DeepSeek decode-corruption hunt, flare#193): `FLARE_SINGLE_SUBMISSION=1` keeps at most one
/// generation submission in flight, disabling the two-frame CPU/GPU overlap introduced in bce032c.
pub(crate) fn single_submission() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FLARE_SINGLE_SUBMISSION").is_some_and(|v| v != "0" && !v.is_empty()))
}
