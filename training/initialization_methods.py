"""Lightweight initialization-method vocabulary shared by artifact readers."""

ACTOR_BODY_FINETUNE = "actor_body_finetune_v1"
FROZEN_ACTOR_PROBE = "frozen_actor_probe_v1"
FULL_ACTOR_FRESH_CRITIC = "full_actor_fresh_critic_v1"
ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC = (
    "actor_body_fresh_head_copied_critic_v1"
)
SELECTIVE_INITIALIZATION_METHODS = frozenset({
    ACTOR_BODY_FINETUNE,
    FROZEN_ACTOR_PROBE,
    FULL_ACTOR_FRESH_CRITIC,
    ACTOR_BODY_FRESH_HEAD_COPIED_CRITIC,
})
TRANSFER_INITIALIZATION_METHODS = frozenset({
    "full_policy",
    *SELECTIVE_INITIALIZATION_METHODS,
})
