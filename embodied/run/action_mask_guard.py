def require_formal_actor(agent, context):
  if not hasattr(agent, "get_avail_actor_enabled"):
    raise TypeError(
        f"{context} requires an agent that exposes "
        "get_avail_actor_enabled()")
  if not agent.get_avail_actor_enabled():
    raise RuntimeError(
        f"{context} requires a formal masked-actor checkpoint; "
        "loaded checkpoint has avail_actor_gate/value=0")
