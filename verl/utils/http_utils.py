# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Centralized HTTP utility module with Ray-based distributed dispatch.

Provides module-level ``http_post()`` and ``http_get()`` async functions.
Requests are dispatched to ``_HttpClientActor`` Ray actors spread across
cluster nodes (round-robin).

Actors are created with deterministic names so that **any** Ray process
(driver or worker) can discover them via ``ray.get_actor()``.
"""

import asyncio
import json as json_mod
import random
from typing import Any

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_ACTOR_NAME_PREFIX = "_paddlerl_http_client_"

_actors: list[Any] = []
_actor_idx: int = 0
_num_actors: int = 0  # set by init, read by _resolve_actors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_distributed_http_client(num_actors_per_node: int = 1, max_connections: int = 0) -> None:
    """Initialize distributed Ray actors for HTTP requests.

    Creates named ``_HttpClientActor`` actors spread across all alive Ray
    nodes.  The actors are registered with deterministic names so that any
    Ray process can later discover them via ``ray.get_actor()``.

    This function is idempotent: calling it multiple times is safe.

    Args:
        num_actors_per_node: Number of ``_HttpClientActor`` actors to create
            per alive Ray node.
        max_connections: Max concurrent connections per actor.
            ``<= 0`` means unlimited (``None``).
    """
    global _actors, _num_actors
    if _actors:
        return  # Already initialized

    if max_connections <= 0:
        max_connections = None

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP client actors.")

    @ray.remote
    class _HttpClientActor:
        def __init__(self, max_connections: int | None):
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max(1, max_connections) if max_connections else None,
                    max_keepalive_connections=max(1, max_connections) if max_connections else None,
                ),
                timeout=httpx.Timeout(None),
            )

        async def do_post(
            self,
            url,
            *,
            json=None,
            data=None,
            content=None,
            headers=None,
            timeout=None,
            max_retries=1,
            params=None,
            retry_interval=1.0,
            exp_backoff=True,
        ):
            """Send POST request."""
            return await _do_request(
                self._client,
                "POST",
                url,
                json=json,
                data=data,
                content=content,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                params=params,
                retry_interval=retry_interval,
                exp_backoff=exp_backoff,
            )

        async def do_get(
            self,
            url,
            *,
            params=None,
            json=None,
            headers=None,
            timeout=None,
            max_retries=1,
            retry_interval=1.0,
            exp_backoff=True,
        ):
            """Send GET request."""
            return await _do_request(
                self._client,
                "GET",
                url,
                params=params,
                json=json,
                headers=headers,
                timeout=timeout,
                max_retries=max_retries,
                retry_interval=retry_interval,
                exp_backoff=exp_backoff,
            )

    created = []
    if max_connections is not None:
        per_actor_conc = (max_connections + len(nodes) - 1) // len(nodes)
    else:
        per_actor_conc = None

    idx = 0
    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(num_actors_per_node):
            actor = _HttpClientActor.options(
                name=f"{_ACTOR_NAME_PREFIX}{idx}",
                lifetime="detached",
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc if per_actor_conc else 1000,
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)
            idx += 1

    _actors = created
    _num_actors = len(created)
    print(f"[http_util] Created {len(created)} distributed HTTP actors across {len(nodes)} nodes")


async def close_http_client() -> None:
    """Kill all distributed actors."""
    global _actors, _actor_idx, _num_actors

    if _actors:
        for idx, actor in enumerate(_actors):
            try:
                ray.kill(actor)
            except Exception:
                print(f"Failed to kill distributed HTTP client actor: {_ACTOR_NAME_PREFIX}{idx}")
                pass
        _actors = []
    _actor_idx = 0
    _num_actors = 0


async def http_post(
    url: str,
    *,
    json: Any = None,
    data: Any = None,
    content: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    max_retries: int = 1,
    params: dict[str, Any] | None = None,
    retry_interval: float = 1.0,
    exp_backoff: bool = True,
) -> Any:
    """Send an HTTP POST request via a distributed Ray actor.

    The request is dispatched to a Ray actor selected by round-robin.

    Args:
        retry_interval: Retry interval in seconds (default ``1.0``).  When
            ``exp_backoff`` is ``False``, this is used as a fixed sleep
            between retries.  When ``exp_backoff`` is ``True``, this is the
            *initial* interval for exponential backoff with jitter.
        exp_backoff: Enable exponential backoff with full-jitter strategy.

    Returns:
        Parsed JSON response, or raw text if JSON decoding fails.
    """
    actor = _next_actor()
    return await actor.do_post.remote(
        url,
        json=json,
        data=data,
        content=content,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        params=params,
        retry_interval=retry_interval,
        exp_backoff=exp_backoff,
    )


async def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    max_retries: int = 1,
    retry_interval: float = 1.0,
    exp_backoff: bool = True,
) -> Any:
    """Send an HTTP GET request via a distributed Ray actor.

    The request is dispatched to a Ray actor selected by round-robin.

    Args:
        retry_interval: Retry interval in seconds (default ``1.0``).  When
            ``exp_backoff`` is ``False``, this is used as a fixed sleep
            between retries.  When ``exp_backoff`` is ``True``, this is the
            *initial* interval for exponential backoff with jitter.
        exp_backoff: Enable exponential backoff with full-jitter strategy.

    Returns:
        Parsed JSON response, or raw text if JSON decoding fails.
    """
    actor = _next_actor()
    return await actor.do_get.remote(
        url,
        params=params,
        json=json,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        retry_interval=retry_interval,
        exp_backoff=exp_backoff,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_actors() -> list[Any]:
    """Discover existing ``_HttpClientActor`` actors by name.

    Called lazily the first time ``_next_actor()`` is invoked in a process
    that did not call ``init_distributed_http_client()`` itself (e.g. a
    Ray worker actor).  Looks up actors by their deterministic names
    ``_paddlerl_http_client_0``, ``_paddlerl_http_client_1``, … until a
    lookup fails.
    """
    actors: list[Any] = []
    idx = 0
    while True:
        try:
            actor = ray.get_actor(f"{_ACTOR_NAME_PREFIX}{idx}")
            actors.append(actor)
            idx += 1
        except ValueError:
            break
    if not actors:
        raise RuntimeError(
            "[http_util] No HTTP client actors found. Ensure init_distributed_http_client() was called in the driver."
        )
    print(f"[http_util] Resolved {len(actors)} existing HTTP client actors by name")
    return actors


def _next_actor():
    """Round-robin select the next distributed actor.

    On first call in a worker process, lazily resolves actors by name.
    """
    global _actors, _actor_idx
    if not _actors:
        _actors = _resolve_actors()
    actor = _actors[_actor_idx % len(_actors)]
    _actor_idx = (_actor_idx + 1) % len(_actors)
    return actor


async def _do_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: Any = None,
    data: Any = None,
    content: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    max_retries: int = 1,
    params: dict[str, Any] | None = None,
    retry_interval: float = 1.0,
    exp_backoff: bool = True,
) -> Any:
    """Core request function with retry logic."""
    req_timeout = httpx.Timeout(timeout) if timeout is not None else None

    # For GET with json body, httpx doesn't support json= on get(),
    # so we encode it as content with the appropriate header.
    effective_content = content
    effective_json = json
    effective_headers = dict(headers) if headers else {}
    if method.upper() == "GET" and json is not None:
        effective_content = json_mod.dumps(json).encode()
        effective_headers.setdefault("Content-Type", "application/json")
        effective_json = None

    retry_count = 0
    while True:
        try:
            response = await client.request(
                method,
                url,
                json=effective_json,
                data=data,
                content=effective_content,
                headers=effective_headers or None,
                params=params,
                timeout=req_timeout,
            )
            response.raise_for_status()
            raw = await response.aread()
            try:
                return json_mod.loads(raw)
            except (json_mod.JSONDecodeError, UnicodeDecodeError):
                return raw.decode() if isinstance(raw, bytes) else raw
        except Exception as e:
            retry_count += 1
            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None
            print(
                f"[http_util] {method} {url} failed "
                f"(attempt {retry_count}/{max_retries}): {e}"
                f"{f', response={response_text}' if response_text else ''}"
            )
            if retry_count >= max_retries:
                raise
            if exp_backoff:
                # Full-jitter exponential backoff:
                cap = retry_interval * (2**retry_count)
                delay = random.uniform(0, cap)
            else:
                delay = retry_interval
            await asyncio.sleep(delay)
