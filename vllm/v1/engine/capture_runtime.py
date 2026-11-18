# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional engine-local capture I/O, isolated from the GPU scheduling loop."""

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Protocol

import msgspec

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest

logger = init_logger(__name__)


class CaptureAdapter(Protocol):
    """RL owns the staging schema; vLLM supplies native request/output objects."""

    async def prepare(self, request: "EngineCoreRequest") -> None: ...

    def observe(self, output: "EngineCoreOutput") -> None: ...

    async def finalize(
        self, request_id: str, served_tokens: int, prefill_key: str | None
    ) -> dict[str, Any]: ...

    def finish(self, request_id: str) -> None: ...

    def discard(self, request_id: str) -> None: ...

    async def close(self) -> None: ...


class EngineCapture:
    """Run asynchronous storage work without blocking input or output sockets."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.adapter: CaptureAdapter | None = None
        self.pending: dict[str, asyncio.Task] = {}
        self.failure: BaseException | None = None
        self.writes: dict[str, asyncio.Task] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._closed_admission = False
        self._admission_lock = threading.Lock()
        self._preparing: set[str] = set()
        self._active: set[str] = set()
        self.thread = threading.Thread(
            target=self._run, name="engine-capture", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def configure(self, factory: str, config: dict[str, Any]) -> Future:
        """Load the operator-selected adapter once, via the engine utility API."""

        async def configure() -> bool:
            if self.adapter is not None:
                raise RuntimeError("engine capture is already configured")
            self.adapter = await resolve_obj_by_qualname(factory)(config)
            return True

        return asyncio.run_coroutine_threadsafe(configure(), self.loop)

    def _adapter(self) -> CaptureAdapter:
        if self.failure is not None:
            raise RuntimeError("engine capture failed") from self.failure
        if self.adapter is None:
            raise RuntimeError("engine capture must be configured before requests")
        return self.adapter

    def prepare(
        self,
        request: "EngineCoreRequest",
        ready: Callable[[], None],
        failed: Callable[[], None],
    ) -> None:
        with self._admission_lock:
            if self._closed_admission:
                raise RuntimeError("engine capture is paused")
            if request.request_id in self._active:
                raise ValueError("duplicate capture request")
            self._active.add(request.request_id)
            self._preparing.add(request.request_id)

        async def prepare() -> None:
            scheduled = False
            try:
                context = request.capture_context
                assert context is not None
                await self._adapter().prepare(request)
                # Serialize ADD publication with input-thread ABORT admission.
                # Otherwise a slow lookup could enqueue ADD after its ABORT.
                with self._admission_lock:
                    if request.request_id in self._preparing:
                        self._metadata[request.request_id] = {
                            "request_id": request.request_id,
                            "prompt_tokens": len(request.prompt_token_ids or []),
                            "phase": context.phase,
                            "prefill_key": context.prefill_key,
                        }
                        ready()
                        scheduled = True
                    else:
                        self._adapter().discard(request.request_id)
            except asyncio.CancelledError:
                if self.adapter is not None:
                    self.adapter.discard(request.request_id)
            except Exception:
                if self.adapter is not None:
                    self.adapter.discard(request.request_id)
                # Preserve the active exception for the engine's error reporter.
                failed()
            finally:
                with self._admission_lock:
                    self._preparing.discard(request.request_id)
                    if not scheduled:
                        self._active.discard(request.request_id)
                        self._metadata.pop(request.request_id, None)
                self.pending.pop(request.request_id, None)

        def start() -> None:
            with self._admission_lock:
                if request.request_id not in self._preparing:
                    self._active.discard(request.request_id)
                    return
            self.pending[request.request_id] = self.loop.create_task(prepare())

        self.loop.call_soon_threadsafe(start)

    def abort_pending(self, request_ids: list[str]) -> None:
        """Cancel unscheduled lookups; retain generated data for stop trimming."""
        with self._admission_lock:
            self._preparing.difference_update(request_ids)

        def abort() -> None:
            for request_id in request_ids:
                if task := self.pending.get(request_id):
                    task.cancel()

        self.loop.call_soon_threadsafe(abort)

    def wire_output(
        self, output: "EngineCoreOutput", engine_index: int
    ) -> "EngineCoreOutput":
        """Keep training arrays in Python; send only text tokens and coordination."""
        with self._admission_lock:
            metadata = self._metadata.get(output.request_id)
            if metadata is None:
                return output
            metadata = {**metadata, "engine_index": engine_index}
        return msgspec.structs.replace(
            output,
            new_logprobs=None,
            routed_experts=None,
            kv_transfer_params={
                **(output.kv_transfer_params or {}),
                "__nemo_rl_capture": metadata,
            },
        )

    def observe(self, output: "EngineCoreOutput") -> None:
        with self._admission_lock:
            if output.request_id not in self._active:
                return

        # CPU arrays are owned by EngineCoreOutput. Retain the object until the
        # adapter consumes it, before the frontend can request finalization.
        def observe() -> None:
            try:
                if self.adapter is not None:
                    self._adapter().observe(output)
            except Exception as error:
                self.failure = error
                logger.exception("Engine capture output failed: %s", output.request_id)

        self.loop.call_soon_threadsafe(observe)

    def finalize(
        self, request_id: str, served_tokens: int, prefill_key: str | None
    ) -> Future:
        """Acknowledge storage only after the frontend supplies its final length."""

        async def write() -> dict[str, Any]:
            try:
                result = await self._adapter().finalize(
                    request_id, served_tokens, prefill_key
                )
                if result.get("disposition") == "capture_failed":
                    raise RuntimeError("engine capture storage rejected the row")
                with self._admission_lock:
                    self._active.discard(request_id)
                    self._metadata.pop(request_id, None)
                return result
            except Exception as error:
                self.failure = error
                raise
            finally:
                self.writes.pop(request_id, None)

        async def finalize() -> dict[str, Any]:
            if request_id in self.writes:
                raise ValueError("capture finalization already pending")
            task = self.loop.create_task(write())
            self.writes[request_id] = task
            # The write and its cleanup stay tracked if a utility caller leaves.
            return await asyncio.shield(task)

        return asyncio.run_coroutine_threadsafe(finalize(), self.loop)

    def finish(self, request_id: str) -> None:
        # Called after the scheduler abort and all previously queued outputs.
        def finish() -> None:
            try:
                self._adapter().finish(request_id)
            except Exception as error:
                self.failure = error
                logger.exception("Could not finish engine capture: %s", request_id)

        self.loop.call_soon_threadsafe(finish)

    def drain(self) -> Future:
        """Close admission and wait for storage; unresolved calls block refit."""
        with self._admission_lock:
            self._closed_admission = True

        async def drain() -> bool:
            await asyncio.gather(*list(self.pending.values()))
            await asyncio.gather(*list(self.writes.values()))
            self._adapter()
            with self._admission_lock:
                if self._active:
                    raise RuntimeError(
                        f"cannot refit with {len(self._active)} unfinalized captures"
                    )
            return True

        return asyncio.run_coroutine_threadsafe(drain(), self.loop)

    def resume(self) -> None:
        if self.failure is not None:
            raise RuntimeError("cannot resume failed capture") from self.failure
        with self._admission_lock:
            if self._active:
                raise RuntimeError("cannot resume undrained capture")
            self._closed_admission = False

    def discard(self, request_id: str) -> Future:
        async def discard() -> None:
            if task := self.pending.get(request_id):
                task.cancel()
                await task
            # Cleanup must remain possible after an output-validation failure.
            if self.adapter is not None:
                self.adapter.discard(request_id)
            with self._admission_lock:
                self._active.discard(request_id)
                self._metadata.pop(request_id, None)

        return asyncio.run_coroutine_threadsafe(discard(), self.loop)

    def close(self) -> None:
        async def close() -> None:
            tasks = list(self.pending.values()) + list(self.writes.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.adapter is not None:
                await self.adapter.close()

        try:
            asyncio.run_coroutine_threadsafe(close(), self.loop).result(timeout=10)
        except Exception:
            # Storage teardown must not prevent executor/GPU cleanup.
            logger.exception("Engine capture shutdown failed")
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)
