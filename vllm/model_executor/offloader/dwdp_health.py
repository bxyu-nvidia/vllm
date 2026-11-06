# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitor same-host DWDP source processes without inference collectives."""

import os
import select
import threading


class PeerProcessMonitor:
    """Use Linux pidfds to detect exited owners, including idle weight servers."""

    def __init__(self, peers: dict[int, int]) -> None:
        self.failed_peer: int | None = None
        self._stop = threading.Event()
        self._fds: dict[int, int] = {}
        self._thread: threading.Thread | None = None
        poll = select.poll()
        try:
            for rank, pid in peers.items():
                try:
                    descriptor = os.pidfd_open(pid)
                except ProcessLookupError:
                    self.failed_peer = rank
                    break
                self._fds[descriptor] = rank
                poll.register(
                    descriptor, select.POLLIN | select.POLLHUP | select.POLLERR
                )
        except BaseException:
            self.close()
            raise
        if self.failed_peer is None:
            self._thread = threading.Thread(
                target=self._watch, args=(poll,), name="dwdp-peer-watch", daemon=True
            )
            self._thread.start()

    def _watch(self, poll) -> None:
        while not self._stop.is_set():
            events = poll.poll(100)
            if events and not self._stop.is_set():
                self.failed_peer = self._fds[events[0][0]]
                return

    def check(self) -> None:
        if self.failed_peer is not None:
            raise RuntimeError(
                f"DWDP weight owner {self.failed_peer} exited; "
                "restart the sharing group"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        for descriptor in self._fds:
            os.close(descriptor)
        self._fds.clear()
