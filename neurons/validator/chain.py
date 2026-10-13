import asyncio
import random

import bittensor as bt


class ChainClient:
    """Owns the AsyncSubtensor handle; reconnects on failure."""

    def __init__(self, network):
        self.network = network
        self.subtensor = None

    async def connect(self):
        delay = 2
        attempt = 1
        while True:
            subtensor = None
            try:
                subtensor = bt.AsyncSubtensor(network=self.network)
                await asyncio.wait_for(subtensor.initialize(), timeout=30)
                self.subtensor = subtensor
                return
            except asyncio.CancelledError:
                await self._close(subtensor)
                raise
            except Exception as e:
                await self._close(subtensor)
                wait_seconds = delay + random.uniform(0, delay * 0.2)
                bt.logging.warning(
                    f"Subtensor connection attempt {attempt} failed: "
                    f"{type(e).__name__}: {e}. "
                    f"Retrying in {wait_seconds:.1f}s."
                )
                await asyncio.sleep(wait_seconds)
                delay = min(delay * 2, 60)
                attempt += 1

    async def reconnect(self):
        old, self.subtensor = self.subtensor, None
        await self._close(old)
        await self.connect()

    async def call(self, rpc_fn, timeout_s=10):
        """
        Retry once on a new connection; one that dropped with a subscription
        open cannot be revived in place. Cancellation is treated as a connection
        fault so the validator keeps running. `timeout_s=None` waits indefinitely.
        """
        try:
            return await self._invoke(rpc_fn, timeout_s)
        except (Exception, asyncio.CancelledError) as e:
            bt.logging.warning(
                f"Subtensor RPC reconnect triggered due to {type(e).__name__}: {e}"
            )
            await self.reconnect()
            return await self._invoke(rpc_fn, timeout_s)

    async def _invoke(self, rpc_fn, timeout_s):
        if self.subtensor is None:
            await self.connect()
        if timeout_s is None:
            return await rpc_fn(self.subtensor)
        return await asyncio.wait_for(rpc_fn(self.subtensor), timeout=timeout_s)

    @staticmethod
    async def _close(subtensor):
        if subtensor is None:
            return
        try:
            await subtensor.close()
        except Exception as e:
            bt.logging.warning(f"Failed to close old subtensor connection: {e}")
