import os
import secrets
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from neurons.validator import commitments, payouts


COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
HOTKEY = "5EnE7ryhVCUi3a6tTUvCq2qydFfv1xxxVi3GHWxciC4MLs2N"
OTHER_COLDKEY = "5GbfobwGAc7QQZzpUv9UCtbKktW7cfXww9sskRY3qNGutw5N"


class PayoutSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def snapshot(self):
        metagraph = SimpleNamespace(hotkeys=[HOTKEY], coldkeys=[COLDKEY])
        query = AsyncMock(return_value=SimpleNamespace(value={"block": 110}))
        subtensor = SimpleNamespace(substrate=SimpleNamespace(query=query),
                                    determine_block_hash=AsyncMock(return_value="epoch-end-hash"))
        config = SimpleNamespace(no_submission_blocks=0)
        decrypted = {0: {"molecules": [], "sequences": ["sequence"]}}
        with patch.object(commitments, "decode_metadata", return_value="submission"), \
                patch.object(commitments, "decrypt_submissions", return_value=(decrypted, {})):
            data, *_ = await commitments.gather_and_decrypt_commitments(
                subtensor, metagraph, 68, 100, 120, config, {}, None,
            )
        return data[0], metagraph, query

    async def test_submission_carries_epoch_end_owner_and_block(self):
        winner, _, query = await self.snapshot()
        self.assertEqual(winner["hotkey"], HOTKEY)
        self.assertEqual(winner["coldkey"], COLDKEY)
        self.assertEqual(winner["ownership_block_hash"], "epoch-end-hash")
        self.assertEqual(query.await_args.kwargs["block_hash"], "epoch-end-hash")

    async def test_later_ownership_and_uid_changes_do_not_redirect_payout(self):
        winner, metagraph, query = await self.snapshot()
        metagraph.hotkeys[0] = OTHER_COLDKEY
        metagraph.coldkeys[0] = OTHER_COLDKEY
        session = Mock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch.dict(os.environ, {"WALLET_TRANSFER_API_KEY": secrets.token_urlsafe(24)}, clear=True), \
                patch.object(payouts.aiohttp, "ClientSession", return_value=session), \
                patch.object(payouts, "_post_payout", new_callable=AsyncMock) as post:
            await payouts.dispatch_bounty_payouts(
                [("nanobody", winner["coldkey"], 0.4)],
                SimpleNamespace(emission_override_enabled=True), 25172,
            )
        self.assertEqual(post.await_count, 1)
        self.assertEqual(post.await_args.args[2], {
            "component": "nanobody", "destination_coldkey": COLDKEY,
            "epoch": 25172, "incentive_proportion": 0.4,
        })
        self.assertEqual(query.await_count, 1)  # Only commitment retrieval; no payout Owner RPC.

    async def test_missing_snapshot_owner_does_not_send(self):
        session = Mock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch.dict(os.environ, {"WALLET_TRANSFER_API_KEY": secrets.token_urlsafe(24)}, clear=True), \
                patch.object(payouts.aiohttp, "ClientSession", return_value=session), \
                patch.object(payouts, "_post_payout", new_callable=AsyncMock) as post:
            for coldkey in (None, "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"):
                await payouts.dispatch_bounty_payouts(
                    [("nanobody", coldkey, 0.4)], SimpleNamespace(emission_override_enabled=True), 25172,
                )
        post.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
