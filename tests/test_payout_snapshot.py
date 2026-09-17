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
    async def dispatch(self, owner, submission_block=110, hotkey=HOTKEY):
        session = Mock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch.dict(os.environ, {"WALLET_TRANSFER_API_KEY": secrets.token_urlsafe(24)}, clear=True), \
                patch.object(payouts.aiohttp, "ClientSession", return_value=session), \
                patch.object(payouts, "_post_payout", new_callable=AsyncMock) as post:
            await payouts.dispatch_bounty_payouts(
                [("nanobody", hotkey, submission_block, 0.4)],
                SimpleNamespace(get_hotkey_owner=owner),
                SimpleNamespace(emission_override_enabled=True), 25172,
            )
        return post

    async def test_submission_block_owner_wins_over_epoch_end_and_current_owner(self):
        metagraph = SimpleNamespace(hotkeys=[HOTKEY], coldkeys=[OTHER_COLDKEY])
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
        winner = data[0]
        self.assertEqual(winner["hotkey"], HOTKEY)
        metagraph.hotkeys[0] = OTHER_COLDKEY  # The UID has since been recycled.
        owner = AsyncMock(side_effect=lambda hotkey, block=None:
                          COLDKEY if hotkey == HOTKEY and block == 110 else OTHER_COLDKEY)
        post = await self.dispatch(owner, winner["block_submitted"], winner["hotkey"])
        owner.assert_awaited_once_with(HOTKEY, block=110)
        self.assertEqual(post.await_args.args[2]["destination_coldkey"], COLDKEY)

    async def test_transient_error_retries_the_same_submission_block(self):
        owner = AsyncMock(side_effect=[RuntimeError("RPC unavailable"), COLDKEY])
        with patch.object(payouts.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            post = await self.dispatch(owner)
        sleep.assert_awaited_once_with(1)
        self.assertEqual(owner.await_count, 2)
        self.assertTrue(all(a.kwargs == {"block": 110} for a in owner.await_args_list))
        self.assertEqual(post.await_count, 1)
        self.assertEqual(post.await_args.args[2]["destination_coldkey"], COLDKEY)

    async def test_exhausted_retries_never_pay_a_current_owner(self):
        owner = AsyncMock(side_effect=RuntimeError("RPC unavailable"))
        with patch.object(payouts.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            post = await self.dispatch(owner)
        self.assertEqual([a.args[0] for a in sleep.await_args_list], [1, 2])
        self.assertEqual(owner.await_count, 3)
        self.assertTrue(all(a.kwargs == {"block": 110} for a in owner.await_args_list))
        post.assert_not_awaited()

    async def test_missing_block_or_null_owner_does_not_send(self):
        for block, coldkey in [(None, COLDKEY), (110, None),
                               (110, "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM")]:
            owner = AsyncMock(return_value=coldkey)
            post = await self.dispatch(owner, block)
            post.assert_not_awaited()
            if block is None:
                owner.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
