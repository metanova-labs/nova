import asyncio
import json
import os

import aiohttp
import bittensor as bt


COMPOUND_PAYOUT_URL = "https://emission-transfer-api.metanova-labs.ai/payouts/compound-epoch-reward"
COMPOUND_PAYOUT_HTTP_TIMEOUT_S = 90
EXPECTED_PAYOUT_STATUSES = {"already_processing", "idempotency_conflict"}


async def dispatch_bounty_payouts(
    payouts: list[tuple[str, str, int, float]],
    subtensor,
    config,
    epoch: int,
) -> None:
    """Pay the winning hotkey's owner at its recorded commitment block."""
    if not payouts:
        bt.logging.info("No payouts to dispatch this epoch.")
        return

    if not config.emission_override_enabled:
        bt.logging.info("Emission override disabled in config.")
        return

    api_key = os.environ.get("WALLET_TRANSFER_API_KEY")

    if not api_key:
        bt.logging.info("Skipping payout transfer: WALLET_TRANSFER_API_KEY not set.")
   
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    timeout = aiohttp.ClientTimeout(total=COMPOUND_PAYOUT_HTTP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for component, hotkey, submission_block, proportion in payouts:
            coldkey = await _resolve_hotkey_owner(subtensor, hotkey, submission_block)
            if not coldkey or coldkey == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM":
                bt.logging.error(f"Missing submission-time coldkey for payout epoch={epoch} component={component}.")
                continue

            body = {
                "component": component,
                "destination_coldkey": coldkey,
                "epoch": epoch,
                "incentive_proportion": round(float(proportion), 8),
            }
            await _post_payout(session, headers, body, epoch, component, coldkey)


async def _post_payout(session, headers, body, epoch, component, coldkey):
    try:
        async with session.post(COMPOUND_PAYOUT_URL, json=body, headers=headers) as resp:
            status_code = resp.status
            text = await resp.text()
    except Exception as e:
        bt.logging.error(
            f"Payout request error epoch={epoch} component={component} coldkey={coldkey} error={e}"
        )
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        bt.logging.error(
            f"Payout response was not valid JSON epoch={epoch} component={component} "
            f"coldkey={coldkey} http_status={status_code}"
        )
        return

    status = data.get("status")
    detail = data.get("detail")

    if status_code != 200:
        bt.logging.error(
            f"Payout request failed epoch={epoch} component={component} coldkey={coldkey} "
            f"http_status={status_code} status={status} detail={detail}"
        )
        return

    if status == "success":
        bt.logging.info(
            f"Payout sent epoch={epoch} component={component} coldkey={coldkey} "
            f"amount_alpha={data.get('amount_alpha')} "
            f"extrinsic={data.get('extrinsic_id')}"
        )
        return

    log = bt.logging.warning if status in EXPECTED_PAYOUT_STATUSES else bt.logging.error
    prefix = "Payout skipped" if status in EXPECTED_PAYOUT_STATUSES else "Payout failed"
    log(f"{prefix} epoch={epoch} component={component} coldkey={coldkey} status={status} detail={detail}")


async def _resolve_hotkey_owner(subtensor, hotkey: str, submission_block: int) -> str | None:
    if submission_block is None:
        return None  # Never fall back to the current owner.
    for attempt in range(3):
        try:
            return await asyncio.wait_for(
                subtensor.get_hotkey_owner(hotkey, block=submission_block), timeout=10,
            )
        except Exception as e:
            bt.logging.warning(
                f"Owner lookup failed hotkey={hotkey} block={submission_block} "
                f"attempt={attempt + 1} error_type={type(e).__name__}"
            )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return None
