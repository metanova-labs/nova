import json
import os

import aiohttp
import bittensor as bt


COMPOUND_PAYOUT_URL = "https://emission-transfer-api.metanova-labs.ai/payouts/compound-epoch-reward"


async def dispatch_bounty_payouts(
    payouts: list[tuple[str, int, float]],
    metagraph,
    config,
    epoch: int,
) -> None:
    """POST up to one compound payout request per epoch component."""
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

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for component, uid, proportion in payouts:
            if uid >= len(metagraph.coldkeys):
                bt.logging.error(f"Payout UID {uid} out of metagraph coldkey range for {component}; skipping.")
                continue

            coldkey = metagraph.coldkeys[uid]
            body = {
                "component": component,
                "destination_coldkey": coldkey,
                "epoch": epoch,
                "incentive_proportion": round(float(proportion), 8),
            }
            await _post_payout(session, headers, body, component, uid)


async def _post_payout(session, headers, body, component, uid):
    try:
        async with session.post(COMPOUND_PAYOUT_URL, json=body, headers=headers) as resp:
            status_code = resp.status
            text = await resp.text()
    except Exception as e:
        bt.logging.error(f"Payout request error component={component} uid={uid}: {e}")
        return

    if status_code != 200:
        bt.logging.error(
            f"Payout request failed component={component} uid={uid} http_status={status_code} body={text}"
        )
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        bt.logging.error(f"Payout response was not valid JSON component={component} uid={uid} body={text}")
        return

    status = data.get("status")
    if status == "success":
        bt.logging.info(
            f"Payout OK component={component} uid={uid} extrinsic={data.get('extrinsic_id')} "
            f"amount_alpha={data.get('amount_alpha')}"
        )
        return

    bt.logging.error(
        f"Payout domain failure component={component} uid={uid} status={status} detail={data.get('detail')}"
    )