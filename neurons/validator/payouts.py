import os
import asyncio
import aiohttp
import bittensor as bt


async def dispatch_bounty_payouts(
    payouts: dict[int, float],
    metagraph,
    config,
    epoch: int,
) -> None:
    """
    POST one /payouts/compound-epoch-reward request per winner.

    payouts: {uid -> proportion of total non-burn incentive in [0, 1]}
    """
    if not payouts:
        bt.logging.info("No payouts to dispatch this epoch.")
        return

    if not config.emission_override_enabled:
        bt.logging.info("Emission override disabled in config.")
        return

    base_url = config.emission_api_base_url
    proportion_field = config.emission_proportion_field
    api_key = os.environ.get("WALLET_TRANSFER_API_KEY")

    if not base_url or not api_key:
        bt.logging.error("Bounty payouts misconfigured: missing api_base_url or WALLET_TRANSFER_API_KEY.")
        return

    url = base_url.rstrip("/") + "/payouts/compound-epoch-reward"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for uid, proportion in payouts.items():
            if uid >= len(metagraph.coldkeys):
                bt.logging.error(f"UID {uid} out of metagraph coldkey range; skipping.")
                continue
            coldkey = metagraph.coldkeys[uid]
            body = {
                "destination_coldkey": coldkey,
                "epoch": epoch,
                proportion_field: round(float(proportion), 8),
            }
            await _post_with_retry(session, url, headers, body, uid)


async def _post_with_retry(session, url, headers, body, uid, max_attempts: int = 3):
    backoff = 5
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                text = await resp.text()
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    status = data.get("status")
                    if status == "success":
                        bt.logging.info(
                            f"Payout OK uid={uid} extrinsic={data.get('extrinsic_id')} "
                            f"amount_alpha={data.get('amount_alpha')}"
                        )
                        return
                    # HTTP 200 but domain failure don't retry
                    bt.logging.error(
                        f"Payout domain failure uid={uid} status={status} detail={data.get('detail')}"
                    )
                    return
                if resp.status in (401, 422):
                    bt.logging.error(f"Non-retryable HTTP {resp.status} for uid={uid}: {text}")
                    return
                bt.logging.warning(f"HTTP {resp.status} on uid={uid} attempt {attempt}: {text}")
        except Exception as e:
            bt.logging.warning(f"Payout request error uid={uid} attempt {attempt}: {e}")

        if attempt < max_attempts:
            await asyncio.sleep(backoff)
            backoff *= 2

    bt.logging.error(f"Payout failed for uid={uid} after {max_attempts} attempts.")