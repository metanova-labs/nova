import bittensor as bt
import asyncio


def is_uid_valid(uid, metagraph):
    if uid is not None:
        if not (0 <= uid < len(metagraph.uids)):
            bt.logging.error(f"Error: target_uid {uid} out of range [0, {len(metagraph.uids)-1}]. Exiting.")
            return False
    return True


async def set_weights(winner_molecules, winner_nanobodies, config):
    bt.logging.debug(
        f"Setting weights for epoch {config.epoch}"
    )

    if config.emission_override_enabled:
        bt.logging.debug(f"Emission override enabled")
    else:
        bt.logging.debug(f"Emission override disabled: setting weights for winner molecules: {winner_molecules} and nanobodies: {winner_nanobodies}")

    burn_rate = 0.722
    wallet_name = config.wallet.name
    wallet_hotkey = config.wallet.hotkey
    netuid = config.netuid

    wallet = bt.wallet(
        name=wallet_name,
        hotkey=wallet_hotkey,
    )

    with bt.subtensor(network=config.network) as subtensor:
        # Download the metagraph for netuid=68
        metagraph = subtensor.metagraph(netuid)

        n = len(metagraph.uids)
        weights = [0.0] * n
        payouts: list[tuple[str, int, float]] = []

        molecule_proportion = 1.0 - config.nanobody_weight
        nanobody_proportion = config.nanobody_weight
        molecule_incentive = (1.0 - burn_rate) * molecule_proportion
        nanobody_incentive = (1.0 - burn_rate) * nanobody_proportion

        # Check registration
        hotkey_ss58 = wallet.hotkey.ss58_address
        if hotkey_ss58 not in metagraph.hotkeys:
            bt.logging.error(f"Hotkey {hotkey_ss58} is not registered on netuid {netuid}. Exiting.")
            return []

        for target_uid in [winner_molecules, winner_nanobodies]:
            if not is_uid_valid(target_uid, metagraph):
                return []

        weights[0] = burn_rate

        if config.emission_override_enabled:
            override_uid = getattr(config, "emission_override_uid", None)
            if override_uid is None:
                bt.logging.error("emission_override_uid not configured; aborting set_weights.")
                return []
            if not is_uid_valid(override_uid, metagraph):
                bt.logging.error(
                    f"Error: emission_override_uid {override_uid} out of range [0, {len(metagraph.uids)-1}]. Exiting."
                )
                return []
            if override_uid == 0:
                bt.logging.error("emission_override_uid must not be the burn UID (0).")
                return []

            weights[override_uid] = 1.0 - burn_rate

            if winner_molecules is not None:
                payouts.append(("molecule", winner_molecules, molecule_proportion))
            if winner_nanobodies is not None:
                payouts.append(("nanobody", winner_nanobodies, nanobody_proportion))
        else:
            if winner_molecules is not None:
                weights[winner_molecules] += molecule_incentive
            else:
                weights[0] += molecule_incentive

            if winner_nanobodies is not None:
                weights[winner_nanobodies] += nanobody_incentive
            else:
                weights[0] += nanobody_incentive

        # Send the weights to the chain with retry logic
        max_retries = 10
        delay_between_retries = 12  # seconds
        for attempt in range(max_retries):
            try:
                bt.logging.info(f"Attempt {attempt + 1} to set weights.")
                result = subtensor.set_weights(
                    netuid=netuid,
                    wallet=wallet,
                    uids=metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=True,
                )
                bt.logging.info(f"Result from set_weights: {result}")

                if result[0] is True:
                    bt.logging.info("Weights set successfully. Exiting retry loop.")
                    bt.logging.info("Done.")
                    return payouts

                bt.logging.info("set_weights returned a non-success response. Will retry if attempts remain.")
                if attempt < max_retries - 1:
                    bt.logging.info(f"Retrying in {delay_between_retries} seconds...")
                    await asyncio.sleep(delay_between_retries)
            except Exception as e:
                bt.logging.error(f"Error setting weights: {e}")

                if attempt < max_retries - 1:
                    bt.logging.info(f"Retrying in {delay_between_retries} seconds...")
                    await asyncio.sleep(delay_between_retries)
                else:
                    bt.logging.error("Failed to set weights after multiple attempts. Exiting.")
                    return []

    bt.logging.error("Failed to set weights after non-success responses. Exiting.")
    return []
