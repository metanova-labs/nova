import sys
import argparse
import bittensor as bt

def main():
    # 1) Parse the target_uids argument
    parser = argparse.ArgumentParser(
        description="Set weights on netuid=68 so that specified UIDs share weight equally."
    )
    parser.add_argument('--target_uids', type=str, required=True,
                        help="The UIDs that will receive weight. Can be a single UID or comma-separated list for ties.")
    parser.add_argument('--wallet_name', type=str, required=True,
                        help="The name of the wallet to use.")
    parser.add_argument('--wallet_hotkey', type=str, required=True,
                        help="The hotkey to use for the wallet.")

    args = parser.parse_args()

    NETUID = 68
    
    wallet = bt.wallet(
        name=args.wallet_name,  
        hotkey=args.wallet_hotkey, 
    )

    # Create Subtensor connection
    subtensor = bt.subtensor()

    # Download the metagraph for netuid=68
    metagraph = subtensor.metagraph(NETUID)

    # Check registration
    hotkey_ss58 = wallet.hotkey.ss58_address
    if hotkey_ss58 not in metagraph.hotkeys:
        print(f"Hotkey {hotkey_ss58} is not registered on netuid {NETUID}. Exiting.")
        sys.exit(1)

    # 2) Build the weight vector
    n = len(metagraph.uids)
    weights = [0.0] * n

    # Parse the target_uids parameter
    if ',' in args.target_uids:
        # Handle multiple UIDs in case of a tie
        target_uids = [int(uid.strip()) for uid in args.target_uids.split(',')]
    else:
        # Handle single UID
        target_uids = [int(args.target_uids)]
    
    # Validate all provided target UIDs
    for uid in target_uids:
        if not (0 <= uid < n):
            print(f"Error: target_uid {uid} out of range [0, {n-1}]. Exiting.")
            sys.exit(1)
    
    # Calculate weight to assign to each winning UID
    weight_per_uid = 1.0 / len(target_uids)
    
    # Set weights for winners
    for uid in target_uids:
        weights[uid] = weight_per_uid

    # 3) Send the weights to the chain
    print(f"Setting weights for {len(target_uids)} UIDs (netuid={NETUID}): {target_uids}")
    print(f"Each winning UID receives weight={weight_per_uid:.4f}")
    print(f"Weights: {weights}")
    result = subtensor.set_weights(
        netuid=NETUID,
        wallet=wallet,
        uids=metagraph.uids,
        weights=weights,
        wait_for_inclusion=True
    )
    print(f"Result from set_weights: {result}")
    print("Done.")

if __name__ == "__main__":
    main() 
