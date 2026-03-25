import numpy as np

def random_number(low, up):
    """
    Draw a random number from a uniform distribution between low and up
    """
    return np.random.rand() * (up - low) + low

def random_number_exp(low, up):
    """
    Draw a random number from a log-uniform distribution between low and up
    """
    return 10 ** random_number(low, up)

def generate_params() -> dict:
    """
    Generate a random set of targets and weights for a single-stage run.
    These have to be fine tuned for each initial condition

    """
    # TARGETS
    current_threshold = random_number(125000, 200000)
    iota_target = random_number(0.08, 0.14)
    
    # WEIGHTS
    # Iota penalty weight: ~50–500
    iota_weight = random_number_exp(1.7, 2.7)

    # QS (nonQS ratio) weight: ~1–300
    qs_weight = random_number_exp(0.0, 2.5)

    # Current penalty weight: ~0.01–1 (gradient is now active via vjp fix;
    # per-coil gradient in DOF space is O(0.1), so weight >1 dominates)
    current_weight = random_number_exp(-2.0, 0.0)

    return {
        "CURRENT_THRESHOLD": float(current_threshold),
        "IOTA_TARGET": float(iota_target),
        "IOTA_WEIGHT": float(iota_weight),
        "QS_WEIGHT": float(qs_weight),
        "CURRENT_WEIGHT": float(current_weight),
    }

if __name__ == "__main__":
    params = generate_params()

    cli_args = (
        f"--iota-target {params['IOTA_TARGET']} "
        f"--iota-weight {params['IOTA_WEIGHT']} "
        f"--qs-weight {params['QS_WEIGHT']} "
        f"--current-threshold {params['CURRENT_THRESHOLD']} "
        f"--current-weight {params['CURRENT_WEIGHT']}"
    )

    # This print is what batch_scan.sh will capture and pass through
    print(cli_args)