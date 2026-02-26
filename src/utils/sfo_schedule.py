
def make_exp_growth_sfo_epochs(
    epochs: int,
    stages: int,
    exp_rate: float,
    power: float = 1.5,
):
    """
    Return: list[int] of epochs where stage increments happen (length == stages).
    Intervals grow as: interval_k = e0 * (exp_rate ** (power * k))
    Choose e0 so sum(interval_k) ~= epochs.

    Notes:
    - epochs: total number of epochs (e.g., 200)
    - stages: how many times to increase (e.g., 6)
    - exp_rate: same rate used for growth (e.g., 2.0)
    - power: 1.5 as requested
    """
    assert epochs > 0
    assert stages >= 1
    assert exp_rate > 0

    # Geometric-ish weights for intervals
    weights = [exp_rate ** (power * k) for k in range(stages)]
    denom = sum(weights)
    # real-valued base interval
    e0 = epochs / denom

    # build cumulative change points
    change_points = []
    t = 0.0
    for k in range(stages):
        t += e0 * weights[k]
        # convert to an epoch index in [0, epochs-1]
        ep = int(round(t))
        ep = min(max(ep, 0), epochs - 1)
        change_points.append(ep)

    # Ensure strictly increasing unique points (avoid duplicates due to rounding)
    uniq = []
    last = -1
    for ep in change_points:
        if ep <= last:
            ep = last + 1
        if ep <= epochs - 1:
            uniq.append(ep)
            last = ep

    # If rounding pushed some points beyond epochs-1, truncate
    uniq = [ep for ep in uniq if ep <= epochs - 1]

    # If we lost points, squeeze them back near the end (rare, but safe)
    while len(uniq) < stages:
        candidate = (uniq[-1] + 1) if uniq else 0
        if candidate > epochs - 1:
            # fallback: pack from the end
            candidate = epochs - (stages - len(uniq))
        if uniq and candidate <= uniq[-1]:
            candidate = uniq[-1] + 1
        if candidate > epochs - 1:
            break
        uniq.append(candidate)

    # Final guarantee: length == stages (if impossible, it will be shorter; but epochs>>stagesなら通常OK)
    return uniq[:stages]
