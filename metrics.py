def mean_rank(output):
    return output.mean() * 100


def mean_rank_reciprocal(output):
    return (1/output).mean() * 100


def hits_at_x(output, x):
    return output.le(x).float().mean() * 100
