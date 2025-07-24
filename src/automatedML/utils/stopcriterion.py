import numpy as np


def get_stop_criterion_fun(method: str):
    if method == "probabilistic":
        return probabilistic_stop_criterion
    elif method == "threshold":
        return threshold_stop_criterion
    elif method == "average_threshold":
        return average_threshold_stop_criterion
    else:
        raise Exception(f"{method} not supported")


def stop_criterion(method: str, cost_history: list, args: tuple) -> bool:
    fun = get_stop_criterion_fun(method)
    return fun(cost_history, *args)


def __normal_pdf(x, mu, sigma):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    b = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return a * b


def probabilistic_stop_criterion(cost_history: list, beta: float) -> bool:
    # See https://onlinelibrary.wiley.com/doi/epdf/10.1111/itor.12010
    history = np.array([x for x in cost_history if np.isfinite(x)])

    if history.size < 2:
        return False

    mean_value = history.mean()
    std_value = history.std()
    min_value = history.min()

    x = np.arange(0, min_value, min_value/100)
    y = __normal_pdf(x, mean_value, std_value)
    bad_prob_value = 1-np.trapz(y, x)
    return bad_prob_value > beta


def threshold_stop_criterion(cost_history: list, th: float) -> bool:
    history = np.array([x for x in cost_history if np.isfinite(x)])
    return history.min() <= th


def average_threshold_stop_criterion(cost_history: list, win_size: int) -> bool:
    history = np.array([x for x in cost_history if np.isfinite(x)])

    if history.size > win_size:
        idx = len(history)-win_size
        history = history[idx:]

    worse_value = max(history)
    best_value = min(history)
    mean_value = history.mean()

    th = (worse_value+best_value)/2
    return mean_value > th

__all__=["stop_criterion"]