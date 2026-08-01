from __future__ import annotations

import numpy as np
import torch

from tensordict.tensordict import TensorDict


def _extract_route(sequence: np.ndarray) -> list[int]:
    route: list[int] = []
    seen: set[int] = set()
    for node in sequence.tolist():
        node = int(node)
        if node == 0:
            break
        if node > 0 and node not in seen:
            route.append(node)
            seen.add(node)
    return route


def _encode_route(route: list[int], seq_len: int) -> np.ndarray:
    encoded = np.zeros(seq_len, dtype=np.int64)
    if route:
        capped = route[: max(seq_len - 1, 0)] if seq_len > 0 else []
        encoded[: len(capped)] = np.asarray(capped, dtype=np.int64)
    return encoded


def _route_length(route: list[int], distance: np.ndarray) -> float:
    if not route:
        return 0.0
    total = float(distance[0, route[0]])
    for left, right in zip(route[:-1], route[1:]):
        total += float(distance[left, right])
    total += float(distance[route[-1], 0])
    return total


def _removal_saving(route: list[int], idx: int, distance: np.ndarray) -> float:
    prev_node = 0 if idx == 0 else route[idx - 1]
    node = route[idx]
    next_node = 0 if idx == len(route) - 1 else route[idx + 1]
    return float(distance[prev_node, node] + distance[node, next_node] - distance[prev_node, next_node])


def _repair_route(route: list[int], distance: np.ndarray, prize: np.ndarray, limit: float) -> list[int]:
    cleaned: list[int] = []
    seen: set[int] = set()
    for node in route:
        node = int(node)
        if node <= 0 or node in seen:
            continue
        cleaned.append(node)
        seen.add(node)

    while cleaned and _route_length(cleaned, distance) > limit + 1e-6:
        removable_indices = range(1, len(cleaned)) if len(cleaned) > 1 else range(len(cleaned))
        best_idx = None
        best_score = None
        for idx in removable_indices:
            node = cleaned[idx]
            saving = max(_removal_saving(cleaned, idx, distance), 1e-6)
            score = float(prize[node]) / saving
            key = (score, float(prize[node]), -saving, idx)
            if best_score is None or key < best_score:
                best_score = key
                best_idx = idx
        if best_idx is None:
            break
        cleaned.pop(best_idx)
    return cleaned


def _try_best_insertions(
    route: list[int],
    distance: np.ndarray,
    prize: np.ndarray,
    limit: float,
    num_nodes: int,
    max_route_len: int,
) -> list[int]:
    route = list(route)
    available = [node for node in range(1, num_nodes) if node not in set(route)]
    current_length = _route_length(route, distance)

    while available and len(route) < max_route_len:
        start_pos = 1 if route else 0
        best = None
        for node in available:
            for pos in range(start_pos, len(route) + 1):
                prev_node = 0 if pos == 0 else route[pos - 1]
                next_node = 0 if pos == len(route) else route[pos]
                delta = float(distance[prev_node, node] + distance[node, next_node] - distance[prev_node, next_node])
                new_length = current_length + delta
                if new_length > limit + 1e-6:
                    continue
                gain = float(prize[node])
                efficiency = gain / max(delta, 1e-6)
                key = (efficiency, gain, -delta, -pos)
                if best is None or key > best[0]:
                    best = (key, node, pos, new_length)
        if best is None:
            break
        _, node, pos, current_length = best
        route.insert(pos, node)
        available.remove(node)
    return route


def _mutate_route(
    route: list[int],
    num_nodes: int,
    max_route_len: int,
    rng: np.random.Generator,
) -> list[int]:
    mutated = list(route)
    if not mutated:
        if num_nodes <= 1 or max_route_len <= 0:
            return mutated
        return [int(rng.integers(1, num_nodes))]

    candidate_ops = ["reverse"]
    if len(mutated) > 1:
        candidate_ops.append("remove")
    if num_nodes - 1 > len(set(mutated)) and len(mutated) < max_route_len:
        candidate_ops.extend(["insert", "swap"])

    op = candidate_ops[int(rng.integers(0, len(candidate_ops)))]

    if op == "reverse" and len(mutated) >= 2:
        low = 1 if len(mutated) > 1 else 0
        if low >= len(mutated):
            return mutated
        left = int(rng.integers(low, len(mutated)))
        right = int(rng.integers(left + 1, len(mutated) + 1))
        mutated[left:right] = reversed(mutated[left:right])
        return mutated

    if op == "remove":
        idx = int(rng.integers(1, len(mutated))) if len(mutated) > 1 else 0
        mutated.pop(idx)
        return mutated

    available = [node for node in range(1, num_nodes) if node not in set(mutated)]
    if op == "insert" and available and len(mutated) < max_route_len:
        node = available[int(rng.integers(0, len(available)))]
        pos_low = 1 if mutated else 0
        pos = int(rng.integers(pos_low, len(mutated) + 1))
        mutated.insert(pos, node)
        return mutated

    if op == "swap" and available and len(mutated) > 1:
        idx = int(rng.integers(1, len(mutated)))
        node = available[int(rng.integers(0, len(available)))]
        mutated[idx] = node
        return mutated

    return mutated


def _score_route(route: list[int], prize: np.ndarray, distance: np.ndarray) -> tuple[float, float]:
    reward = float(prize[np.asarray(route, dtype=np.int64)].sum()) if route else 0.0
    length = _route_length(route, distance)
    return reward, length


def _improve_single(
    route: list[int],
    distance: np.ndarray,
    prize: np.ndarray,
    limit: float,
    num_nodes: int,
    max_route_len: int,
    max_iterations: int,
    num_candidates: int,
    rng: np.random.Generator,
) -> list[int]:
    current = _repair_route(route, distance, prize, limit)[:max_route_len]
    current = _try_best_insertions(current, distance, prize, limit, num_nodes, max_route_len)
    current_reward, current_length = _score_route(current, prize, distance)

    for _ in range(max_iterations):
        best_candidate = None
        best_reward = current_reward
        best_length = current_length

        for _ in range(num_candidates):
            mutated = _mutate_route(current, num_nodes, max_route_len, rng)
            repaired = _repair_route(mutated, distance, prize, limit)
            candidate = _try_best_insertions(repaired[:max_route_len], distance, prize, limit, num_nodes, max_route_len)
            reward, length = _score_route(candidate, prize, distance)
            if reward > best_reward + 1e-6 or (abs(reward - best_reward) <= 1e-6 and length + 1e-6 < best_length):
                best_candidate = candidate
                best_reward = reward
                best_length = length

        if best_candidate is None:
            continue
        current = best_candidate
        current_reward = best_reward
        current_length = best_length

    return current


def local_search(
    td: TensorDict,
    actions: torch.Tensor,
    max_iterations: int = 1,
    num_threads: int = None,
    num_candidates: int = 4,
) -> torch.Tensor:
    del num_threads

    if actions is None or actions.dim() != 2 or actions.size(0) == 0:
        return actions

    td_cpu = td.detach().cpu() if hasattr(td, "detach") else td.cpu()
    actions_cpu = actions.detach().cpu()

    distances = torch.cdist(td_cpu["locs"], td_cpu["locs"]).numpy().astype(np.float32)
    prizes = td_cpu["prize"].numpy().astype(np.float32)
    limits = td_cpu["max_length"][..., 0].numpy().astype(np.float32) + 1e-6
    actions_np = actions_cpu.numpy().astype(np.int64)

    rng = np.random.default_rng()
    seq_len = actions_np.shape[1]
    num_nodes = distances.shape[1]
    max_route_len = max(seq_len - 1, 0)
    max_iterations = max(1, int(max_iterations))
    num_candidates = max(1, int(num_candidates))

    improved = np.zeros_like(actions_np)
    for batch_idx in range(actions_np.shape[0]):
        route = _extract_route(actions_np[batch_idx])
        improved_route = _improve_single(
            route=route,
            distance=distances[batch_idx],
            prize=prizes[batch_idx],
            limit=float(limits[batch_idx]),
            num_nodes=num_nodes,
            max_route_len=max_route_len,
            max_iterations=max_iterations,
            num_candidates=num_candidates,
            rng=rng,
        )
        improved[batch_idx] = _encode_route(improved_route, seq_len)

    return torch.from_numpy(improved).to(device=actions.device, dtype=actions.dtype)
