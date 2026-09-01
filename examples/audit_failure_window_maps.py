import argparse
import csv
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np


GRID_CELL_SIZE = 5.0
INVALID_POSITION = -10000.0
ACTOR_TYPES = {1, 2, 3}
GRID_TYPES = {4, 5, 6}


class MapFormatError(RuntimeError):
    pass


class Reader:
    def __init__(self, path):
        self.path = path
        self.data = memoryview(path.read_bytes())
        self.offset = 0

    def take(self, dtype, count=1):
        dtype = np.dtype(dtype)
        size = dtype.itemsize * count
        if count < 0 or self.offset + size > len(self.data):
            raise MapFormatError(
                f"short read at byte {self.offset}: need {size}, "
                f"file has {len(self.data) - self.offset}"
            )
        out = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.offset)
        self.offset += size
        return out

    def i32(self):
        return int(self.take("<i4")[0])

    def f32(self, count=1):
        return self.take("<f4", count)

    def skip_f32(self, count):
        self.take("<f4", count)

    def bytes(self, count):
        if self.offset + count > len(self.data):
            raise MapFormatError(f"short read at byte {self.offset}: need {count}")
        out = bytes(self.data[self.offset : self.offset + count])
        self.offset += count
        return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("results/failure_window/failure_window_unique_maps.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/failure_window")
    )
    return parser.parse_args()


def add_issue(issues, map_id, category, entity_idx=-1, timestep=-1, detail=""):
    issues.append(
        {
            "map_id": map_id,
            "category": category,
            "entity_idx": entity_idx,
            "timestep": timestep,
            "detail": detail,
        }
    )


def nonfinite_indices(*arrays):
    mask = np.zeros(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask |= ~np.isfinite(array)
    return np.flatnonzero(mask)


def audit_map(path, map_id, issues, stats):
    reader = Reader(path)
    scenario_id = reader.bytes(16).split(b"\0", 1)[0].decode("utf-8", "replace")
    sdc_track_index = reader.i32()
    num_tracks = reader.i32()
    if not 0 <= num_tracks <= 1_000_000:
        raise MapFormatError(f"invalid num_tracks_to_predict={num_tracks}")
    tracks = reader.take("<i4", num_tracks).astype(np.int64, copy=False)
    num_objects = reader.i32()
    num_roads = reader.i32()
    if not 0 <= num_objects <= 1_000_000 or not 0 <= num_roads <= 1_000_000:
        raise MapFormatError(f"invalid counts: objects={num_objects}, roads={num_roads}")

    if sdc_track_index < -1 or sdc_track_index >= num_objects:
        add_issue(issues, map_id, "sdc_index_out_of_range", detail=str(sdc_track_index))
    for value in tracks[(tracks < 0) | (tracks >= num_objects)][:20]:
        add_issue(issues, map_id, "track_index_out_of_range", detail=str(int(value)))

    mean_entities = []
    grid_entities = []
    for entity_idx in range(num_objects + num_roads):
        stored_map_id = reader.i32()
        entity_type = reader.i32()
        reader.i32()  # entity id
        size = reader.i32()
        if not 0 <= size <= 1_000_000:
            raise MapFormatError(f"entity {entity_idx}: invalid array_size={size}")
        if stored_map_id != map_id:
            add_issue(
                issues,
                map_id,
                "stored_map_id_mismatch",
                entity_idx,
                detail=str(stored_map_id),
            )

        x = reader.f32(size)
        y = reader.f32(size)
        z = reader.f32(size)
        all_fields = [x, y, z]

        if entity_type in ACTOR_TYPES:
            vx = reader.f32(size)
            vy = reader.f32(size)
            vz = reader.f32(size)
            heading = reader.f32(size)
            valid = reader.take("<i4", size)
            expert = [reader.f32(size) for _ in range(5)]
            all_fields += [vx, vy, vz, heading, *expert]

            bad_valid = np.flatnonzero((valid != 0) & (valid != 1))
            for timestep in bad_valid[:20]:
                add_issue(
                    issues,
                    map_id,
                    "invalid_valid_flag",
                    entity_idx,
                    int(timestep),
                    str(int(valid[timestep])),
                )

            active = valid == 1
            if active.any():
                bad = np.flatnonzero(
                    active
                    & (
                        ~np.isfinite(x)
                        | ~np.isfinite(y)
                        | ~np.isfinite(z)
                        | ~np.isfinite(vx)
                        | ~np.isfinite(vy)
                        | ~np.isfinite(vz)
                        | ~np.isfinite(heading)
                    )
                )
                for timestep in bad[:20]:
                    add_issue(
                        issues,
                        map_id,
                        "nonfinite_valid_actor_state",
                        entity_idx,
                        int(timestep),
                    )
            mean_entities.append((entity_type, x, y, valid))
        elif entity_type in GRID_TYPES:
            grid_entities.append((entity_idx, x, y))
            mean_entities.append((entity_type, x, y, None))
        elif entity_type >= 4:
            mean_entities.append((entity_type, x, y, None))

        for field_idx, array in enumerate(all_fields):
            bad = np.flatnonzero(~np.isfinite(array))
            if bad.size:
                add_issue(
                    issues,
                    map_id,
                    "nonfinite_binary_array",
                    entity_idx,
                    int(bad[0]),
                    f"field={field_idx}, count={bad.size}",
                )

        scalars = reader.f32(6)  # width, length, height, goal x/y/z
        reader.i32()  # mark_as_expert
        if not np.isfinite(scalars).all():
            add_issue(issues, map_id, "nonfinite_scalar", entity_idx)

    # Reproduce set_means() and the translated grid coordinates with float32
    # arithmetic before checking init_grid_map().
    mean_x = np.float32(0.0)
    mean_y = np.float32(0.0)
    point_count = 0
    for entity_type, x, y, valid in mean_entities:
        use = valid != 0 if entity_type in ACTOR_TYPES else np.ones(len(x), dtype=bool)
        for value_x, value_y in zip(x[use], y[use]):
            point_count += 1
            divisor = np.float32(point_count)
            mean_x = np.float32(mean_x + np.float32(np.float32(value_x - mean_x) / divisor))
            mean_y = np.float32(mean_y + np.float32(np.float32(value_y - mean_y) / divisor))

    shifted_grid_entities = []
    for entity_idx, x, y in grid_entities:
        sx = x.copy()
        sy = y.copy()
        shift = sx != np.float32(INVALID_POSITION)
        sx[shift] = np.float32(sx[shift] - mean_x)
        sy[shift] = np.float32(sy[shift] - mean_y)
        shifted_grid_entities.append((entity_idx, sx, sy))

    points = [
        (float(x), float(y))
        for _, xs, ys in shifted_grid_entities
        for x, y in zip(xs, ys)
        if x != INVALID_POSITION and y != INVALID_POSITION
    ]
    if not points:
        add_issue(issues, map_id, "no_valid_grid_points", detail=scenario_id)
        stats["maps_without_grid_points"] += 1
        return reader.offset

    px = np.asarray([p[0] for p in points], dtype=np.float64)
    py = np.asarray([p[1] for p in points], dtype=np.float64)
    if not np.isfinite(px).all() or not np.isfinite(py).all():
        add_issue(issues, map_id, "nonfinite_grid_extent", detail=scenario_id)
        stats["maps_with_nonfinite_grid_extent"] += 1
        return reader.offset

    left = np.float32(px.min())
    right = np.float32(px.max())
    bottom = np.float32(py.min())
    top = np.float32(py.max())
    cell_size = np.float32(GRID_CELL_SIZE)
    cols = math.ceil(float(np.float32(np.float32(right - left) / cell_size)))
    rows = math.ceil(float(np.float32(np.float32(top - bottom) / cell_size)))
    if cols <= 0 or rows <= 0:
        add_issue(
            issues,
            map_id,
            "nonpositive_grid_shape",
            detail=f"cols={cols}, rows={rows}",
        )
        stats["maps_with_nonpositive_grid_shape"] += 1
        return reader.offset

    bad_segments = 0
    for entity_idx, x, y in shifted_grid_entities:
        for segment in range(max(0, len(x) - 1)):
            mid_x = float((np.float32(x[segment]) + np.float32(x[segment + 1])) / np.float32(2.0))
            mid_y = float((np.float32(y[segment]) + np.float32(y[segment + 1])) / np.float32(2.0))
            if not math.isfinite(mid_x) or not math.isfinite(mid_y):
                bad = True
            else:
                grid_x = int(np.float32(np.float32(mid_x) - left) / cell_size)
                grid_y = int(np.float32(np.float32(mid_y) - bottom) / cell_size)
                bad = grid_x < 0 or grid_x >= cols or grid_y < 0 or grid_y >= rows
            if bad:
                bad_segments += 1
                if bad_segments <= 20:
                    add_issue(
                        issues,
                        map_id,
                        "grid_index_minus_one",
                        entity_idx,
                        segment,
                        f"mid=({mid_x},{mid_y}), grid=({cols},{rows})",
                    )
    if bad_segments:
        stats["maps_with_grid_index_minus_one"] += 1
        stats["grid_index_minus_one_segments"] += bad_segments
    return reader.offset


def main():
    args = parse_args()
    issues = []
    stats = Counter()

    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    for row in candidates:
        map_id = int(row["map_id"])
        path = Path(row["map_path"])
        try:
            base_bytes = audit_map(path, map_id, issues, stats)
            stats["maps_parsed"] += 1
            stats["base_bytes"] += base_bytes
        except (OSError, MapFormatError, ValueError, struct.error) as error:
            add_issue(issues, map_id, "parse_error", detail=str(error))
            stats["maps_failed_to_parse"] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    issue_path = args.output_dir / "failure_window_static_issues.csv"
    fields = ["map_id", "category", "entity_idx", "timestep", "detail"]
    with issue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)

    category_counts = Counter(issue["category"] for issue in issues)
    summary = {
        "candidate_maps": len(candidates),
        "issue_rows": len(issues),
        "maps_with_any_issue": len({issue["map_id"] for issue in issues}),
        "issue_categories": dict(sorted(category_counts.items())),
        **dict(sorted(stats.items())),
    }
    summary_path = args.output_dir / "failure_window_static_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"issues = {issue_path}")
    print(f"summary = {summary_path}")
    print("FAILURE_WINDOW_STATIC_AUDIT_PASSED")


if __name__ == "__main__":
    main()
