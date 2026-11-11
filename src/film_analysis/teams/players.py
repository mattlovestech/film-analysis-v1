"""Find players in a film frame and read the uniform colors on each one.

OpenCV's people detector supplies the boxes. NumPy holds the pixels inside
each box so the jersey and pants can be averaged without the grass.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class PlayerBox:
    """One detected player, in pixels of the source frame."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PlayerColors:
    """Jersey, pants, and helmet colors measured on one group of players."""

    jersey: str
    pants: str
    helmet: str


def team_colors_from_players(
    frames: Sequence[bytes],
) -> tuple[PlayerColors, PlayerColors] | None:
    """The white-jersey team and the colored-jersey team, from detected players."""

    light: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
    colored: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
    for frame in frames:
        image = _bgr(frame)
        if image is None:
            continue
        for box in find_players(frame):
            crop = _player_crop(image, box)
            if crop is not None and _striped(crop):
                continue
            sample = _uniform_colors(image, box)
            if sample is None:
                continue
            jersey, _pants, _helmet = sample
            if _light_cloth(jersey):
                light.append(sample)
            else:
                colored.append(sample)
    if len(light) < 1 or len(colored) < 1:
        return None
    return _group(light), _group(colored)


def find_players(jpeg: bytes) -> list[PlayerBox]:
    """People in this frame. An unreadable frame has none."""

    image = _bgr(jpeg)
    if image is None:
        return []
    return _separate(_hog_people(image))


def _bgr(jpeg: bytes):
    if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8\xff"):
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    array = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image


def _hog_people(image) -> list[PlayerBox]:
    """OpenCV's default people detector. Coordinates are on the original frame."""

    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 960 / longest if longest > 960 else 1.0
    view = image
    if scale != 1.0:
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, weights = hog.detectMultiScale(
        view,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    boxes: list[PlayerBox] = []
    for index, (x, y, width, height) in enumerate(found):
        weight = _weight(weights, index)
        if weight < 0.2 or width < 12 or height < 24:
            continue
        boxes.append(
            PlayerBox(
                int(x / scale),
                int(y / scale),
                int(width / scale),
                int(height / scale),
            )
        )
    return boxes


def _weight(weights, index: int) -> float:
    if index >= len(weights):
        return 0.0
    value = weights[index]
    flat = getattr(value, "flat", None)
    if flat is not None:
        return float(next(iter(flat)))
    return float(value)


def _uniform_colors(
    image,
    box: PlayerBox,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
    """Helmet, jersey, and pants from the top, middle, and lower part of a player."""

    height, width = image.shape[:2]
    x = max(0, box.x)
    y = max(0, box.y)
    box_width = min(box.width, width - x)
    box_height = min(box.height, height - y)
    if box_width < 8 or box_height < 16:
        return None
    helmet = _cloth(image, x, y, box_width, int(box_height * 0.22))
    jersey = _cloth(image, x, y + int(box_height * 0.22), box_width, int(box_height * 0.30))
    pants = _cloth(image, x, y + int(box_height * 0.55), box_width, int(box_height * 0.28))
    if helmet is None or jersey is None or pants is None:
        return None
    return jersey, pants, helmet


def _cloth(image, x: int, y: int, width: int, height: int) -> tuple[int, int, int] | None:
    if width < 4 or height < 4:
        return None
    inset_x = x + width // 4
    inset_width = max(1, width // 2)
    crop = image[y : y + height, inset_x : inset_x + inset_width]
    if crop.size == 0:
        return None
    kept: list[tuple[int, int, int]] = []
    for blue, green, red in crop.reshape(-1, 3)[::2]:
        red, green, blue = int(red), int(green), int(blue)
        if _skip(red, green, blue):
            continue
        kept.append((red, green, blue))
    if len(kept) < 6:
        return None
    count = len(kept)
    return (
        round(sum(red for red, _green, _blue in kept) / count),
        round(sum(green for _red, green, _blue in kept) / count),
        round(sum(blue for _red, _green, blue in kept) / count),
    )


def _skip(red: int, green: int, blue: int) -> bool:
    peak = max(red, green, blue)
    if peak < 35:
        return True
    if green > red + 12 and green > blue + 8 and green > 70:
        return True
    return peak - min(red, green, blue) < 18 and peak < 170


def _light_cloth(color: tuple[int, int, int]) -> bool:
    peak = max(color)
    floor = min(color)
    return peak > 185 and peak - floor < 55


def _group(
    samples: Sequence[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]],
) -> PlayerColors:
    return PlayerColors(
        _hex(_mean(sample[0] for sample in samples)),
        _hex(_mean(sample[1] for sample in samples)),
        _hex(_mean(sample[2] for sample in samples)),
    )


def _mean(colors: Sequence[tuple[int, int, int]]) -> tuple[int, int, int]:
    rows = list(colors)
    count = len(rows)
    return (
        round(sum(red for red, _green, _blue in rows) / count),
        round(sum(green for _red, green, _blue in rows) / count),
        round(sum(blue for _red, _green, blue in rows) / count),
    )


def full_body_shot(
    frames: Sequence[bytes],
    jersey_hex: str,
    other_hex: str = "",
    avoid: Sequence[bytes] = (),
) -> bytes | None:
    """A sharp look at this jersey. Several teammates in it beat one distant player.

    A close-up or a waist-up group counts. The other team stays out of the crop.
    """

    target = _rgb(jersey_hex)
    other = _rgb(other_hex) if other_hex else None
    if target is None:
        return None
    best: tuple[tuple[int, int, int], object] | None = None
    fresh: tuple[tuple[int, int, int], object] | None = None
    for frame in frames:
        image = _bgr(frame)
        if image is None or not _field_frame(image):
            continue
        matched: list[PlayerBox] = []
        for box in _separate(_hog_people(image)):
            if not _large_enough(box, image.shape[0]):
                continue
            sample = _uniform_colors(image, box)
            if sample is None:
                continue
            own = _distance(sample[0], target)
            if not _jersey_close(sample[0], target, own):
                continue
            if other is not None and _distance(sample[0], other) <= own:
                continue
            if other is not None:
                share_limit = 0.22 if _light_cloth(other) else 0.12
                if _other_share(image, box, other) > share_limit:
                    continue
            piece = _player_crop(image, box)
            if piece is None or _striped(piece):
                continue
            matched.append(box)
        matched.extend(_jersey_boxes(image, target, other))
        settled = False
        for group in _teammate_groups(matched):
            crop = _group_crop(image, group)
            if crop is None or not _clear_uniform_crop(crop):
                continue
            if not _big_on_frame(crop, image) or _grass(crop) > 0.5 or _broadcast_crowd(crop, image):
                continue
            if _broadcast_graphic(crop, target) or _flat_graphic(crop, target):
                continue
            if _broadcast_graphic(crop, target):
                continue
            fill = _color_fill(crop, target)
            if fill < 0.38:
                continue
            sharp = _sharpness(crop)
            rank = (-int(fill * 1000), 0 if len(group) > 1 else 1, -sharp)
            if best is None or rank < best[0]:
                best = (rank, crop)
            if not _matches_saved(crop, avoid) and (fresh is None or rank < fresh[0]):
                fresh = (rank, crop)
            if (
                fill >= 0.32
                and sharp >= 40
                and _grass(crop) <= 0.4
                and _big_on_frame(crop, image)
            ):
                settled = True
        if settled and fresh is not None:
            break
    chosen = fresh or best
    if chosen is None:
        return None
    return _jpeg_edge(chosen[1], 1920)


def _jersey_close(
    sample: tuple[int, int, int],
    target: tuple[int, int, int],
    distance: int,
) -> bool:
    """True when this cloth is the target jersey — gray must not match blue."""

    span = max(target) - min(target)
    bright = max(target) > 165 and span < 45
    if bright:
        sample_span = max(sample) - min(sample)
        return distance <= 70 * 70 * 3 and max(sample) > 150 and sample_span < 55
    # Broadcast crops dull team colors; still reject olive/mud and wrong-channel cloth.
    if distance > 58 * 58 * 3:
        return False
    target_channel = max(range(3), key=lambda index: target[index])
    sample_channel = max(range(3), key=lambda index: sample[index])
    if (
        sample_channel != target_channel
        and sample[target_channel] < sample[sample_channel] - 12
    ):
        return False
    return sample[target_channel] >= target[target_channel] - 55


def lone_players(
    frames: Sequence[bytes],
    avoid: Sequence[bytes] = (),
) -> tuple[bytes | None, bytes | None]:
    """A white jersey and a colored jersey from the film.

    A sharp group of the same jersey beats one player. A blurry or tiny crop
    is left out. Photos in ``avoid`` are skipped when another crop of that
    jersey is in the frames.
    """

    best: dict[str, tuple[tuple[float, int, int], object]] = {}
    fresh: dict[str, tuple[tuple[float, int, int], object]] = {}
    for frame in frames:
        image = _bgr(frame)
        if image is None or not _field_frame(image):
            continue
        height = image.shape[0]
        boxes = [box for box in find_players(frame) if _large_enough(box, height)]
        by_kind: dict[str, list[PlayerBox]] = {"white": [], "color": []}
        for box in boxes:
            crop = _player_crop(image, box)
            if crop is None or _striped(crop):
                continue
            kind = _jersey_kind(crop)
            if kind is None:
                continue
            by_kind[kind].append(box)
        for kind, kind_boxes in by_kind.items():
            for group in _teammate_groups(kind_boxes):
                crop = _group_crop(image, group)
                if crop is None or not _clear_uniform_crop(crop):
                    continue
                if not _big_on_frame(crop, image) or _grass(crop) > 0.5 or _broadcast_crowd(crop, image):
                    continue
                score = (_cloth_fill(crop), 1 if len(group) > 1 else 0, _sharpness(crop))
                current = best.get(kind)
                if current is None or score > current[0]:
                    best[kind] = (score, crop)
                if _matches_saved(crop, avoid):
                    continue
                current_fresh = fresh.get(kind)
                if current_fresh is None or score > current_fresh[0]:
                    fresh[kind] = (score, crop)
    white = fresh.get("white") or best.get("white")
    colored = fresh.get("color") or best.get("color")
    return (
        None if white is None else _jpeg_edge(white[1], 1920),
        None if colored is None else _jpeg_edge(colored[1], 1920),
    )


def usable_player_jpeg(data: bytes) -> bool:
    """True when this JPEG shows a jersey clearly, not a referee or a wide field."""

    image = _bgr(data)
    return image is not None and _clear_uniform_crop(image)


def _usable_player_crop(crop, *, require_alone: bool = False) -> bool:
    """A clear uniform crop. A full body and grass above the helmet are not required."""

    del require_alone
    return _clear_uniform_crop(crop)


def _clear_uniform_crop(crop) -> bool:
    """One jersey, sharp enough to recognize, and not mostly grass or a referee."""

    height, width = crop.shape[:2]
    if height < 64 or width < 36 or max(height, width) < 80:
        return False
    if _striped(crop):
        return False
    if _grass(crop) > 0.65:
        return False
    return _sharpness(crop) >= 8


def _field_frame(image) -> bool:
    """A game frame. A title card, a band, or a graphic has no grass."""

    return _grass(image) >= 0.08


def _flat_graphic(crop, target: tuple[int, int, int]) -> bool:
    """A logo or title card. A white or gray jersey is not a saturated poster."""

    if max(target) - min(target) >= 45:
        return False
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    graphic = (hsv[:, :, 1] > 80) & ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 85))
    return float(graphic.mean()) > 0.30
    """A game frame. A title card, a band, or a graphic has no grass."""

    return _grass(image) >= 0.08


def _broadcast_crowd(crop, image) -> bool:
    """A crop of the whole broadcast. The stands are not a uniform card."""

    if max(image.shape[:2]) < 900:
        return False
    frame = image.shape[0] * image.shape[1]
    return crop.shape[0] * crop.shape[1] > frame * 0.4


def _big_on_frame(crop, image) -> bool:
    """The jersey has to be large in this frame. A distant speck is not a card."""

    frame_long = max(image.shape[:2])
    crop_long = max(crop.shape[:2])
    crop_short = min(crop.shape[:2])
    if frame_long >= 900:
        return crop_long >= int(frame_long * 0.2) and crop_short >= int(frame_long * 0.12)
    return crop_long >= max(96, int(frame_long * 0.11))


def _jersey_boxes(image, target: tuple[int, int, int], other: tuple[int, int, int] | None):
    """Large regions of this jersey, including a group of teammates standing together."""

    import cv2
    import numpy as np

    height, width = image.shape[:2]
    longest = max(height, width)
    scale = 640 / longest if longest > 640 else 1.0
    view = image
    if scale != 1.0:
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    view_height, view_width = view.shape[:2]
    pixels = view.astype(np.int16)
    target_bgr = np.array((target[2], target[1], target[0]), dtype=np.int16)
    distance = np.square(pixels - target_bgr).sum(axis=2)
    bright = max(target) > 165 and max(target) - min(target) < 45
    limit = 42 * 42 * 3
    mask = distance <= limit
    target_chroma = max(target) - min(target)
    if target_chroma < 45:
        chroma = view.max(axis=2).astype(np.int16) - view.min(axis=2).astype(np.int16)
        mask &= chroma <= target_chroma + 28
    if bright:
        mask &= (view[:, :, 0] > 145) & (view[:, :, 1] > 145) & (view[:, :, 2] > 145)
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    grass = (
        (hsv[:, :, 0] > 35)
        & (hsv[:, :, 0] < 80)
        & (hsv[:, :, 1] > 40)
        & (hsv[:, :, 2] > 35)
        & (view[:, :, 1] > view[:, :, 0])
        & (view[:, :, 1] > view[:, :, 2])
    )
    mask &= ~grass
    if other is not None:
        other_bgr = np.array((other[2], other[1], other[0]), dtype=np.int16)
        other_distance = np.square(pixels - other_bgr).sum(axis=2)
        mask &= distance + 800 < other_distance
    binary = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = view_height * view_width
    boxes: list[PlayerBox] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width * box_height < frame_area * 0.012:
            continue
        if box_height < view_height * 0.1 and box_width > box_height * 3:
            continue
        center_y = y + box_height / 2
        if center_y < view_height * 0.06 or center_y > view_height * 0.92:
            continue
        boxes.append(
            PlayerBox(
                int(x / scale),
                int(y / scale),
                max(1, int(box_width / scale)),
                max(1, int(box_height / scale)),
            )
        )
    return boxes


def _broadcast_graphic(crop, target: tuple[int, int, int]) -> bool:
    """A scoreboard band. A saturated team jersey is not one."""

    span = max(target) - min(target)
    if span > 50 and max(target) > 80:
        return False
    height = crop.shape[0]
    band = crop[int(height * 0.7) :, :]
    if band.size == 0:
        return False
    import cv2

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturated = hsv[:, :, 1] > 110
    red = saturated & ((hue < 12) | (hue > 165))
    blue = saturated & (hue > 100) & (hue < 135)
    return float((red | blue).mean()) > 0.28


def _large_enough(box: PlayerBox, frame_height: int) -> bool:
    """A player or jersey big enough to see. A distant speck is not."""

    if box.height < 48 or box.width < 24:
        return False
    return box.height >= max(48, int(frame_height * 0.12))


def _teammate_groups(boxes: Sequence[PlayerBox]) -> list[list[PlayerBox]]:
    """Players of one jersey who stand near each other, kept as one crop."""

    pending = list(boxes)
    groups: list[list[PlayerBox]] = []
    while pending:
        group = [pending.pop(0)]
        grew = True
        while grew:
            grew = False
            still: list[PlayerBox] = []
            for box in pending:
                if any(_near_teammate(box, other) for other in group):
                    group.append(box)
                    grew = True
                else:
                    still.append(box)
            pending = still
        groups.append(group)
    return groups


def _near_teammate(left: PlayerBox, right: PlayerBox) -> bool:
    lx = left.x + left.width / 2
    ly = left.y + left.height / 2
    rx = right.x + right.width / 2
    ry = right.y + right.height / 2
    limit = 1.25 * (left.width + right.width)
    return (lx - rx) ** 2 + (ly - ry) ** 2 <= limit ** 2


def _group_crop(image, group: Sequence[PlayerBox]):
    """The teammates, with a small margin. The jersey stays the subject."""

    if not group:
        return None
    frame_height, frame_width = image.shape[:2]
    left = min(box.x for box in group)
    top = min(box.y for box in group)
    right = max(box.x + box.width for box in group)
    bottom = max(box.y + box.height for box in group)
    width = max(1, right - left)
    height = max(1, bottom - top)
    pad_x = int(width * 0.12)
    pad_y = int(height * 0.08)
    x0 = max(0, left - pad_x)
    y0 = max(0, top - pad_y)
    x1 = min(frame_width, right + pad_x)
    y1 = min(frame_height, bottom + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 16:
        return None
    return image[y0:y1, x0:x1]


def _color_fill(crop, target: tuple[int, int, int]) -> float:
    """Share of the crop that is this jersey, grass left out."""

    total = 0
    matched = 0
    for blue, green, red in crop.reshape(-1, 3)[::2]:
        red, green, blue = int(red), int(green), int(blue)
        if _skip(red, green, blue) and green > red + 12 and green > blue + 8:
            continue
        total += 1
        pixel = (red, green, blue)
        if not _same_cloth(pixel, target):
            continue
        matched += 1
    if total < 8:
        return 0.0
    return matched / total


def _cloth_fill(crop) -> float:
    """Share of the crop that is cloth rather than grass."""

    return max(0.0, 1.0 - _grass(crop))


def _sharpness(crop) -> int:
    """How much edge detail is in the crop. A blurry broadcast frame scores low.

    Luminance misses a bright jersey against similarly bright grass, so each
    color channel counts too.
    """

    import cv2

    scores = [cv2.Laplacian(crop[:, :, channel], cv2.CV_64F).var() for channel in range(3)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return int(min(5000, max(scores)))


def _matches_saved(crop, saved: Sequence[bytes]) -> bool:
    """True when this crop is the same player photo already on the card."""

    if not saved:
        return False
    import cv2

    small = cv2.resize(crop, (24, 40), interpolation=cv2.INTER_AREA)
    for item in saved:
        other = _bgr(item)
        if other is None:
            continue
        thumb = cv2.resize(other, (24, 40), interpolation=cv2.INTER_AREA)
        if float(cv2.absdiff(small, thumb).mean()) < 16:
            return True
    return False


def _detected_people(image) -> list[PlayerBox]:
    """People large enough to show a uniform. Coordinates are on this frame."""

    import cv2

    height = image.shape[0]
    scale = 1100 / height if 0 < height < 1100 else 1.0
    view = image
    if scale != 1.0:
        view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, weights = hog.detectMultiScale(
        view,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    boxes: list[PlayerBox] = []
    for index, (x, y, width, height) in enumerate(found):
        weight = _weight(weights, index)
        if weight < 0.5:
            continue
        box = PlayerBox(
            int(x / scale),
            int(y / scale),
            max(1, int(width / scale)),
            max(1, int(height / scale)),
        )
        if box.height < 70 or box.width < 16:
            continue
        boxes.append(box)
    return boxes


def _crowd(box: PlayerBox, boxes: Sequence[PlayerBox]) -> bool:
    """True when another person stands inside this crop."""

    for other in boxes:
        if other is box:
            continue
        if _overlap(box, other) > 0.55 and _overlap(other, box) > 0.55:
            continue
        center_x = other.x + other.width / 2
        center_y = other.y + other.height / 2
        if box.x <= center_x <= box.x + box.width and box.y <= center_y <= box.y + box.height:
            return True
    return False


def _alone(crop) -> bool:
    """A full uniform on the field, with grass above the helmet and beside the body."""

    height, width = crop.shape[:2]
    if height < 90 or width < 28:
        return False
    top = crop[: max(1, int(height * 0.2)), :]
    if _grass(top[:, : max(1, int(width * 0.3))]) < 0.9:
        return False
    if _grass(top[:, int(width * 0.7) :]) < 0.9:
        return False
    if _grass(top[:, int(width * 0.28) : int(width * 0.72)]) < 0.4:
        return False
    if _grass(crop[int(height * 0.82) :, :]) < 0.5:
        return False
    torso = crop[int(height * 0.25) : int(height * 0.75), :]
    left = _grass(torso[:, : max(1, int(width * 0.16))])
    right = _grass(torso[:, int(width * 0.84) :])
    return left >= 0.75 and right >= 0.75


def _open_field(crop) -> bool:
    """Grass on both sides of the torso — enough room that this is not a pile."""

    height, width = crop.shape[:2]
    if height < 90 or width < 28:
        return False
    torso = crop[int(height * 0.25) : int(height * 0.7), :]
    left = _grass(torso[:, : max(1, int(width * 0.14))])
    right = _grass(torso[:, int(width * 0.86) :])
    return left >= 0.4 and right >= 0.4


def _grass(region) -> float:
    if region.size == 0:
        return 0.0
    import cv2

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    green = region[:, :, 1]
    mask = (
        (hue > 35)
        & (hue < 80)
        & (hsv[:, :, 1] > 40)
        & (hsv[:, :, 2] > 35)
        & (green > region[:, :, 0])
        & (green > region[:, :, 2])
    )
    return float(mask.mean())


def _striped(crop) -> bool:
    """A black-and-white striped official, including a soft broadcast picture."""

    try:
        return _sharp_stripes(crop) or _soft_stripes(crop) or _adaptive_stripes(crop)
    except Exception:
        # A bad crop must not stop the team cards.
        return False


def _sharp_stripes(crop) -> bool:
    """Crisp vertical gray bars. Colored cloth is left out by saturation."""

    height, width = crop.shape[:2]
    torso = crop[int(height * 0.18) : int(height * 0.55), int(width * 0.2) : int(width * 0.8)]
    if torso.size == 0:
        return False
    import cv2
    import numpy as np

    sat = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32).mean(axis=0)
    if float(np.mean(sat)) > 55:
        return False
    columns = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=0)
    for cutoff in (145.0, float(np.median(columns))):
        light = columns > cutoff
        flips = int(np.sum(light[1:] != light[:-1]))
        if flips >= 3:
            return True
    return False


def _soft_stripes(crop) -> bool:
    """Gray bars on the official once the picture is too soft for a hard cutoff.

    The shirt is measured on the person, not the grass around them. A white
    jersey stays bright with little dark/light alternation. A colored jersey's
    bright columns are not gray.
    """

    shirt = _shirt_columns(crop)
    if shirt is None:
        return False
    columns, saturation = shirt
    import numpy as np

    if columns.size < 8:
        return False
    median = float(np.median(columns))
    span = float(columns.max() - columns.min())
    # Bright white bars can push the median up; still treat strong zebra contrast.
    if median > 185 and span < 55:
        return False
    if median > 165 and span < 40:
        return False
    smooth = np.convolve(columns, np.ones(3) / 3, mode="same")
    peaks = _gray_peaks(smooth, saturation)
    return _dark_bar_between(peaks, smooth, saturation)


def _adaptive_stripes(crop) -> bool:
    """Vertical light/dark bars around the local median — typical referee zebra.

    Fixed white cutoffs miss mid-tone broadcast shirts. A solid white jersey
    has almost no dark columns; a colored jersey keeps higher saturation.
    Soft distant crops raise saturation from grass bleed, so a high flip count
    still counts when enough of the shirt stays near gray.
    """

    import cv2
    import numpy as np

    height, width = crop.shape[:2]
    if height < 40 or width < 20:
        return False
    torso = crop[
        int(height * 0.16) : int(height * 0.58),
        int(width * 0.16) : int(width * 0.84),
    ]
    if torso.size == 0 or torso.shape[1] < 8:
        return False
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    columns = gray.mean(axis=0)
    column_sat = sat.mean(axis=0)
    mean_sat = float(np.mean(column_sat))
    if mean_sat > 110:
        return False
    span = float(columns.max() - columns.min())
    if span < 32:
        return False
    median = float(np.median(columns))
    light = columns > median
    flips = int(np.sum(light[1:] != light[:-1]))
    if flips < 4:
        return False
    bright = float(np.mean(columns > median + 8))
    dark = float(np.mean(columns < median - 8))
    if bright < 0.10 or dark < 0.12:
        return False
    # Alternating bars stay near gray; a number or logo is a small patch.
    gray_share = float(np.mean(column_sat < 80))
    if flips >= 6 and gray_share >= 0.28:
        return True
    if mean_sat > 58:
        return False
    return gray_share >= 0.7


def _shirt_columns(crop):
    """Brightness and saturation across the official's shirt, grass removed."""

    import cv2
    import numpy as np

    height, width = crop.shape[:2]
    if height < 40 or width < 16:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 40)
    column_grass = grass.mean(axis=0)
    core = int(np.argmin(column_grass))
    if float(column_grass[core]) > 0.8:
        return None
    left = core
    while left > 0 and float(column_grass[left - 1]) < 0.84:
        left -= 1
    right = core + 1
    while right < width and float(column_grass[right]) < 0.84:
        right += 1
    if right - left < 8:
        return None
    row_grass = grass[:, left:right].mean(axis=1)
    rows = np.flatnonzero(row_grass < 0.72)
    if rows.size < 16:
        return None
    body = crop[int(rows[0]) : int(rows[-1]) + 1, left:right]
    shirt = body[int(body.shape[0] * 0.05) : int(body.shape[0] * 0.62)]
    if shirt.shape[0] < 8 or shirt.shape[1] < 8:
        return None
    columns = cv2.cvtColor(shirt, cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=0)
    saturation = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32).mean(axis=0)
    cloth = saturation < 110
    if int(cloth.sum()) < 8:
        return None
    start, end = _longest_run(cloth)
    if end - start < 8:
        return None
    return columns[start:end], saturation[start:end]


def _longest_run(mask) -> tuple[int, int]:
    best = (0, 0)
    start: int | None = None
    for index, bit in enumerate(mask):
        if bit and start is None:
            start = index
        elif not bit and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask))
    return best


def _gray_peaks(smooth, saturation) -> list[int]:
    """Light gray bars. A colored jersey's bright columns are left out."""

    peaks: list[int] = []
    for index in range(1, len(smooth) - 1):
        if smooth[index] < smooth[index - 1] or smooth[index] <= smooth[index + 1]:
            continue
        left = smooth[max(0, index - 8) : index]
        if left.size == 0:
            continue
        rise = float(smooth[index] - left.min())
        if rise < 12 or float(smooth[index]) < 115 or float(saturation[index]) > 65:
            continue
        if peaks and index - peaks[-1] <= 2:
            if smooth[index] > smooth[peaks[-1]]:
                peaks[-1] = index
            continue
        peaks.append(index)
    return peaks


def _dark_bar_between(peaks: list[int], smooth, saturation) -> bool:
    """A black bar between two light bars. One jersey number is not that pattern."""

    import numpy as np

    # Pair each peak with the next one.
    for left, right in pairwise(peaks):
        if right - left < 3:
            continue
        floor_index = left + int(np.argmin(smooth[left : right + 1]))
        if floor_index >= len(smooth) or floor_index >= len(saturation):
            continue
        floor = float(smooth[floor_index])
        if floor > 90 or float(saturation[floor_index]) > 120:
            continue
        if float(smooth[left]) - floor < 25 or float(smooth[right]) - floor < 25:
            continue
        return True
    return False


def _jersey_kind(crop) -> str | None:
    """White cloth, or a solid color. Gray sideline clothes are neither."""

    if _striped(crop):
        return None
    sample = _jersey_sample(crop)
    if sample is None:
        return None
    saturation, value, fill = sample
    if fill < 0.12:
        return None
    if value >= 165 and saturation < 80:
        return "white"
    if saturation >= 40 or value < 90:
        return "color"
    return None


def _jersey_sample(crop) -> tuple[float, float, float] | None:
    import cv2

    height, width = crop.shape[:2]
    band = crop[int(height * 0.2) : int(height * 0.5), int(width * 0.25) : int(width * 0.75)]
    if band.size == 0:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    kept: list[tuple[int, int]] = []
    for hue, saturation, value in hsv.reshape(-1, 3):
        hue, saturation, value = int(hue), int(saturation), int(value)
        if 35 < hue < 95 and saturation > 40 and value > 35:
            continue
        if value < 12:
            continue
        kept.append((saturation, value))
    if len(kept) < 12:
        return None
    count = len(kept)
    return (
        sum(item[0] for item in kept) / count,
        sum(item[1] for item in kept) / count,
        count / band.reshape(-1, 3).shape[0],
    )


def _jersey_sample_rgb(crop) -> tuple[int, int, int] | None:
    """Average jersey RGB from the torso band, grass skipped."""

    height, width = crop.shape[:2]
    band = crop[int(height * 0.22) : int(height * 0.52), int(width * 0.25) : int(width * 0.75)]
    if band.size == 0:
        return None
    kept: list[tuple[int, int, int]] = []
    for blue, green, red in band.reshape(-1, 3)[::2]:
        red, green, blue = int(red), int(green), int(blue)
        if _skip(red, green, blue):
            continue
        kept.append((red, green, blue))
    if len(kept) < 8:
        return None
    count = len(kept)
    return (
        round(sum(red for red, _green, _blue in kept) / count),
        round(sum(green for _red, green, _blue in kept) / count),
        round(sum(blue for _red, _green, blue in kept) / count),
    )


def uniform_crops(jpeg: bytes, jersey_hex: str, limit: int = 2) -> list[bytes]:
    """Close crops of the players wearing this jersey, so the cloth fills the frame."""

    image = _bgr(jpeg)
    target = _rgb(jersey_hex)
    if image is None or target is None or limit < 1:
        return []
    try:
        import cv2
    except ImportError:
        return []
    mask = _jersey_mask(image, target)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    frame_area = height * width
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < frame_area * 0.004 or area > frame_area * 0.55:
            continue
        if box_height < 24 or box_width < 16:
            continue
        boxes.append((x, y, box_width, box_height))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    crops: list[bytes] = []
    for x, y, box_width, box_height in boxes:
        crop = _expanded(image, x, y, box_width, box_height)
        if _striped(crop):
            continue
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        crops.append(encoded.tobytes())
        if len(crops) == limit:
            break
    return crops


def colors_on_crop(jpeg: bytes) -> PlayerColors | None:
    """Helmet, jersey, and pants measured on one close crop of a player."""

    image = _bgr(jpeg)
    if image is None:
        return None
    height, width = image.shape[:2]
    sample = _uniform_colors(image, PlayerBox(0, 0, width, height))
    if sample is None:
        return None
    jersey, pants, helmet = sample
    return PlayerColors(_hex(jersey), _hex(pants), _hex(helmet))


def _rgb(value: str) -> tuple[int, int, int] | None:
    text = value.strip()
    if len(text) != 7 or not text.startswith("#"):
        return None
    try:
        return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))
    except ValueError:
        return None


def _jersey_mask(image, target: tuple[int, int, int]):
    import cv2
    import numpy as np

    blue, green, red = cv2.split(image)
    distance = (
        (red.astype(np.int32) - target[0]) ** 2
        + (green.astype(np.int32) - target[1]) ** 2
        + (blue.astype(np.int32) - target[2]) ** 2
    )
    mask = (distance < 70 * 70).astype(np.uint8) * 255
    thin = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, thin)
    block = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, block)


def _expanded(image, x: int, y: int, width: int, height: int):
    frame_height, frame_width = image.shape[:2]
    pad_x = int(width * 0.12)
    pad_y = int(height * 0.08)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(frame_width, x + width + pad_x)
    bottom = min(frame_height, y + height + pad_y)
    return image[top:bottom, left:right]


def _standing(box: PlayerBox) -> bool:
    """A clear full-body player, not a distant speck in the wide shot."""

    return box.height >= 140 and box.width >= 32 and box.height >= box.width * 1.55


def _inside(box: PlayerBox, frame_width: int, frame_height: int) -> bool:
    return (
        box.y > 4
        and box.x > 4
        and box.y + box.height < frame_height - 4
        and box.x + box.width < frame_width - 4
    )


def _same_cloth(pixel: tuple[int, int, int], target: tuple[int, int, int]) -> bool:
    """This pixel is the jersey, not a nearby crowd color."""

    if _distance(pixel, target) > 42 * 42 * 3:
        return False
    target_chroma = max(target) - min(target)
    pixel_chroma = max(pixel) - min(pixel)
    return not (target_chroma < 45 and pixel_chroma > target_chroma + 28)


def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2


def _other_share(image, box: PlayerBox, other: tuple[int, int, int]) -> float:
    """Share of the cloth in this box that belongs to the other jersey."""

    height, width = image.shape[:2]
    x = max(0, box.x)
    y = max(0, box.y)
    box_width = min(box.width, width - x)
    box_height = min(box.height, height - y)
    crop = image[y : y + box_height, x : x + box_width]
    if crop.size == 0:
        return 1.0
    # White uniforms need a tighter match so gray turf glare is not counted.
    limit = 40 * 40 * 3 if _light_cloth(other) else 55 * 55 * 3
    other_pixels = 0
    cloth = 0
    for blue, green, red in crop.reshape(-1, 3)[::3]:
        red, green, blue = int(red), int(green), int(blue)
        if _skip(red, green, blue):
            continue
        cloth += 1
        if _distance((red, green, blue), other) < limit:
            other_pixels += 1
    if cloth < 8:
        return 1.0
    return other_pixels / cloth


def _player_crop(image, box: PlayerBox):
    frame_height, frame_width = image.shape[:2]
    pad_x = int(box.width * 0.08)
    pad_y = int(box.height * 0.04)
    left = max(0, box.x - pad_x)
    top = max(0, box.y - pad_y)
    right = min(frame_width, box.x + box.width + pad_x)
    bottom = min(frame_height, box.y + box.height + pad_y)
    if right - left < 8 or bottom - top < 16:
        return None
    return image[top:bottom, left:right]


def crop_picked_player(
    frame: bytes, box: tuple[float, float, float, float]
) -> bytes | None:
    """Cut the player a model boxed out of a full-size frame, as a portrait card.

    The box is in fractions of the frame and is often a little tight or off, so
    the crop is padded. Referees and players too small to show a uniform are left out.
    """

    image = _bgr(frame)
    if image is None:
        return None
    import cv2

    frame_height, frame_width = image.shape[:2]
    left, top, right, bottom = box
    pad_x = (right - left) * 0.3
    pad_y = (bottom - top) * 0.1
    x0 = max(0, int((left - pad_x) * frame_width))
    x1 = min(frame_width, int(round((right + pad_x) * frame_width)))
    y0 = max(0, int((top - pad_y) * frame_height))
    y1 = min(frame_height, int(round((bottom + pad_y) * frame_height)))
    crop = image[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    if height < 60 or width < 24 or height < width:
        return None
    if _striped(crop):
        return None
    if height < 480:
        scale = min(4.0, 480 / height)
        crop = cv2.resize(
            crop,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return _jpeg_edge(crop, 640)


def _jpeg_edge(image, edge: int) -> bytes | None:
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > edge:
        scale = edge / longest
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return encoded.tobytes()


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _separate(boxes: list[PlayerBox]) -> list[PlayerBox]:
    kept: list[PlayerBox] = []
    for box in sorted(boxes, key=lambda item: item.width * item.height, reverse=True):
        if all(_overlap(box, other) < 0.45 for other in kept):
            kept.append(box)
    return kept[:12]


def _overlap(left: PlayerBox, right: PlayerBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    shared = (x2 - x1) * (y2 - y1)
    area = left.width * left.height
    return shared / area if area else 0.0
