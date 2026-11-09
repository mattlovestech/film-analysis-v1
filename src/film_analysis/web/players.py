"""Find players in a film frame and read the uniform colors on each one.

OpenCV's people detector supplies the boxes. NumPy holds the pixels inside
each box so the jersey and pants can be averaged without the grass.
"""

from collections.abc import Sequence
from dataclasses import dataclass


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

    height = image.shape[0]
    scale = 720 / height if height < 720 else 1.0
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
        if area < frame_area * 0.002 or area > frame_area * 0.18:
            continue
        if box_height < box_width * 0.45:
            continue
        boxes.append((x, y, box_width, box_height))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
    crops: list[bytes] = []
    for x, y, box_width, box_height in boxes:
        crop = _expanded(image, x, y, box_width, box_height)
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
    pad_x = int(width * 0.4)
    above = int(height * 0.7)
    below = int(height * 1.15)
    left = max(0, x - pad_x)
    top = max(0, y - above)
    right = min(frame_width, x + width + pad_x)
    bottom = min(frame_height, y + height + below)
    return image[top:bottom, left:right]


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
