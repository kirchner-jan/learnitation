import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, rotate, shift, zoom
from torch.utils.data import Dataset


class BasicShapeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def apply_transformations(image, rotation=0, scale=1.0, translation=(0, 0), size=28):
    """Apply geometric transformations to an image"""
    # Ensure image is float32
    image = image.astype(np.float32)

    # Apply rotation
    if rotation != 0:
        image = rotate(image, rotation, reshape=False, mode="constant", cval=0)

    # Apply scaling
    if scale != 1.0:
        scaled = zoom(image, scale, mode="constant", cval=0, order=1)
        if scale > 1.0:
            # If scaled up, take center crop
            excess = scaled.shape[0] - size
            if excess > 0:
                crop = excess // 2
                image = scaled[crop : crop + size, crop : crop + size]
        else:
            # If scaled down, pad to original size
            pad_size = (size - scaled.shape[0]) // 2
            image = np.pad(scaled, pad_size, mode="constant")
            # Ensure exact size
            image = image[:size, :size]

    # Apply translation
    if translation != (0, 0):
        image = shift(image, translation, mode="constant", cval=0)

    # Ensure final size is correct
    image = image[:size, :size]
    if image.shape != (size, size):
        pad_width = ((0, size - image.shape[0]), (0, size - image.shape[1]))
        image = np.pad(image, pad_width, mode="constant")

    return image


def create_circle(size=28, rotation=0, scale=1.0, translation=(0, 0), radius=None):
    if radius is None:
        radius = size // 4
    canvas = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    y, x = np.ogrid[-center : size - center, -center : size - center]
    mask = x * x + y * y <= radius * radius
    canvas[mask] = 1
    return apply_transformations(canvas, rotation, scale, translation, size)


def create_square(size=28, rotation=0, scale=1.0, translation=(0, 0)):
    canvas = np.zeros((size, size), dtype=np.float32)
    margin = size // 4
    canvas[margin : size - margin, margin : size - margin] = 1
    return apply_transformations(canvas, rotation, scale, translation, size)


def create_line(size=28, rotation=0, scale=1.0, translation=(0, 0)):
    canvas = np.zeros((size, size), dtype=np.float32)
    margin = size // 4
    thickness = size // 8
    start = size // 2 - thickness // 2
    end = start + thickness
    canvas[start:end, margin : size - margin] = 1
    return apply_transformations(canvas, rotation, scale, translation, size)


def create_triangle(size=28, rotation=0, scale=1.0, translation=(0, 0)):
    canvas = np.zeros((size, size), dtype=np.float32)
    margin = size // 4
    height = size - 2 * margin
    base = height

    # Create the triangle points
    top = (size // 2, margin)
    bottom_left = (size // 2 - base // 2, size - margin)
    bottom_right = (size // 2 + base // 2, size - margin)

    # Draw the triangle using vectorized operations
    y, x = np.mgrid[0:size, 0:size]
    points = np.vstack([x.ravel(), y.ravel()])

    # Compute barycentric coordinates
    v0 = np.array([bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]])
    v1 = np.array([top[0] - bottom_left[0], top[1] - bottom_left[1]])
    v2 = points - np.array([[bottom_left[0]], [bottom_left[1]]])

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)

    # Compute barycentric coordinates
    dots = np.dot(v2.T, np.array([v0, v1]))
    dot02 = dots[:, 0]
    dot12 = dots[:, 1]

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is inside triangle
    mask = (u >= 0) & (v >= 0) & (u + v <= 1)
    canvas[points[1, mask], points[0, mask]] = 1

    return apply_transformations(canvas, rotation, scale, translation, size)


def create_pentagon(size=28, rotation=0, scale=1.0, translation=(0, 0)):
    canvas = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 3
    vertices = []

    # Generate pentagon vertices
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        vertices.append((int(x), int(y)))

    # Draw the pentagon
    for i in range(5):
        j = (i + 1) % 5
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]

        # Draw line between vertices
        t = np.linspace(0, 1, 100)
        x = (x1 * (1 - t) + x2 * t).astype(int)
        y = (y1 * (1 - t) + y2 * t).astype(int)
        mask = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        canvas[y[mask], x[mask]] = 1

    # Fill the pentagon
    from scipy.ndimage import binary_fill_holes

    canvas = binary_fill_holes(canvas).astype(np.float32)
    return apply_transformations(canvas, rotation, scale, translation, size)


def create_figure_eight(size=28, rotation=0, scale=1.0, translation=(0, 0)):
    canvas = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 6

    for cy in [center - radius, center + radius]:
        y, x = np.ogrid[-(cy) : size - cy, -center : size - center]
        mask = x * x + y * y <= radius * radius
        canvas[mask] = 1

    return apply_transformations(canvas, rotation, scale, translation, size)


def create_shape_dataset(
    num_samples=1000,
    size=28,
    test_train_split=0.8,
    rotation_range=(-30, 30),
    scale_range=(0.8, 1.2),
    translation_range=(-2, 2),
    include_shapes={
        0: create_circle,
        1: create_square,
        2: create_line,
        3: create_triangle,
        4: create_pentagon,
        5: create_figure_eight,
    },
    turn_off_morphism_for_shape={i: False for i in range(6)},
):
    """
    Create dataset of basic shapes with random transformations.
    Uninclude shapes by putting in a redacted include_shapes arg.
    Turn off e.g. rotation for shape 1 by setting turn_off_morphism_for_shape[1]
    to {"rotation": True, "scale": False, "translation": False}
    """
    shape_functions = include_shapes

    samples_per_class = num_samples // len(shape_functions)
    total_samples = samples_per_class * len(shape_functions)

    # Pre-allocate arrays
    X = np.zeros((total_samples, size, size), dtype=np.float32)
    y = np.zeros(total_samples, dtype=np.int64)
    transformations = []

    idx = 0
    for shape_idx in shape_functions:
        for _ in range(samples_per_class):
            # Generate random transformations
            rotation = np.random.uniform(*rotation_range)
            scale = np.random.uniform(*scale_range)
            tx = np.random.randint(*translation_range)
            ty = np.random.randint(*translation_range)

            if isinstance(turn_off_morphism_for_shape[shape_idx], dict):
                if turn_off_morphism_for_shape[shape_idx]["rotation"]:
                    rotation = 0
                if turn_off_morphism_for_shape[shape_idx]["scale"]:
                    scale = 0
                if turn_off_morphism_for_shape[shape_idx]["translation"]:
                    tx = 0
                    ty = 0
            # Create shape with transformations
            shape = shape_functions[shape_idx](
                size=size, rotation=rotation, scale=scale, translation=(tx, ty)
            )

            X[idx] = shape
            y[idx] = shape_idx
            transformations.append(
                {"rotation": rotation, "scale": scale, "translation": (tx, ty)}
            )
            idx += 1

    # Shuffle the dataset
    indices = np.random.permutation(total_samples)
    X = X[indices]
    y = y[indices]
    transformations = [transformations[i] for i in indices]

    # Split into train and test
    split_idx = int(test_train_split * total_samples)
    train_dataset = BasicShapeDataset(X[:split_idx], y[:split_idx])
    test_dataset = BasicShapeDataset(X[split_idx:], y[split_idx:])

    return (
        train_dataset,
        test_dataset,
        transformations[:split_idx],
        transformations[split_idx:],
    )


def visualize_shapes(dataset, transformations, indices=None):
    """Visualize samples with their transformation parameters"""
    shape_names = ["Circle", "Square", "Line", "Triangle", "Pentagon", "Figure Eight"]

    if indices is None:
        indices = np.random.choice(len(dataset), 6, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        trans = transformations[idx]

        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(
            f'{shape_names[label]}\nRot: {trans["rotation"]:.1f}°\n'
            f'Scale: {trans["scale"]:.2f}\n'
            f'Trans: {trans["translation"]}'
        )

    plt.tight_layout()
    plt.show()


no_rotate_square = {i: False for i in range(6)}
no_rotate_square[1] = {"rotation": True, "scale": False, "translation": False}

train_dataset, test_dataset, train_transforms, test_transforms = create_shape_dataset(
    num_samples=1000,
    rotation_range=(-45, 45),
    scale_range=(0.7, 1.3),
    translation_range=(-3, 3),
    turn_off_morphism_for_shape=no_rotate_square,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
visualize_shapes(train_dataset, train_transforms)
