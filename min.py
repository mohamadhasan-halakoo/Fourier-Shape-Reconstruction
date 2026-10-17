import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_contour(image_path, num_points=512):
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Extract external contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError("No contour found in the image.")

    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    points = contour[:, 0, :].astype(float)

    # Close the contour
    if np.linalg.norm(points[0] - points[-1]) > 1e-8:
        points = np.vstack([points, points[0]])

    # Compute cumulative arc length
    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    cumulative_length = np.concatenate([
        [0],
        np.cumsum(segment_lengths)
    ])

    total_length = cumulative_length[-1]

    if total_length == 0:
        raise ValueError("The extracted contour is degenerate.")

    # Uniform arc-length resampling
    target_lengths = np.linspace(
        0,
        total_length,
        num_points,
        endpoint=False
    )

    x = np.interp(
        target_lengths,
        cumulative_length,
        points[:, 0]
    )

    y = np.interp(
        target_lengths,
        cumulative_length,
        points[:, 1]
    )

    # Complex representation
    z = x + 1j * y

    return image, z


def reconstruct_fourier(z, num_coefficients):
    """
    Keep the strongest Fourier coefficients and reconstruct
    the contour.

    num_coefficients is the total number of retained coefficients.
    """

    N = len(z)

    # Compute Fourier coefficients
    coefficients = np.fft.fft(z) / N

    # Rank frequencies by coefficient magnitude
    ranking = np.argsort(np.abs(coefficients))[::-1]

    # Keep the strongest coefficients
    selected = ranking[:min(num_coefficients, N)]

    filtered = np.zeros(N, dtype=complex)
    filtered[selected] = coefficients[selected]

    # Reconstruct the contour
    reconstructed = np.fft.ifft(filtered * N)

    return reconstructed


def main():
    image_path = "input.png"

    image, z = extract_contour(image_path)

    coefficient_counts = [5, 10, 20, 50, 100]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9)
    )

    # Original image
    axes[0, 0].imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    # Original contour
    axes[0, 1].plot(z.real, -z.imag)
    axes[0, 1].set_title("Extracted contour")
    axes[0, 1].axis("equal")

    for ax, count in zip(
        [axes[0, 2], axes[1, 0],
         axes[1, 1], axes[1, 2]],
        coefficient_counts[:4]
    ):
        reconstructed = reconstruct_fourier(z, count)

        ax.plot(
            reconstructed.real,
            -reconstructed.imag
        )

        ax.set_title(
            f"Reconstruction: {count} coefficients"
        )

        ax.axis("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()