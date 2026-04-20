URL_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/640px-Bikesgray.jpg"
URL_2 = ""
URL_3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

ROBERTS_THRESHOLD = 60
CANNY_LOW         = 50
CANNY_HIGH        = 150
NOISE_AMOUNT      = 40

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import urllib.request
from io import BytesIO
from PIL import Image


def load_image_from_url(url):
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("L")
    return np.array(img, dtype=np.uint8)


def load_image_color_from_url(url):
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def add_gaussian_noise(img, amount=40):
    noise = np.random.normal(0, amount, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def roberts_edge_detection(img, threshold=60):
    img_f = img.astype(np.float32)
    rows, cols = img_f.shape

    a = img_f[:-1, :-1]
    b = img_f[:-1,  1:]
    c = img_f[ 1:, :-1]
    d = img_f[ 1:,  1:]

    Gx = a - d
    Gy = b - c
    G  = np.sqrt(Gx**2 + Gy**2)

    edges = np.zeros((rows, cols), dtype=np.uint8)
    edges[:-1, :-1] = np.where(G > threshold, 255, 0).astype(np.uint8)

    return edges


print("Ielādē attēlus...")

img1_color = load_image_color_from_url(URL_1)
img1       = np.array(Image.fromarray(img1_color).convert("L"), dtype=np.uint8)

if URL_2.strip():
    img2_color = load_image_color_from_url(URL_2)
    img2       = np.array(Image.fromarray(img2_color).convert("L"), dtype=np.uint8)
else:
    img2_color = add_gaussian_noise(img1_color, NOISE_AMOUNT)
    img2       = add_gaussian_noise(img1, NOISE_AMOUNT)

img3_color = load_image_color_from_url(URL_3)
img3       = np.array(Image.fromarray(img3_color).convert("L"), dtype=np.uint8)

images       = [img1, img2, img3]
images_color = [img1_color, img2_color, img3_color]
labels = [
    "Attēls 1\n(skaidrs objekts)",
    "Attēls 2\n(ar troksni)",
    "Attēls 3\n(brīva izvēle)",
]

print("Apstrādā attēlus...")

canny_results   = []
roberts_results = []

for img in images:
    canny_results.append(cv2.Canny(img, CANNY_LOW, CANNY_HIGH))
    roberts_results.append(roberts_edge_detection(img, ROBERTS_THRESHOLD))

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    "Malu detektēšanas atskaite\n"
    f"Canny: low={CANNY_LOW}, high={CANNY_HIGH}  |  Roberts: threshold={ROBERTS_THRESHOLD}",
    fontsize=15, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.25)
col_titles = [
    "Oriģināls (krāsains)",
    "Oriģināls (pelēktoņi)",
    "Canny malu detektors",
    "Roberts operators",
]

for row, (color, gray, canny, roberts, lbl) in enumerate(
    zip(images_color, images, canny_results, roberts_results, labels)
):
    for col, (data, ctitle) in enumerate(zip([color, gray, canny, roberts], col_titles)):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(data) if col == 0 else ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        top_title = ctitle if row == 0 else ""
        row_label = lbl   if col == 0 else ""
        if top_title and row_label:
            ax.set_title(f"{top_title}\n{row_label}", fontsize=10)
        elif top_title:
            ax.set_title(top_title, fontsize=10)
        elif row_label:
            ax.set_title(row_label, fontsize=10)

plt.savefig("atskaite.png", dpi=150, bbox_inches="tight")
plt.show()

fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
fig2.suptitle("Canny vs Roberts — detalizēts salīdzinājums", fontsize=13)

for r, (method_name, results) in enumerate([("Canny", canny_results), ("Roberts", roberts_results)]):
    for c, (result, lbl) in enumerate(zip(results, labels)):
        axes[r, c].imshow(result, cmap="gray", vmin=0, vmax=255)
        axes[r, c].axis("off")
        axes[r, c].set_title(f"{method_name}\n{lbl}", fontsize=9)

plt.tight_layout()
plt.savefig("salīdzinājums.png", dpi=150, bbox_inches="tight")
plt.show()
