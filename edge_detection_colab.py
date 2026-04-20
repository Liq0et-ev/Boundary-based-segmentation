# ============================================================
#  Malu detektēšana – Canny & Roberts operators
#  Google Colab — palaidiet šūnas secīgi
# ============================================================

# ---------- UZSTĀDĪJUMI: aizstājiet saites ar saviem attēliem ----------
URL_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesgray.jpg/640px-Bikesgray.jpg"
# ^ Attēls ar skaidri redzamu objektu

URL_2 = ""   # Atstājiet tukšu — troksnis tiks pievienots URL_1 automātiski
             # VAI ievadiet saiti uz jau trokšņainu attēlu

URL_3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
# ^ Brīvi izvēlēts attēls

ROBERTS_THRESHOLD = 60   # Slieksnis Roberts operatoram (0–255)
CANNY_LOW         = 50   # Canny zemākais slieksnis
CANNY_HIGH        = 150  # Canny augstākais slieksnis
NOISE_AMOUNT      = 40   # Gausa trokšņa intensitāte — standartnovirze σ (0–255)
# -----------------------------------------------------------------------


# ── 1. Bibliotēkas ──────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import urllib.request
from io import BytesIO
from PIL import Image


# ── 2. Palīgfunkcijas ───────────────────────────────────────────────────

def load_image_from_url(url: str) -> np.ndarray:
    """Ielādē attēlu no URL kā pelēktoņu numpy masīvu (uint8)."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("L")   # pelēktoņi
    return np.array(img, dtype=np.uint8)


def load_image_color_from_url(url: str) -> np.ndarray:
    """Ielādē attēlu no URL kā RGB krāsu numpy masīvu (uint8)."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def add_gaussian_noise(img: np.ndarray, amount: int = 40) -> np.ndarray:
    """
    Pievieno Gausa troksni attēlam.
    amount — standartnovirze σ; lielāka vērtība = vairāk trokšņa.
    Darbojas ar pelēktoņu un krāsu attēliem.
    """
    noise = np.random.normal(0, amount, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


# ── 3. Roberts operators (BEZ iebūvētām bibliotēku metodēm) ─────────────
#
#  Matricas (no lekcijas):
#    Mx = [ 1  0 ]    My = [ 0  1 ]
#         [ 0 -1 ]         [-1  0 ]
#
#  Pikseļu apkārtne:  img[x,y] = [ a  b ]
#                                 [ c  d ]
#
#  Gx = a - d  (Mx · apkārtne)
#  Gy = b - c  (My · apkārtne)
#  G  = sqrt(Gx² + Gy²)
#
#  Ja G > slieksnis → mala (balts), pretējā gadījumā → fons (melns)

def roberts_edge_detection(img: np.ndarray, threshold: int = 60) -> np.ndarray:
    """
    Realizē Roberts malu detektoru manuāli, izmantojot tikai numpy
    masīvu indeksēšanu (nevis iebūvētās filtru metodes).
    """
    img_f = img.astype(np.float32)
    rows, cols = img_f.shape

    # Vektorizēta realizācija, kas atbilst pikseļu iterācijai:
    #   a = img[i,   j  ]
    #   b = img[i,   j+1]
    #   c = img[i+1, j  ]
    #   d = img[i+1, j+1]
    a = img_f[:-1, :-1]   # augšējais kreisais
    b = img_f[:-1,  1:]   # augšējais labais
    c = img_f[ 1:, :-1]   # apakšējais kreisais
    d = img_f[ 1:,  1:]   # apakšējais labais

    Gx = a - d             # Mx operatora rezultāts
    Gy = b - c             # My operatora rezultāts
    G  = np.sqrt(Gx**2 + Gy**2)

    # Sliekšņošana → binārā malu karte
    edges = np.zeros((rows, cols), dtype=np.uint8)
    edges[:-1, :-1] = np.where(G > threshold, 255, 0).astype(np.uint8)

    return edges


# ── 4. Attēlu ielāde ────────────────────────────────────────────────────

print("Ielādē attēlus...")

img1_color = load_image_color_from_url(URL_1)
img1       = np.array(Image.fromarray(img1_color).convert("L"), dtype=np.uint8)
print(f"  Attēls 1 ielādēts: {img1.shape}")

if URL_2.strip():
    img2_color = load_image_color_from_url(URL_2)
    img2       = np.array(Image.fromarray(img2_color).convert("L"), dtype=np.uint8)
    print(f"  Attēls 2 ielādēts no URL: {img2.shape}")
else:
    img2_color = add_gaussian_noise(img1_color, NOISE_AMOUNT)
    img2       = add_gaussian_noise(img1, NOISE_AMOUNT)
    print(f"  Attēls 2: trokšņains variants no Attēla 1 (Gausa σ={NOISE_AMOUNT})")

img3_color = load_image_color_from_url(URL_3)
img3       = np.array(Image.fromarray(img3_color).convert("L"), dtype=np.uint8)
print(f"  Attēls 3 ielādēts: {img3.shape}")

images        = [img1, img2, img3]
images_color  = [img1_color, img2_color, img3_color]
labels = [
    "Attēls 1\n(skaidrs objekts)",
    "Attēls 2\n(ar troksni)",
    "Attēls 3\n(brīva izvēle)",
]


# ── 5. Malu detektēšana ─────────────────────────────────────────────────

print("\nApstrādā attēlus...")

canny_results  = []
roberts_results = []

for i, img in enumerate(images):
    # Canny — iebūvēta funkcija (atļauts pēc uzdevuma nosacījumiem)
    canny = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
    canny_results.append(canny)

    # Roberts — manuāla realizācija (bez iebūvētām metodēm)
    roberts = roberts_edge_detection(img, ROBERTS_THRESHOLD)
    roberts_results.append(roberts)

    print(f"  Attēls {i+1}: Canny ✓  Roberts ✓")


# ── 6. Vizualizācija (Atskaite) ─────────────────────────────────────────
#  4 kolonnas: Krāsu oriģināls | Pelēktoņu oriģināls | Canny | Roberts

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    "Malu detektēšanas atskaite\n"
    f"Canny: low={CANNY_LOW}, high={CANNY_HIGH}  |  "
    f"Roberts: threshold={ROBERTS_THRESHOLD}",
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
    col_data = [color, gray, canny, roberts]
    for col, (data, ctitle) in enumerate(zip(col_data, col_titles)):
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.imshow(data)                        # krāsains — bez cmap
        else:
            ax.imshow(data, cmap="gray", vmin=0, vmax=255)
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
print("\nAtskaite saglabāta: atskaite.png")
plt.show()


# ── 7. Detalizēta salīdzinājuma tabula (papildus) ───────────────────────

fig2, axes = plt.subplots(2, 3, figsize=(16, 9))
fig2.suptitle("Canny vs Roberts — detalizēts salīdzinājums", fontsize=13)

method_rows = [("Canny", canny_results), ("Roberts", roberts_results)]

for r, (method_name, results) in enumerate(method_rows):
    for c, (result, lbl) in enumerate(zip(results, labels)):
        ax = axes[r, c]
        ax.imshow(result, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(f"{method_name}\n{lbl}", fontsize=9)

plt.tight_layout()
plt.savefig("salīdzinājums.png", dpi=150, bbox_inches="tight")
print("Salīdzinājums saglabāts: salīdzinājums.png")
plt.show()

print("\nVisi uzdevumi pabeigti!")
