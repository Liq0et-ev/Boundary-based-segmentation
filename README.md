# Boundary-Based Segmentation — Malu detektēšana

Uz robežām balstītas segmentācijas Python programma Google Colab videi.  
Realizē **Canny** un **Roberts** malu detektēšanas algoritmus uz trim attēliem.

---

## Saturs

- [Uzdevuma apraksts](#uzdevuma-apraksts)
- [Izmantotās metodes](#izmantotās-metodes)
  - [Sāls un piparu troksnis](#sāls-un-piparu-troksnis)
  - [Canny malu detektors](#canny-malu-detektors)
  - [Roberts operators](#roberts-operators)
- [Koda struktūra](#koda-struktūra)
- [Izmantotās funkcijas](#izmantotās-funkcijas)
- [Palaišana Google Colab](#palaišana-google-colab)

---

## Uzdevuma apraksts

Programma apstrādā **3 attēlus**:

| # | Attēls | Apraksts |
|---|--------|----------|
| 1 | Oriģināls | Attēls ar skaidri redzamu objektu |
| 2 | Trokšņains | Tas pats attēls ar pievienotu sāls & piparu troksni |
| 3 | Brīvs | Jebkurš programmētāja izvēlēts attēls |

Uz visiem attēliem tiek pielietoti divi malu detektēšanas algoritmi:
- **Canny** — ar iebūvētu `cv2.Canny()` funkciju
- **Roberts** — manuāla realizācija **bez** iebūvētām bibliotēku metodēm

Rezultāts — atskaite ar 4 kolonnām katram attēlam:  
`Krāsains oriģināls | Pelēktoņu oriģināls | Canny | Roberts`

---

## Izmantotās metodes

### Sāls un piparu troksnis

**Salt & Pepper** troksnis simulē sensoru bojājumus — daļa pikseļu nejauši kļūst pilnīgi balti (255) vai pilnīgi melni (0).

```
Bojāto pikseļu skaits = kopējais pikseļu skaits × amount
  → puse = sāls  (vērtība 255)
  → puse = pipari (vērtība 0)
```

Parametrs `NOISE_AMOUNT = 0.05` nozīmē 5% bojātu pikseļu.

---

### Canny malu detektors

Daudzpakāpju algoritms precīzai malu noteikšanai:

1. **Gausa filtrēšana** — attēla izlīdzināšana trokšņa mazināšanai
2. **Gradienta aprēķins** — Sobel operatori horizontālajam un vertikālajam gradientam:
   ```
   G  = √(Gx² + Gy²)
   Θ  = atan2(Gy, Gx)
   ```
3. **Lokālo maksimumu atlase** *(Non-Maximum Suppression)* — viltus pikseļu noņemšana
4. **Histerēzes sliekšņošana** — divi sliekšņi malu apstiprināšanai:
   - `CANNY_LOW  = 50`  — zemākais slieksnis
   - `CANNY_HIGH = 150` — augstākais slieksnis

Pikseļi virs augstākā sliekšņa → stipras malas.  
Pikseļi starp sliekšņiem → vājas malas (iekļauj tikai ja savienotas ar stipru malu).  
Pikseļi zem zemākā sliekšņa → nav malas.

---

### Roberts operators

Ātrais gradienta operators 2×2 kodoliem. Piemērots maziem, asiem objektiem.

**Kodoli:**

```
Mx = [ 1  0 ]     My = [ 0  1 ]
     [ 0 -1 ]          [-1  0 ]
```

**Pikseļu apkārtne:**

```
img[x,y] = [ a  b ]
            [ c  d ]
```

**Gradienta aprēķins:**

```
Gx = a - d          (Mx operatora rezultāts)
Gy = b - c          (My operatora rezultāts)
G  = √(Gx² + Gy²)
```

**Sliekšņošana:**

```
Ja G > ROBERTS_THRESHOLD (60) → mala (255, balts)
Pretējā gadījumā             → fons  (0,   melns)
```

> **Svarīgi:** Roberts operators realizēts **manuāli** tikai ar NumPy masīvu indeksēšanu — bez `cv2.filter2D`, `scipy.ndimage` vai citām iebūvētām filtru metodēm.

---

## Koda struktūra

```
edge_detection_colab.py
│
├── UZSTĀDĪJUMI (faila sākums)
│   ├── URL_1, URL_2, URL_3      # Attēlu saites
│   ├── ROBERTS_THRESHOLD        # Roberts sliekšņa vērtība
│   ├── CANNY_LOW, CANNY_HIGH    # Canny sliekšņi
│   └── NOISE_AMOUNT             # Trokšņa intensitāte
│
├── 1. Bibliotēkas
├── 2. Palīgfunkcijas
│   ├── load_image_from_url()        # Pelēktoņu ielāde
│   ├── load_image_color_from_url()  # RGB ielāde
│   └── add_salt_pepper_noise()      # Trokšņa pievienošana
│
├── 3. Roberts operators
│   └── roberts_edge_detection()     # Manuāla realizācija
│
├── 4. Attēlu ielāde
├── 5. Malu detektēšana (Canny + Roberts)
├── 6. Atskaite (4×3 attēlu režģis)
└── 7. Salīdzinājuma tabula (Canny vs Roberts)
```

---

## Izmantotās funkcijas

| Funkcija | Bibliotēka | Apraksts |
|----------|-----------|----------|
| `load_image_from_url(url)` | `urllib`, `PIL` | Ielādē attēlu no URL pelēktoņos |
| `load_image_color_from_url(url)` | `urllib`, `PIL` | Ielādē attēlu no URL RGB formātā |
| `add_salt_pepper_noise(img, amount)` | `numpy` | Pievieno sāls & piparu troksni |
| `roberts_edge_detection(img, threshold)` | `numpy` | **Manuāls** Roberts operators |
| `cv2.Canny(img, low, high)` | `opencv-python` | Canny malu detektors |
| `Image.open().convert()` | `PIL` | Attēla formāta konvertēšana |
| `plt.imshow()` | `matplotlib` | Attēlu vizualizācija |
| `plt.savefig()` | `matplotlib` | Atskaites saglabāšana |
| `np.sqrt()`, `np.where()` | `numpy` | Gradienta aprēķins Roberts metodē |
| `np.random.randint()` | `numpy` | Nejauša pikseļu atlase trokšņa ģenerēšanai |

---

## Palaišana Google Colab

1. Atveriet [colab.research.google.com](https://colab.research.google.com)
2. Izveidojiet jaunu notebook (`File → New notebook`)
3. Iekopējiet `edge_detection_colab.py` saturu šūnā
4. Aizstājiet URL saites faila sākumā:

```python
URL_1 = "https://..."   # Attēls ar skaidri redzamu objektu
URL_2 = ""              # Tukšs → automātisks troksnis no URL_1
URL_3 = "https://..."   # Brīvs attēls
```

5. Palaidiet šūnu — tiks saglabāti `atskaite.png` un `salīdzinājums.png`

### Nepieciešamās bibliotēkas

Google Colab vidē visas bibliotēkas ir pieejamas pēc noklusējuma:

```
opencv-python  (cv2)
numpy
matplotlib
Pillow         (PIL)
```

---

## Atskaites rezultāts

Programma ģenerē divus attēlu failus:

- **`atskaite.png`** — pilna atskaite ar visiem attēliem (4 kolonnas × 3 rindas)
- **`salīdzinājums.png`** — Canny vs Roberts salīdzinājums (2 rindas × 3 kolonnas)
