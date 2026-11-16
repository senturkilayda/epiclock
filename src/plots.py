"""Medium yazısı için grafikler.

Tasarım kuralları:
- Başlık bulguyu söyler, alt başlık nasıl okunacağını açıklar.
- Tek y ekseni, ince işaretler, silik ızgara, renk yalnızca anlam taşıdığında.
- Kimlik hiçbir zaman yalnızca renkle verilmez (etiket, işaret şekli, metin).
- Sentetik veriyle üretilen her grafikte görünür bir uyarı etiketi bulunur.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------- renk ve mürekkep
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"        # ana seri
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#9ec5f4"
BLUE_PALE = "#cde2fb"
RED = "#e34948"         # ıraksak çiftin sıcak kutbu (yaşla artan)
NEUTRAL = "#f0efec"

FONT_STACK = ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 10.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
    })


def _frame(fig, title: str, subtitle: str, note: str, synthetic: bool) -> None:
    """Başlık, alt başlık, kaynak notu ve gerekiyorsa sentetik veri etiketi."""
    fig.text(0.04, 0.955, title, fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=10.5, color=INK_2, va="top", linespacing=1.4)
    fig.text(0.04, 0.025, note, fontsize=8.5, color=MUTED, va="bottom")
    if synthetic:
        fig.text(
            0.96, 0.025, "SENTETİK VERİ · YALNIZCA KOD TESTİ", fontsize=8,
            fontweight="bold", color=INK_2, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", fc=NEUTRAL, ec=AXIS, lw=0.8),
        )


def num(x: float, d: int = 1, sign: bool = False) -> str:
    """Türkçe sayı biçimi: ondalık virgül ve gerçek eksi işareti (−)."""
    txt = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return txt.replace(".", ",").replace("-", "−")


def possessive(n: int) -> str:
    """Sayıya iyelik eki: 2 -> "2'si", 11 -> "11'i", 40 -> "40'ı" (okunuşa göre)."""
    n = int(round(n))
    if n == 0:
        return "0'ı"
    if n % 100 == 0:
        return f"{n}'ü"
    tens = {10: "u", 20: "si", 30: "u", 40: "ı", 50: "si", 60: "ı", 70: "i", 80: "i", 90: "ı"}
    ones = {1: "i", 2: "si", 3: "ü", 4: "ü", 5: "i", 6: "sı", 7: "si", 8: "i", 9: "u"}
    suffix = ones[n % 10] if n % 10 else tens[n % 100]
    return f"{n}'{suffix}"


def possessive_text(number_text: str) -> str:
    """Ondalıklı sayılar için iyelik eki: "6,5" -> "6,5'i" (son okunan basamağa göre)."""
    if "," not in number_text:
        return possessive(int(number_text))
    last = int(number_text[-1])
    if last == 0:
        return possessive(int(number_text.split(",")[0]))
    suffix = possessive(last).split("'")[1]
    return f"{number_text}'{suffix}"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[grafik] {path.name}")
    return path


SOURCE = "Veri: Hannum ve ark. (2013), GEO GSE40279 · Analiz: epiclock"


# ---------------------------------------------------------------- 1. yaş dağılımı
def plot_age_distribution(age: np.ndarray, path: Path, synthetic: bool) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.76, bottom=0.14)

    bins = np.arange(15, 106, 5)
    counts, _, _ = ax.hist(age, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    med = np.median(age)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.axvline(med, color=INK_2, lw=1, ls=(0, (3, 3)), ymax=0.9)
    ax.text(med, counts.max() * 1.12, f"medyan {med:.0f}", color=INK_2, fontsize=9.5,
            ha="center", va="center")

    ax.set_xlabel("Kronolojik yaş (yıl)")
    ax.set_ylabel("Kişi sayısı")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(15, 105)

    _frame(fig,
           f"{len(age)} kişi, {age.min():.0f}–{age.max():.0f} yaş",
           "Kan örneklerinin yaş dağılımı. Gençler az olduğu için model bu yaşlarda\n"
           "daha az örnekten öğreniyor; bölmeleri bu yüzden yaşa göre tabakalandırıyoruz.",
           SOURCE, synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 2. tahmin vs gerçek
def plot_pred_vs_true(y: np.ndarray, pred: np.ndarray, model_label: str,
                      path: Path, synthetic: bool, split: str = "random") -> Path:
    mae = np.mean(np.abs(pred - y))
    r = np.corrcoef(y, pred)[0, 1]

    fig, ax = plt.subplots(figsize=(10, 7.2))
    fig.subplots_adjust(left=0.10, right=0.70, top=0.80, bottom=0.11)

    lo, hi = 10, 105
    grid = np.array([lo, hi])
    ax.fill_between(grid, grid - mae, grid + mae, color=BLUE_PALE, alpha=0.7, lw=0, zorder=1)
    ax.plot(grid, grid, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.scatter(y, pred, s=34, color=BLUE, alpha=0.8, edgecolor=SURFACE, linewidth=0.8, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Kronolojik yaş (yıl)")
    ax.set_ylabel("DNA metilasyonundan tahmin edilen yaş (yıl)")

    ax.text(hi - 2, hi - 4, "tahmin = gerçek", color=MUTED, fontsize=9, ha="right", rotation=45,
            rotation_mode="anchor", transform_rotates_text=True)

    # Sağ kenarda okuma rehberi ve metrikler
    fig.text(0.73, 0.74, f"{num(mae)} yıl", fontsize=30, fontweight="bold", color=INK, va="top")
    fig.text(0.73, 0.655, "ortalama mutlak hata (MAE)", fontsize=10, color=INK_2, va="top")
    fig.text(0.73, 0.58, f"r = {num(r, 2)}", fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.73, 0.535, "Pearson korelasyonu", fontsize=10, color=INK_2, va="top")
    fig.text(0.73, 0.44,
             f"Her nokta test verisindeki\nbir kişi (n = {len(y)}).\n\n"
             "Açık mavi bant: kesikli\nçizginin ± MAE kadar çevresi.\n\n"
             "Çizginin üstündeki noktalar\nyaşından büyük, altındakiler\nküçük tahmin edilenler.",
             fontsize=9.5, color=INK_2, va="top", linespacing=1.45)

    _frame(fig,
           f"{model_label}: görmediği kişilerde ortalama {num(mae)} yıl hata",
           ("Test verisi modelin eğitiminde, CpG seçiminde ve hiperparametre seçiminde hiç kullanılmadı."
            + ("\nTest kişileri, eğitimde hiç görülmeyen ölçüm plakalarından geliyor." if split == "plate" else "")),
           SOURCE, synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 3. model karşılaştırması
def plot_model_comparison(fold_metrics: pd.DataFrame, test_metrics: pd.DataFrame,
                          labels: dict, path: Path, synthetic: bool, split: str = "random") -> Path:
    cv_mean = fold_metrics.groupby("model")["MAE"].mean().sort_values(ascending=False)
    order = list(cv_mean.index)  # en kötü altta, en iyi üstte
    if "baseline" in order:      # taban çizgisini her zaman en alta koy
        order.remove("baseline")
        order = ["baseline"] + order
    test = test_metrics.set_index("model")["MAE"]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=0.24, right=0.95, top=0.74, bottom=0.20)

    for i, m in enumerate(order):
        color = MUTED if m == "baseline" else BLUE
        folds = fold_metrics.loc[fold_metrics["model"] == m, "MAE"].to_numpy()
        ax.plot([folds.min(), folds.max()], [i, i], color=color, lw=2, alpha=0.35, solid_capstyle="round")
        ax.scatter(folds, np.full_like(folds, i), s=26, facecolor=SURFACE, edgecolor=color, lw=1.2, zorder=3)
        ax.scatter(cv_mean[m], i, s=110, color=color, edgecolor=SURFACE, lw=1.5, zorder=4)
        ax.scatter(test[m], i, s=80, marker="D", facecolor=SURFACE, edgecolor=INK, lw=1.4, zorder=5)
        ax.text(max(folds.max(), test[m]) + 0.25, i, f"{num(cv_mean[m], 2)} yıl",
                va="center", fontsize=10, color=INK, fontweight="bold" if m != "baseline" else "normal")

    ax.set_yticks(range(len(order)), [labels.get(m, m) for m in order])
    ax.tick_params(axis="y", labelsize=11, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Ortalama mutlak hata (yıl) · küçük olan daha iyi")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=BLUE, mew=1.2, label="Tek bir CV fold'u"),
        Line2D([], [], marker="o", ls="", ms=10, mfc=BLUE, mec=SURFACE, label="CV ortalaması (etiketli değer)"),
        Line2D([], [], marker="D", ls="", ms=8, mfc=SURFACE, mec=INK, mew=1.4, label="Dokunulmamış test verisi"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3)

    ranked = [m for m in cv_mean.sort_values().index if m != "baseline"]
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    base = cv_mean.get("baseline", np.nan)
    # Fark fold'lar arası oynamadan küçükse "en iyi" demek fazla iddialı olur.
    fold_sd = fold_metrics.loc[fold_metrics["model"] == best, "MAE"].std()
    if second is not None and cv_mean[second] - cv_mean[best] < fold_sd:
        title = f"{labels[best]} ve {labels[second]} aynı düzeyde"
    else:
        title = f"{labels[best]} en düşük hatayı veriyor"
    _frame(fig,
           title,
           f"Herkese ortalama yaşı yazan taban çizgisi {num(base)} yıl hata yapıyor. "
           f"{fold_metrics['fold'].nunique()} katlı "
           + ("plakaya göre gruplanmış " if split == "plate" else "")
           + "çapraz doğrulamada CpG seçimi\nve hiperparametreler her fold'da yalnızca o fold'un "
           "eğitim kısmından öğrenildi.",
           SOURCE, synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 4. yaşa bağlı sapma
def _binned_mean(x, y, edges):
    idx = np.digitize(x, edges) - 1
    centers, means = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 5:
            centers.append(x[sel].mean())
            means.append(y[sel].mean())
    return np.array(centers), np.array(means)


def plot_residual_vs_age(y: np.ndarray, pred: np.ndarray, model_label: str,
                         path: Path, synthetic: bool) -> Path:
    resid = pred - y
    slope = np.polyfit(y, resid, 1)[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.74, bottom=0.13)

    ax.axhline(0, color=AXIS, lw=1.2)
    ax.scatter(y, resid, s=22, color=BLUE_LIGHT, alpha=0.75, edgecolor=SURFACE, lw=0.6, zorder=2)
    cx, cy = _binned_mean(y, resid, np.arange(15, 111, 10))
    ax.plot(cx, cy, color=BLUE_DARK, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
    ax.text(cx[-1] + 1.5, cy[-1], "10 yıllık\ndilim ortalaması", color=INK_2, fontsize=9, va="center")

    lim = np.nanmax(np.abs(resid)) * 1.1
    ax.set_ylim(-lim, lim)
    ax.set_xlim(15, 112)
    ax.text(16, lim * 0.92, "yaşından büyük tahmin", color=MUTED, fontsize=9, va="top")
    ax.text(16, -lim * 0.92, "yaşından küçük tahmin", color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel("Kronolojik yaş (yıl)")
    ax.set_ylabel("Tahmin − gerçek yaş (yıl)")

    if slope < -0.03:
        title = "Model gençleri yaşlı, yaşlıları genç tahmin etme eğiliminde"
    elif slope > 0.03:
        title = "Model gençleri genç, yaşlıları yaşlı tahmin etme eğiliminde"
    else:
        title = "Tahmin hatası yaşa göre belirgin bir yön göstermiyor"
    _frame(fig, title,
           f"{model_label}, fold dışı tahminler (n = {len(y)}). Eğim: {num(slope, 2, sign=True)} yıl/yıl. Bu sapma varsa "
           "'yaş ivmesi'\nham farkla değil, tahmin edilen yaşın kronolojik yaşa göre artığıyla hesaplanmalı.",
           SOURCE, synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 5. en etkili CpG'ler
def plot_top_cpgs(X: np.ndarray, age: np.ndarray, cpg_ids: list[str], coefs: pd.Series,
                  path: Path, synthetic: bool, n: int = 6) -> Path:
    top = coefs.head(n)
    col_of = {c: i for i, c in enumerate(cpg_ids)}

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.71, bottom=0.11, hspace=0.48, wspace=0.28)

    for ax, (cpg, coef) in zip(axes.flat, top.items()):
        beta = X[:, col_of[cpg]]
        ok = ~np.isnan(beta)
        # Yön, model katsayısından değil verinin kendisinden: çok değişkenli modelde
        # katsayının işareti, CpG'nin yaşla tek başına ilişkisinin tersi olabilir.
        rho = stats.spearmanr(age[ok], beta[ok])[0]
        color = RED if rho > 0 else BLUE
        direction = f"yaşla artıyor ↑ (ρ = {num(rho, 2)})" if rho > 0 else f"yaşla azalıyor ↓ (ρ = {num(rho, 2)})"
        if np.sign(rho) != np.sign(coef):
            direction += " · katsayı ters"
        ax.scatter(age[ok], beta[ok], s=10, color=color, alpha=0.35, lw=0)
        cx, cy = _binned_mean(age[ok], beta[ok], np.arange(15, 111, 10))
        ax.plot(cx, cy, color=INK, lw=2)
        ax.set_title(cpg, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=16)
        ax.text(0, 1.03, direction, transform=ax.transAxes, fontsize=9, color=INK_2)
        ax.set_ylim(max(0, np.nanmin(beta) - 0.05), min(1, np.nanmax(beta) + 0.05))

    for ax in axes[1]:
        ax.set_xlabel("Yaş (yıl)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Metilasyon (beta)")

    _frame(fig,
           f"Yaş tahmininde en çok ağırlık taşıyan {n} CpG",
           "Elastic Net katsayısının mutlak değerine göre sıralandı. Noktalar eğitim verisindeki kişiler,\n"
           "siyah çizgi 10 yıllık dilim ortalaması. Yön ve ρ (Spearman) doğrudan veriden hesaplandı.",
           SOURCE, synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 6. yazıdaki varsayımsal örnek
def plot_hypothetical_shift(resid: np.ndarray, path: Path, synthetic: bool,
                            chrono: float = 60, before: float = 58, after: float = 52) -> Path:
    r_before, r_after = before - chrono, after - chrono
    share = np.mean(resid <= r_after) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.72, bottom=0.14)

    lim = max(np.ceil(np.abs(resid).max()), abs(r_after) + 4)
    edges = np.arange(-lim, lim + 1, 1)
    counts, _ = np.histogram(resid, bins=edges)
    colors = [BLUE if e + 1 <= r_after else BLUE_LIGHT for e in edges[:-1]]
    ax.bar(edges[:-1] + 0.5, counts, width=1, color=colors, edgecolor=SURFACE, linewidth=1.5)

    top = counts.max()
    for x, lab, ha, dx in [
        (r_before, f"Müdahale öncesi\ntahmin {before:.0f} ({num(r_before, 0, sign=True)})", "left", 0.4),
        (r_after, f"Müdahale sonrası\ntahmin {after:.0f} ({num(r_after, 0, sign=True)})", "right", -0.4),
    ]:
        ax.axvline(x, color=INK, lw=1.2, ymax=0.9)
        ax.text(x + dx, top * 1.3, lab, ha=ha, va="center", fontsize=9.5, color=INK)
    ax.annotate("", xy=(r_after + 0.2, top * 1.12), xytext=(r_before - 0.2, top * 1.12),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2))
    ax.set_ylim(0, top * 1.55)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(f"Tahmin − gerçek yaş (yıl) · varsayımsal kişinin gerçek yaşı {chrono:.0f}")
    ax.set_ylabel("Kişi sayısı")

    ax.text(r_after - 0.6, top * 0.55, f"Kişilerin %{possessive_text(num(share))}\nhiçbir müdahale\nolmadan bu bölgede",
            ha="right", fontsize=10, color=INK, fontweight="bold")

    title = f"Tahmini gerçek yaşından en az {abs(r_after):.0f} yıl genç çıkan kişilerin oranı %{num(share)}"
    _frame(fig,
           title,
           "Koyu çubuklar, tahmini gerçek yaşından en az bu kadar genç çıkan kişiler (fold dışı tahminler).\n"
           "Dağılım kişiler arası farkı gösterir; aynı kişide tekrarlanan ölçümün gürültüsünü göstermez.",
           SOURCE + " · Varsayımsal kişi: yazıdaki 60 → 58 → 52 örneği", synthetic)
    return _save(fig, path)


# ---------------------------------------------------------------- 7. plaka (parti) etkisi
def p_text(p: float) -> str:
    """p değerini okunur yazar: çok küçükse 'p < 0,001'."""
    return "p < 0,001" if p < 0.001 else f"p = {num(p, 3)}"


def plot_plate_effect(resid: np.ndarray, plate: pd.Series, path: Path, synthetic: bool,
                      age: np.ndarray | None = None) -> Path:
    """Plakalara göre tahmin hatası.

    Plakaların yaş dağılımı farklıysa, yaşa bağlı sapma (Görsel 4) tek başına
    plaka farkı gibi görünebilir. ``age`` verilirse bu sapma önce çıkarılır:
    hatanın yaşa göre doğrusal eğilimi bulunur ve her kişinin hatasından düşülür.
    """
    plate = plate.astype(str).to_numpy()
    raw = resid.copy()
    adjusted = age is not None
    if adjusted:
        slope, intercept = np.polyfit(age, resid, 1)
        resid = resid - (slope * age + intercept)

    df = pd.DataFrame({"resid": resid, "raw": raw, "plate": plate})
    if adjusted:
        df["age"] = age
    try:
        order = sorted(df["plate"].unique(), key=lambda p: float(p))
    except ValueError:
        order = sorted(df["plate"].unique())
    groups = [df.loc[df["plate"] == p, "resid"].to_numpy() for p in order]
    h, p_val = stats.kruskal(*groups)
    h_raw, p_raw = stats.kruskal(*[df.loc[df["plate"] == p, "raw"].to_numpy() for p in order])
    medians = [np.median(g) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.72, bottom=0.19)
    rng = np.random.default_rng(0)
    ax.axhline(0, color=AXIS, lw=1.2)
    for i, (g, med) in enumerate(zip(groups, medians)):
        jitter = rng.uniform(-0.22, 0.22, len(g))
        ax.scatter(i + jitter, g, s=16, color=BLUE_LIGHT, alpha=0.8, lw=0)
        ax.plot([i - 0.3, i + 0.3], [med] * 2, color=BLUE_DARK, lw=2.5, solid_capstyle="round")
        ax.text(i + 0.34, med, num(med, 1, sign=True), va="center", fontsize=8.5, color=INK)

    labels = []
    for p_ in order:
        sub = df[df["plate"] == p_]
        lab = f"{p_}\nn = {len(sub)}"
        if adjusted:
            lab += f"\nyaş {sub['age'].median():.0f}"
        labels.append(lab)
    ax.set_xticks(range(len(order)), labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Ölçüm plakası (kişi sayısı, medyan yaş)" if adjusted else "Ölçüm plakası")
    ax.set_ylabel("Yaşa göre düzeltilmiş hata (yıl)" if adjusted else "Tahmin − gerçek yaş (yıl)")

    spread = max(medians) - min(medians)
    if p_val < 0.05:
        verdict = f"Plaka medyanları arasında {num(spread)} yıllık fark var"
    else:
        verdict = "Plakalar arasında belirgin bir hata farkı görülmüyor"
    if adjusted:
        sub = (f"Koyu çizgi: plaka medyanı. Yaşa bağlı sapma çıkarıldıktan sonra Kruskal–Wallis "
               f"H = {num(h)}, {p_text(p_val)}\n(düzeltmeden önce H = {num(h_raw)}). "
               "Kalan fark yaş dağılımıyla açıklanamıyor; nedeni bu veriyle belirlenemiyor.")
    else:
        sub = (f"Koyu çizgi: plaka medyanı. Kruskal–Wallis H = {num(h)}, {p_text(p_val)}. "
               "Fark varsa, plakalara göre\ngruplanmış çapraz doğrulama (GroupKFold) daha dürüst "
               "bir hata tahmini verir.")
    _frame(fig, verdict, sub, SOURCE, synthetic)
    return _save(fig, path)
