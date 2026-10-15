#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VII. Cài đặt thuật toán K-means bằng Python

    1. Đọc dữ liệu
    2. Lựa chọn thuộc tính
    3. Đánh giá số cụm bằng Elbow Method và Silhouette
    4. Huấn luyện mô hình
    5. Chạy thử, tính SSE, Silhouette trung bình và từng khách hàng

Chạy:  python3 kmeans_hoan_chinh.py
Cần:   python3 -m pip install numpy pandas matplotlib scikit-learn

Đầu vào là CSV đã xử lý. Chương trình không tự chuẩn hóa dữ liệu.
Silhouette dùng khoảng cách Euclid giữa các điểm, không dùng khoảng cách
đến tâm cụm. Tính bằng sklearn.metrics trên toàn bộ dữ liệu, không lấy mẫu
ngẫu nhiên. Thuật toán K-means được viết trực tiếp bằng NumPy.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh, không cần cửa sổ đồ họa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples

CSV_MAC_DINH = "khach_hang_train70.csv"
THU_MUC_KET_QUA = Path("ket_qua_kmeans")


# ----------------------------------------------------------------------
# Hàm nhập liệu tiện ích
# ----------------------------------------------------------------------
def nhap_so_nguyen(thong_bao, toi_thieu, mac_dinh=None):
    while True:
        s = input(thong_bao).strip()
        if s == "" and mac_dinh is not None:
            return mac_dinh
        try:
            gia_tri = int(s)
            if gia_tri >= toi_thieu:
                return gia_tri
        except ValueError:
            pass
        print(f"  Vui lòng nhập số nguyên >= {toi_thieu}.")


# ----------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------
def doc_du_lieu():
    print("=== 1. ĐỌC DỮ LIỆU ===")
    duong_dan = input(f"Đường dẫn file CSV (Enter = '{CSV_MAC_DINH}'): ").strip()
    duong_dan = Path(duong_dan or CSV_MAC_DINH).expanduser()
    if not duong_dan.is_file():
        raise SystemExit(f"Không tìm thấy file: {duong_dan}")

    du_lieu = pd.read_csv(duong_dan, encoding="utf-8-sig")
    print(f"Đã đọc {du_lieu.shape[0]} dòng, {du_lieu.shape[1]} cột.")
    print(du_lieu.head().to_string(index=False))

    if du_lieu.isna().any().any():
        raise SystemExit("Dữ liệu còn ô thiếu (NaN). Hãy xử lý trước khi phân cụm.")
    return du_lieu


# ----------------------------------------------------------------------
# 2. Lựa chọn thuộc tính
# ----------------------------------------------------------------------
def chon_thuoc_tinh(du_lieu):
    print("\n=== 2. LỰA CHỌN THUỘC TÍNH ===")
    cot_so = [c for c in du_lieu.columns
              if pd.api.types.is_numeric_dtype(du_lieu[c])
              and not pd.api.types.is_bool_dtype(du_lieu[c])]
    if not cot_so:
        raise SystemExit("Không có cột số nào để phân cụm.")

    for i, ten in enumerate(cot_so, start=1):
        print(f"  {i}. {ten}")
    s = input("Nhập số thứ tự các thuộc tính cần dùng, cách nhau bởi dấu phẩy "
              "(Enter = dùng tất cả): ").strip()

    if s == "":
        da_chon = cot_so
    else:
        try:
            chi_so = sorted({int(p) for p in s.split(",")})
            da_chon = [cot_so[i - 1] for i in chi_so if 1 <= i <= len(cot_so)]
        except ValueError:
            raise SystemExit("Danh sách thuộc tính không hợp lệ.")
        if not da_chon:
            raise SystemExit("Chưa chọn thuộc tính hợp lệ nào.")

    print("Thuộc tính được dùng:", ", ".join(da_chon))
    X = du_lieu[da_chon].to_numpy(dtype=float)
    if len(X) == 0:
        raise SystemExit("Tập dữ liệu không có dòng nào.")
    if not np.isfinite(X).all():
        raise SystemExit("Các thuộc tính được chọn phải có giá trị số hữu hạn.")
    return da_chon, X


# ----------------------------------------------------------------------
# Thuật toán K-means (viết trực tiếp)
# ----------------------------------------------------------------------
def khoang_cach(X, tam):
    """Ma trận khoảng cách Euclid, kích thước n x K."""
    return np.sqrt(((X[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2))


def kmeans(X, k, max_iter, seed):
    rng = np.random.default_rng(seed)

    # Bước 1: chọn ngẫu nhiên K điểm dữ liệu làm tâm ban đầu
    chi_so = rng.choice(len(X), size=k, replace=False)
    tam = X[chi_so].copy()
    tam_ban_dau = tam.copy()
    nhan = None
    so_vong = 0

    for so_vong in range(1, max_iter + 1):
        # Bước 2: gán mỗi điểm vào tâm gần nhất
        nhan_moi = khoang_cach(X, tam).argmin(axis=1)

        # Điều kiện dừng: nhãn không còn thay đổi
        if nhan is not None and np.array_equal(nhan, nhan_moi):
            break
        nhan = nhan_moi

        # Bước 3: tâm mới = trung bình các điểm của cụm (cụm rỗng giữ tâm cũ)
        for j in range(k):
            if np.any(nhan == j):
                tam[j] = X[nhan == j].mean(axis=0)

    # Gán lại theo tâm cuối để nhãn, SSE và tâm luôn khớp nhau
    kc = khoang_cach(X, tam)
    nhan = kc.argmin(axis=1)
    sse = float((kc[np.arange(len(X)), nhan] ** 2).sum())
    return {"tam": tam, "tam_ban_dau": tam_ban_dau, "nhan": nhan,
            "sse": sse, "so_vong": so_vong}


# ----------------------------------------------------------------------
# Silhouette bằng scikit-learn
# ----------------------------------------------------------------------
def tinh_silhouette(X, nhan, tinh_tung_diem=True):
    """Tính Silhouette trung bình và, nếu cần, điểm từng khách hàng.

    silhouette_score: điểm trung bình toàn bộ dữ liệu để so sánh K.
    silhouette_samples: điểm từng khách hàng theo đúng thứ tự dòng của X.
    Chỉ xác định khi có từ 2 đến N-1 cụm có dữ liệu.
    Dùng chính X và nhãn do K-means tạo ra, với khoảng cách Euclid.
    """
    X = np.asarray(X, dtype=float)
    nhan = np.asarray(nhan)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X phải là ma trận số hữu hạn có ít nhất một thuộc tính.")
    if nhan.ndim != 1 or len(nhan) != len(X):
        raise ValueError("Nhãn phải là một mảng có đúng N phần tử.")
    n, so_cum = len(X), len(np.unique(nhan))
    if not 2 <= so_cum <= n - 1:
        return {"trung_binh": None, "tung_diem": None,
                "ly_do": "Cần từ 2 đến N-1 cụm có dữ liệu; "
                          f"hiện có {so_cum} cụm và N = {n}."}

    trung_binh = float(silhouette_score(X, nhan, metric="euclidean"))
    tung_diem = (silhouette_samples(X, nhan, metric="euclidean")
                 if tinh_tung_diem else None)
    return {"trung_binh": trung_binh, "tung_diem": tung_diem, "ly_do": None}


# ----------------------------------------------------------------------
# 3. Đánh giá số cụm bằng Elbow Method và Silhouette
# ----------------------------------------------------------------------
def goi_y_khuyu_tay(ks, sse):
    """Điểm 'khuỷu tay': điểm xa nhất so với đoạn thẳng nối đầu - cuối đường SSE."""
    if len(ks) < 3 or sse[0] == sse[-1]:
        return ks[0]
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(sse) - sse[-1]) / (sse[0] - sse[-1])
    # đoạn thẳng nối (0,1) -> (1,0): x + y - 1 = 0
    khoang = np.abs(x + y - 1) / np.sqrt(2)
    return ks[int(khoang.argmax())]


def elbow(X, max_iter, so_lan_chay=10):
    print("\n=== 3. ĐÁNH GIÁ K BẰNG ELBOW VÀ SILHOUETTE ===")
    k_max = nhap_so_nguyen("Thử K từ 1 đến (Enter = 10): ", 2, mac_dinh=10)
    k_max = min(k_max, len(X))

    print(f"Mỗi K chạy {so_lan_chay} lần với seed từ 0 đến {so_lan_chay - 1}.")
    print("Giữ mô hình có SSE nhỏ nhất; Silhouette tính trên chính mô hình đó.")
    print("K = 1 hoặc số cụm thực tế không hợp lệ: Silhouette = N/A.")
    ks, sse_list, danh_gia = [], [], []
    for k in range(1, k_max + 1):
        # Mỗi K chạy nhiều lần với seed khác nhau, lấy SSE nhỏ nhất cho ổn định
        cac_lan_chay = (dict(kmeans(X, k, max_iter, seed), seed=seed)
                       for seed in range(so_lan_chay))
        tot_nhat = min(cac_lan_chay, key=lambda ket_qua: ket_qua["sse"])
        sil = tinh_silhouette(X, tot_nhat["nhan"],
                             tinh_tung_diem=False)["trung_binh"]
        ks.append(k)
        sse_list.append(tot_nhat["sse"])
        danh_gia.append({"K": k, "SSE": tot_nhat["sse"],
                         "Silhouette": sil, "Seed": tot_nhat["seed"],
                         "SoCumThucTe": len(np.unique(tot_nhat["nhan"]))})
        chuoi_sil = "N/A" if sil is None else f"{sil:.6f}"
        print(f"  K = {k:<2}  SSE = {tot_nhat['sse']:.4f}  "
              f"Silhouette = {chuoi_sil}  seed = {tot_nhat['seed']}")

    k_goi_y = goi_y_khuyu_tay(ks, sse_list)

    THU_MUC_KET_QUA.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, sse_list, "o-")
    plt.axvline(k_goi_y, color="red", linestyle="--", label=f"Gợi ý K = {k_goi_y}")
    plt.xticks(ks)
    plt.xlabel("Số cụm K")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    anh = THU_MUC_KET_QUA / "elbow.png"
    plt.savefig(anh, dpi=150)
    plt.close()

    pd.DataFrame(danh_gia).to_csv(
        THU_MUC_KET_QUA / "elbow.csv", index=False)
    print(f"Gợi ý K theo điểm khuỷu tay: {k_goi_y}")
    print(f"Đã lưu biểu đồ: {anh}")

    hop_le = [dong for dong in danh_gia if dong["Silhouette"] is not None]
    if hop_le:
        tot_nhat_sil = max(hop_le, key=lambda dong: dong["Silhouette"])
        plt.figure(figsize=(7, 4.5))
        plt.plot([dong["K"] for dong in hop_le],
                 [dong["Silhouette"] for dong in hop_le], "o-")
        plt.axvline(tot_nhat_sil["K"], color="green", linestyle="--",
                    label=f"Silhouette cao nhất: K = {tot_nhat_sil['K']}")
        plt.xticks([dong["K"] for dong in hop_le])
        plt.xlabel("Số cụm K")
        plt.ylabel("Silhouette trung bình")
        plt.title("Đánh giá K bằng Silhouette")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(THU_MUC_KET_QUA / "silhouette_theo_k.png", dpi=150)
        plt.close()
        print("K có Silhouette cao nhất trong các K đã thử: "
              f"{tot_nhat_sil['K']} (Silhouette = {tot_nhat_sil['Silhouette']:.6f}).")
    print(f"Bảng SSE, Silhouette và seed: {THU_MUC_KET_QUA / 'elbow.csv'}")
    print("Silhouette càng cao càng tốt theo tiêu chí này; "
          "kết hợp Elbow và ý nghĩa các nhóm khách hàng để chọn K.")
    return k_goi_y


# ----------------------------------------------------------------------
# 4. Huấn luyện mô hình
# ----------------------------------------------------------------------
def huan_luyen(X, k_goi_y):
    print("\n=== 4. HUẤN LUYỆN MÔ HÌNH ===")
    k = nhap_so_nguyen(f"Nhập số cụm K (Enter = {k_goi_y}): ", 1, mac_dinh=k_goi_y)
    if k > len(X):
        raise SystemExit("K không được lớn hơn số dòng dữ liệu.")
    max_iter = nhap_so_nguyen("Nhập số vòng lặp tối đa (Enter = 300): ", 1, mac_dinh=300)
    seed = nhap_so_nguyen("Nhập seed (Enter = 42): ", 0, mac_dinh=42)

    ket_qua = kmeans(X, k, max_iter, seed)
    ket_qua["silhouette"] = tinh_silhouette(X, ket_qua["nhan"])
    ket_qua.update({"k": k, "max_iter": max_iter, "seed": seed})
    return ket_qua


# ----------------------------------------------------------------------
# 5. Chạy thử: hiển thị và lưu kết quả
# ----------------------------------------------------------------------
def chay_thu(du_lieu, ten_cot, X, kq):
    print("\n=== 5. CHẠY THỬ ===")
    k, nhan = kq["k"], kq["nhan"]
    dem = np.bincount(nhan, minlength=k)

    print(f"K = {k}; seed = {kq['seed']}; số vòng tối đa = {kq['max_iter']}")
    print(f"Số vòng lặp đã chạy: {kq['so_vong']}")
    print(f"SSE (Within cluster sum of squared errors): {kq['sse']:.6f}")
    sil = kq["silhouette"]
    if sil["trung_binh"] is None:
        print(f"Silhouette: không xác định. {sil['ly_do']}")
    else:
        print(f"Silhouette trung bình: {sil['trung_binh']:.6f}")

    print("\nTâm các cụm:")
    bang = pd.DataFrame(kq["tam"].T, index=ten_cot,
                        columns=[f"Cluster {j}" for j in range(k)])
    bang.insert(0, "Full Data", X.mean(axis=0))
    print(bang.round(4).to_string())

    print("\nSố phần tử mỗi cụm:")
    for j in range(k):
        print(f"  Cluster {j}: {dem[j]} ({100 * dem[j] / len(X):.1f}%)")

    THU_MUC_KET_QUA.mkdir(exist_ok=True)
    bang.to_csv(THU_MUC_KET_QUA / "tam_cum.csv", encoding="utf-8-sig")
    gan_cum = du_lieu.copy()
    gan_cum["Cluster"] = nhan
    # Giữ đúng thứ tự khách hàng; ô trống nghĩa là không xác định (K = 1).
    gan_cum["Silhouette"] = (sil["tung_diem"] if sil["tung_diem"] is not None
                             else np.nan)
    gan_cum.to_csv(THU_MUC_KET_QUA / "khach_hang_gan_cum.csv",
                   index=False, encoding="utf-8-sig")
    print("Nhãn cụm và Silhouette từng khách hàng: "
          f"{THU_MUC_KET_QUA / 'khach_hang_gan_cum.csv'}")
    pd.DataFrame([{"K": k, "Seed": kq["seed"], "SoDiem": len(X),
                   "SoThuocTinh": X.shape[1], "SoVongLap": kq["so_vong"],
                   "SSE": kq["sse"], "Silhouette": sil["trung_binh"]}]).to_csv(
        THU_MUC_KET_QUA / "danh_gia_mo_hinh.csv",
        index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả vào thư mục: {THU_MUC_KET_QUA.resolve()}")


def main():
    du_lieu = doc_du_lieu()                          # 1
    ten_cot, X = chon_thuoc_tinh(du_lieu)            # 2
    k_goi_y = elbow(X, max_iter=300)                 # 3
    ket_qua = huan_luyen(X, k_goi_y)                 # 4
    chay_thu(du_lieu, ten_cot, X, ket_qua)           # 5


if __name__ == "__main__":
    main()
