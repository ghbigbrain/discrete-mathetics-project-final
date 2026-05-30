import math
import time
import os
import random
import tracemalloc

# =====================================================================
# TỪ ĐIỂN LƯU KẾT QUẢ TỐI ƯU (BEST KNOWN SOLUTIONS) TỪ TSPLIB
# =====================================================================
OPTIMAL_SOLUTIONS = {
    "a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686,
    "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "bier127": 118282,
    "brazil58": 25395, "brd14051": 469385, "brg180": 1950, "burma14": 3323,
    "ch130": 6110, "ch150": 6528, "d198": 15780, "d493": 35002, "d657": 48912,
    "d1291": 50801, "d1655": 62128, "d2103": 80450, "d15112": 1573084,
    "d18512": 645238, "dantzig42": 699, "dsj1000": 18660188, "eil51": 426,
    "eil76": 538, "eil101": 629, "fl417": 11861, "fl1400": 20127,
    "fl1577": 22249, "fl3795": 28772, "fnl4461": 182566, "fri26": 937,
    "gil262": 2378, "gr17": 2085, "gr21": 2707, "gr24": 1272, "gr48": 5046,
    "gr96": 55209, "gr120": 6942, "gr137": 69853, "gr202": 40160,
    "gr229": 134602, "gr431": 171414, "gr666": 294358, "hk48": 11461,
    "kroA100": 21282, "kroB100": 22141, "kroC100": 20749, "kroD100": 21294,
    "kroE100": 22068, "kroA150": 26524, "kroB150": 26130, "kroA200": 29368,
    "kroB200": 29437, "lin105": 14379, "lin318": 42029, "linhp318": 41345,
    "nrw1379": 56638, "p654": 34643, "pa561": 2763, "pcb442": 50778,
    "pcb1173": 56892, "pcb3038": 137694, "pla7397": 23260728,
    "pla33810": 66048945, "pla85900": 142382641, "pr76": 108159,
    "pr107": 44303, "pr124": 59030, "pr136": 96772, "pr144": 58537,
    "pr152": 73682, "pr226": 80369, "pr264": 49135, "pr299": 48191,
    "pr439": 107217, "pr1002": 259045, "pr2392": 378032, "rat99": 1211,
    "rat195": 2323, "rat575": 6773, "rat783": 8806, "rd100": 7910,
    "rd400": 15281, "rl1304": 252948, "rl1323": 270199, "rl1889": 316536,
    "rl5915": 565530, "rl5934": 556045, "rl11849": 923288, "si175": 21407,
    "si535": 48450, "si1032": 92650, "st70": 675, "swiss42": 1273,
    "ts225": 126643, "tsp225": 3916, "u159": 42080, "u574": 36905,
    "u724": 41910, "u1060": 224094, "u1432": 152970, "u1817": 57201,
    "u2152": 64253, "u2319": 234256, "ulysses16": 6859, "ulysses22": 7013,
    "usa13509": 19982859, "vm1084": 239297, "vm1748": 336556
}

# =====================================================================
# 1. HỆ THỐNG ĐỌC VÀ CHUYỂN ĐỔI DỮ LIỆU THÔNG MINH
# =====================================================================
def compute_distance(c1, c2, weight_type):
    """Tính khoảng cách toán học giữa 2 tọa độ (chỉ dùng khi dataset có tọa độ X, Y)."""
    x1, y1 = c1
    x2, y2 = c2
    
    if weight_type == "EUC_2D":
        # Chuẩn Euclid làm tròn về số nguyên
        return int(round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2)))
        
    elif weight_type == "ATT":
        # Chuẩn Pseudo-Euclidean
        rij = math.sqrt(((x1 - x2)**2 + (y1 - y2)**2) / 10.0)
        tij = int(round(rij))
        return tij + 1 if tij < rij else tij
        
    elif weight_type == "GEO":
        # Chuẩn bề mặt Trái Đất (Haversine/TSPLIB chuẩn)
        def to_rad(deg_coord):
            deg = int(deg_coord)
            return math.radians(deg + ((deg_coord - deg) * 100.0) / 60.0)
        
        RRR = 6378.388
        lat1, lon1 = to_rad(x1), to_rad(y1)
        lat2, lon2 = to_rad(x2), to_rad(y2)
        
        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        
    elif weight_type == "CEIL_2D":
        # Chuẩn CEIL_2D (Dùng riêng cho các file như dsj1000)
        return int(math.ceil(math.sqrt((x1 - x2)**2 + (y1 - y2)**2)))

    return 0

def load_tsp_dataset(filepath):
    """
    Đọc mọi file TSPLIB và chuyển hóa thành Ma trận khoảng cách D (N x N).
    Hỗ trợ cả dữ liệu dạng Tọa độ (NODE_COORD_SECTION) lẫn Dữ liệu thô (EDGE_WEIGHT_SECTION).
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Khởi tạo các thông tin mặc định
    dimension = 0
    weight_type = "EUC_2D"
    weight_format = ""
    
    # Giai đoạn 1: Quét Header
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            weight_format = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION") or line.startswith("EDGE_WEIGHT_SECTION"):
            break
        idx += 1

    # Tạo ma trận D 2D kích thước N x N ban đầu toàn số 0
    D = [[0] * dimension for _ in range(dimension)]
    
    # Giai đoạn 2A: Xử lý file chứa tọa độ (NODE_COORD_SECTION)
    if lines[idx].startswith("NODE_COORD_SECTION"):
        coords = []
        idx += 1
        while idx < len(lines) and lines[idx] != "EOF":
            parts = lines[idx].split()
            if len(parts) >= 3:
                # Dùng float() tự động xử lý được cả số khoa học (như 1.11630e+03)
                coords.append((float(parts[1]), float(parts[2]))) 
            idx += 1
            
        # Tự động tính toán và điền vào ma trận D
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    D[i][j] = compute_distance(coords[i], coords[j], weight_type)

    # Giai đoạn 2B: Xử lý file chứa mảng Explicit (EDGE_WEIGHT_SECTION)
    elif lines[idx].startswith("EDGE_WEIGHT_SECTION"):
        idx += 1
        numbers = []
        # Gom toàn bộ các con số trong các dòng bên dưới thành 1 mảng 1 chiều
        while idx < len(lines) and lines[idx] != "EOF" and not lines[idx].startswith("DISPLAY_DATA_SECTION"):
            numbers.extend([int(float(x)) for x in lines[idx].split()])
            idx += 1
            
        # Ánh xạ mảng 1 chiều vào ma trận 2D dựa theo format
        num_idx = 0
        if weight_format == "UPPER_ROW":
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    D[i][j] = D[j][i] = numbers[num_idx]
                    num_idx += 1
                    
        elif weight_format == "LOWER_DIAG_ROW":
            for i in range(dimension):
                for j in range(i + 1):
                    D[i][j] = D[j][i] = numbers[num_idx]
                    num_idx += 1
                    
        elif weight_format == "FULL_MATRIX":
            for i in range(dimension):
                for j in range(dimension):
                    D[i][j] = numbers[num_idx]
                    num_idx += 1

    return D, dimension, weight_type

# =====================================================================
# CÁC THUẬT TOÁN ĐÃ ĐƯỢC CHUYỂN ĐỔI ĐỂ CHẠY TRÊN MA TRẬN D
# Việc sử dụng Ma trận giúp tốc độ thực thi tăng lên cực kỳ đáng kể
# =====================================================================

# --- 1. NEAREST NEIGHBOR ---
def nearest_neighbor_tsp(D, n, start_index=0):
    unvisited = set(range(n))
    current = start_index
    unvisited.remove(current)
    
    tour = [current]
    total_distance = 0.0

    while unvisited:
        # Tìm index có khoảng cách D[current][city] nhỏ nhất
        next_city = min(unvisited, key=lambda city: D[current][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        total_distance += D[current][next_city]
        current = next_city

    total_distance += D[current][tour[0]]
    return tour, total_distance


# --- 2. CHEAPEST INSERTION ---
def cheapest_insertion_tsp(D, n):
    if n <= 2: 
        return list(range(n)), (0 if n<2 else D[0][1]*2)

    # Tìm cặp đỉnh gần nhau nhất làm mầm
    best_dist = float('inf')
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] < best_dist:
                best_dist = D[i][j]
                best_pair = (i, j)
                
    tour = list(best_pair)
    remaining = set(range(n)) - set(tour)

    while remaining:
        best_node, best_pos, best_delta = None, None, float('inf')
        m = len(tour)
        
        for city in remaining:
            # Thử chèn city vào tất cả các vị trí
            for i in range(m):
                a, b = tour[i], tour[(i + 1) % m]
                delta = D[a][city] + D[city][b] - D[a][b]
                if delta < best_delta:
                    best_delta, best_node, best_pos = delta, city, i + 1

        tour.insert(best_pos, best_node)
        remaining.remove(best_node)

    total_distance = sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    return tour, total_distance


# --- 3. HILL CLIMBING 2-OPT ---
def hill_climbing_tsp(tour, initial_distance, D, n):
    if n < 3: return tour, initial_distance
    current_tour = tour.copy()
    current_distance = initial_distance
    
    while True:
        best_delta, best_i, best_j = 0.0, -1, -1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = current_tour[i - 1], current_tour[i]
                c, d = current_tour[j], current_tour[(j + 1) % n] 

                # Sử dụng ma trận D để truy xuất dữ liệu O(1)
                delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
                
                if delta < best_delta - 1e-9: 
                    best_delta, best_i, best_j = delta, i, j
        
        if best_i == -1: break
        current_tour[best_i : best_j + 1] = reversed(current_tour[best_i : best_j + 1])
        current_distance += best_delta
        
    return current_tour, current_distance


# --- 4. SIMULATED ANNEALING ---
def simulated_annealing_tsp(tour, initial_distance, D, n, temp=100.0, cooling_rate=0.99999):
    if n < 3: return tour, initial_distance
    current_tour = tour.copy()
    current_distance = initial_distance
    best_tour, best_distance = current_tour.copy(), current_distance

    while temp > 1e-4:
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)

        a, b = current_tour[i - 1], current_tour[i]
        c, d = current_tour[j], current_tour[(j + 1) % n]

        delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
        
        if delta < 0 or math.exp(-delta / temp) > random.random():
            current_tour[i : j + 1] = reversed(current_tour[i : j + 1])
            current_distance += delta
            if current_distance < best_distance:
                best_distance = current_distance
                best_tour = current_tour.copy()

        temp *= cooling_rate
    return best_tour, best_distance


# =====================================================================
# CHƯƠNG TRÌNH CHÍNH
# =====================================================================
if __name__ == "__main__":
    # Điền thư mục chứa các file .tsp của bạn
    base_path = r"/home/ghbigbrain/Downloads/dataset/tsplib-master"
    
    # Bạn có thể gõ "bayg29.tsp", "dantzig42.tsp", "d493.tsp" hoặc bất kỳ file nào khác
    file_name = "ali535.tsp" 
    dataset_path = os.path.join(base_path, file_name)

    try:
        # Nạp Dữ Liệu
        D_matrix, num_vertices, w_type = load_tsp_dataset(dataset_path)
        dataset_name = file_name.replace('.tsp', '').strip()
        
        print("=" * 80)
        print(f"[THÔNG TIN DATASET (Hệ thống Auto-Matrix)]")
        print(f" -> Tên File: {file_name}")
        print(f" -> Số đỉnh : {num_vertices}")
        print(f" -> Loại đo lường nhận diện được: {w_type}")
        print("=" * 80 + "\n")
        
        # --- Chạy Phương Pháp 1: NN + SA ---
        tracemalloc.start()
        start_time_m1 = time.time()
        
        init_tour_m1, init_dist_m1 = nearest_neighbor_tsp(D_matrix, num_vertices)
        final_tour_m1, final_dist_m1 = simulated_annealing_tsp(init_tour_m1, init_dist_m1, D_matrix, num_vertices)
        
        elapsed_time_m1 = (time.time() - start_time_m1) * 1000 
        _, peak_mem_m1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # --- Chạy Phương Pháp 2: CI + HC ---
        tracemalloc.start()
        start_time_m2 = time.time()
        
        init_tour_m2, init_dist_m2 = cheapest_insertion_tsp(D_matrix, num_vertices)
        final_tour_m2, final_dist_m2 = hill_climbing_tsp(init_tour_m2, init_dist_m2, D_matrix, num_vertices)
        
        elapsed_time_m2 = (time.time() - start_time_m2) * 1000 
        _, peak_mem_m2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # --- Phân Tích Báo Cáo ---
        imp_rate_m1 = ((init_dist_m1 - final_dist_m1) / init_dist_m1) * 100
        imp_rate_m2 = ((init_dist_m2 - final_dist_m2) / init_dist_m2) * 100
        mem_kb_m1, mem_kb_m2 = peak_mem_m1 / 1024, peak_mem_m2 / 1024
        optimal_dist = OPTIMAL_SOLUTIONS.get(dataset_name)

        print("=" * 100)
        print("                 BÁO CÁO ĐỐI SÁNH HIỆU NĂNG (ARCHITECTURE: DISTANCE MATRIX)")
        print("=" * 100)
        print(f"{'Thông số đo đạc (Metrics)':<40} | {'Method 1 (NN+SA)':<25} | {'Method 2 (CI+HC)':<25}")
        print("-" * 100)
        print(f"{'1. Lộ trình khởi tạo (Initial Tour)':<40} | {init_dist_m1:<25.2f} | {init_dist_m2:<25.2f}")
        print(f"{'2. Lộ trình tối ưu cuối (Final Tour)':<40} | {final_dist_m1:<25.2f} | {final_dist_m2:<25.2f}")
        print(f"{'3. Tỷ lệ tự cải thiện (Improvement)':<40} | {f'{imp_rate_m1:.2f}%':<25} | {f'{imp_rate_m2:.2f}%':<25}")
        print("-" * 100)
        print(f"{'4. Tổng thời gian chạy (Execution Time)':<40} | {f'{elapsed_time_m1:.2f} ms':<25} | {f'{elapsed_time_m2:.2f} ms':<25}")
        print(f"{'5. Bộ nhớ RAM đỉnh (Peak Memory)':<40} | {f'{mem_kb_m1:.2f} KB':<25} | {f'{mem_kb_m2:.2f} KB':<25}")
        print("-" * 100)

        if optimal_dist:
            gap_m1, gap_m2 = ((final_dist_m1 - optimal_dist) / optimal_dist) * 100, ((final_dist_m2 - optimal_dist) / optimal_dist) * 100
            eff_m1, eff_m2 = (optimal_dist / final_dist_m1) * 100, (optimal_dist / final_dist_m2) * 100
            print(f"{'6. Tối ưu tuyệt đối (Best Known Solution)':<40} | {optimal_dist:<25} | {optimal_dist:<25}")
            print(f"{'7. Độ lệch tối ưu (Gap to Optimal)':<40} | {f'+{gap_m1:.2f}% (Sai số)':<25} | {f'+{gap_m2:.2f}% (Sai số)':<25}")
            print(f"{'8. Hiệu quả đạt được (Accuracy %)':<40} | {f'{eff_m1:.2f}%':<25} | {f'{eff_m2:.2f}%':<25}")
        else:
            print(f"{'6. Tối ưu tuyệt đối (Best Known Solution)':<40} | {'Không có dữ liệu':<25} | {'Không có dữ liệu':<25}")
        print("=" * 100)

    except FileNotFoundError:
        print(f"[ERROR] Không tìm thấy file '{file_name}' trong '{base_path}'.")
