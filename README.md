# README

## Mô tả dự án

Repository này chứa các cài đặt cho 5 thuật toán tối ưu mô phỏng tự nhiên phổ biến trong lĩnh vực Trí tuệ Nhân tạo: **ACO**, **PSO**, **Firefly Algorithm**, **Cuckoo Search**, và **Artificial Bee Colony (ABC)**. 
Các thuật toán được áp dụng để giải các bài toán tối ưu như TSP hoặc hàm liên tục trong toán học.

---

## Các thư viện chung
1. Cài Python >= 3.10.
2. Cài thư viện cần thiết:
pip install numpy
pip install matplotlib

---

## 1. Ant Colony Optimization (ACO)

**Ý tưởng:** Thuật toán mô phỏng hành vi của đàn kiến tìm đường đến nguồn thức ăn. Kiến để lại pheromone trên tuyến đường tốt, giúp các kiến sau có xu hướng chọn đường tối ưu.

**Bài toán áp dụng:** TSP - The travelling saleman problem, một bài toán tìm chu trình ngắn nhất trong đồ thị có trọng số.

**Cài đặt:**

* Chạy file **demo_tsp.py** trong thư mục ACO_PSO. Câu lệnh tương ứng: best_route_aco, best_len_aco, history_aco, times_aco = tester.test_aco("data/small.txt", plot=True)
* Có thể thay đổi file thành data/tiny.txt, data/TSP51.txt,... mỗi file chứa số đỉnh khác nhau, ví dụ: file small.txt chứa 30 đỉnh tương ứng trường hợp test case nhỏ.

---

## 2. Particle Swarm Optimization (PSO)

**Ý tưởng:** Mô phỏng sự di chuyển của bầy chim/cá tìm kiếm thức ăn. Mỗi hạt cập nhật vận tốc dựa trên kinh nghiệm cá nhân và cả đàn.

**Bài toán áp dụng:** TSP - The travelling saleman problem, một bài toán tìm chu trình ngắn nhất trong đồ thị có trọng số.

**Cài đặt:**

* Chạy file **demo_tsp.py** trong thư mục ACO_PSO. Câu lệnh tương ứng: best_route_pso, best_len_pso, history_pso, times_pso = tester.test_pso("data/small.txt", plot = True)
* Có thể thay đổi file thành data/tiny.txt, data/TSP51.txt,... tương tự như ACO.

---

## 3. Firefly Algorithm

**Ý tưởng:** Bầy đom đóm bị hấp dẫn bởi độ sáng của nhau. Con sáng hơn thu hút con khác, giúp dẫn tới cực trị tối ưu.

**Cài đặt:**

* Trong file **readme.md** của thư mục firefly.

---

## 4. Cuckoo Search (CS)

**Ý tưởng:** Dựa trên hành vi ký sinh của chim cu: đặt trứng vào tổ chim khác. Thuật toán sử dụng Lévy Flight để khám phá mạnh mẽ không gian tìm kiếm.


**Cài đặt:**

* Chạy trực tiếp trên file jupiter notebook **CS.ipynb**. 
* Thay đổi hàm chạy thử nghiệm ở câu lệnh: bench_rastrigin = benchmark_functions[1] #Cell thứ 4 (phần 4b). Ví dụ, benchmark_functions[0] là hàm Sphere Function, benchmark_functions[2] là hàm Rosenbrock Function.

---

## 5. Artificial Bee Colony (ABC)

**Ý tưởng:** Mô phỏng hành vi tìm mật của ong: ong thợ, ong quan sát, ong do thám.


**Cài đặt:**

* Chạy file **run.py** trong thư mục ABC. Thay đổi hàm tương ứng ở câu lệnh: measure_performance(algo, _f.circle, 1, 1, 0)
* Có thể thay đổi hàm ở _f.circle (hàm đường tròn) thành các hàm như _f.rastrigin, _f.ackley,... và thay đổi các giá trị x,y tương ứng.

---

Tác giả: Nhóm 11 - 23TNT - Cơ sở AI - HCMUS


## Tác giả

Nhóm 11 – 23TNT – Môn Cơ sở Trí tuệ Nhân tạo.
