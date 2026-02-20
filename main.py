import numpy as np
import matplotlib.pyplot as plt

# ================== Вспомогательные функции ==================
def normalize(v):
    """Нормировка вектора."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def print_table(title, rows, headers=None, fmt=".3f"):
    """Вывод таблицы в консоль (упрощённый вариант)."""
    print(f"\n{title}")
    if headers:
        print("\t".join(headers))
    for row in rows:
        print("\t".join([f"{x:{fmt}}" if isinstance(x, (int, float)) else str(x) for x in row]))

def orthogonal_projection(points, n):
    """
    Ортогональная проекция точек на плоскость, проходящую через начало координат
    и ортогональную вектору n.
    Возвращает проекции (в мировой системе) и нормированную нормаль.
    """
    n_norm = normalize(n)
    proj = []
    for p in points:
        t = np.dot(p, n_norm)
        p1 = p - t * n_norm
        proj.append(p1)
    return np.array(proj), n_norm

def basis_from_normal(n_norm):
    """
    Построение ортонормированного базиса (e1, e2, e3) такого, что e3 = -n_norm,
    e2 – проекция вектора Z = (0,0,1) на плоскость, e1 = e2 × e3.
    Возвращает e1, e2, e3.
    """
    Z = np.array([0., 0., 1.])
    # проекция Z на плоскость (ортогональную n_norm)
    Z_proj = Z - np.dot(Z, n_norm) * n_norm
    if np.linalg.norm(Z_proj) < 1e-6:
        # если Z коллинеарен нормали, берём другой вектор
        Z_proj = np.array([1., 0., 0.]) - np.dot(np.array([1.,0.,0.]), n_norm) * n_norm
    e2 = normalize(Z_proj)
    e3 = -n_norm
    e1 = np.cross(e2, e3)
    # проверка ориентации (определитель должен быть +1)
    if np.linalg.det([e1, e2, e3]) < 0:
        e1 = -e1  # меняем знак для правой тройки
    return e1, e2, e3

def compute_external_normals(A, B, C, D):
    """
    Вычисляет внешние нормали для всех четырёх граней тетраэдра,
    используя метод определителей с четвёртой вершиной.
    Возвращает словарь: {'BCD': n, 'ACD': n, 'ABD': n, 'ABC': n}
    """
    def external_for_face(P, Q, R, S):
        # P, Q, R – вершины грани (порядок важен для ориентации)
        # S – четвёртая вершина тетраэдра (не принадлежит грани)
        v1 = Q - P
        v2 = R - P
        v_out = S - P  # вектор из P в четвёртую вершину
        # два возможных векторных произведения
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v1)
        # смешанные произведения
        det1 = np.dot(v_out, n1)
        det2 = np.dot(v_out, n2)
        # выбираем ту нормаль, для которой соответствующее смешанное произведение отрицательно
        if det1 < 0:
            return n1
        elif det2 < 0:
            return n2
        else:
            # на случай равенства (маловероятно) возвращаем n1
            return n1

    normals = {}
    normals['BCD'] = external_for_face(B, C, D, A)
    normals['ACD'] = external_for_face(A, C, D, B)
    normals['ABD'] = external_for_face(A, B, D, C)
    normals['ABC'] = external_for_face(A, B, C, D)
    return normals

def visible_faces_ortho(normals, view_dir):
    """Определяет видимые грани при ортогональном проектировании (взгляд вдоль view_dir)."""
    visible = []
    for name, n in normals.items():
        if np.dot(view_dir, n) < 0:
            visible.append(name)
    return visible

def central_projection(points, N, plane_normal):
    """
    Центральная проекция точек из точки N на плоскость, проходящую через начало
    и ортогональную plane_normal.
    Возвращает массив проекций.
    """
    proj = []
    for P in points:
        vec = P - N
        denom = np.dot(plane_normal, vec)
        if abs(denom) < 1e-12:
            proj.append(np.full(3, np.inf))  # бесконечность
        else:
            t = -np.dot(plane_normal, N) / denom
            P1 = N + t * vec
            proj.append(P1)
    return np.array(proj)

def camera_basis(N, H, up=np.array([0.,1.,0.])):
    """
    Построение базиса камеры: e3 направлен от центра к камере (N-H), нормирован.
    e1 = up × e3, затем e2 = e3 × e1.
    Возвращает e1, e2, e3.
    """
    e3 = normalize(N - H)  # направление от центра к камере
    # если up коллинеарен e3, берём другой up
    if abs(np.dot(up, e3)) > 0.999:
        up = np.array([0.,0.,1.])
    e1 = normalize(np.cross(up, e3))
    e2 = np.cross(e3, e1)  # правая тройка
    return e1, e2, e3

def view_matrix(e1, e2, e3, N):
    """
    Матрица вида 4x4.
    """
    M = np.eye(4)
    M[0, :3] = e1
    M[1, :3] = e2
    M[2, :3] = e3
    M[0, 3] = -np.dot(e1, N)
    M[1, 3] = -np.dot(e2, N)
    M[2, 3] = -np.dot(e3, N)
    return M

def rotation_matrix(axis, angle_deg):
    """
    Матрица поворота вокруг единичного вектора axis на угол angle_deg (против часовой).
    Используется формула Родрига.
    """
    angle = np.radians(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c + (1-c)*ux*ux,      (1-c)*ux*uy - s*uz,  (1-c)*ux*uz + s*uy],
        [(1-c)*uy*ux + s*uz,   c + (1-c)*uy*uy,      (1-c)*uy*uz - s*ux],
        [(1-c)*uz*ux - s*uy,   (1-c)*uz*uy + s*ux,   c + (1-c)*uz*uz]
    ])

def plot_projection(points_2d, visible_faces, title, labels=['A', 'B', 'C', 'D']):
    """
    Строит 2D график проекции. points_2d – массив (4,2) координат вершин.
    visible_faces – список названий видимых граней (например ['ABC','ACD']).
    """
    # соответствие граней и индексов вершин
    face_indices = {
        'ABC': [0,1,2],
        'ABD': [0,1,3],
        'ACD': [0,2,3],
        'BCD': [1,2,3]
    }
    plt.figure()
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    # рисуем все рёбра невидимых граней пунктиром (опционально)
    all_faces = ['ABC','ABD','ACD','BCD']
    for face in all_faces:
        idx = face_indices[face]
        x = [points_2d[i,0] for i in idx] + [points_2d[idx[0],0]]
        y = [points_2d[i,1] for i in idx] + [points_2d[idx[0],1]]
        if face in visible_faces:
            plt.plot(x, y, 'b-', linewidth=2)
        else:
            plt.plot(x, y, 'r--', linewidth=1, alpha=0.5)
    # подписи вершин
    for i, label in enumerate(labels):
        plt.text(points_2d[i,0], points_2d[i,1], label, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# ================== Основная программа ==================
def main():
    # Данные из примера (можно заменить на свой вариант)
    A = np.array([24., 19., 37.])
    B = np.array([25., 14., 32.])
    C = np.array([24., 19., 31.])
    D = np.array([24., 11., 32.])
    n = np.array([5., 1., 10.])               # для задания 1
    N = np.array([-11., -15., -10.])          # для заданий 2,3

    # Вершины в одном массиве
    points = np.array([A, B, C, D])
    labels = ['A', 'B', 'C', 'D']

    # === Задание 1 ===
    print("="*50)
    print("ЗАДАНИЕ 1. Ортогональное проектирование")
    print("="*50)

    # а) ортогональные проекции
    proj1, n_norm = orthogonal_projection(points, n)
    print("\nНормированный вектор n_n =", n_norm.round(3))
    print("\nПроекции вершин (мировые координаты):")
    for i, p in enumerate(proj1):
        print(f"{labels[i]}1 = {p.round(3)}")

    # б) базис в плоскости и 2D координаты
    e1, e2, e3 = basis_from_normal(n_norm)
    print("\nБазис в плоскости:")
    print(f"e1 = {e1.round(3)}")
    print(f"e2 = {e2.round(3)}")
    print(f"e3 = {e3.round(3)}")

    points_2d_1 = np.array([[np.dot(p, e1), np.dot(p, e2)] for p in proj1])
    print("\nКоординаты в плоскости (X, Y):")
    for i, (x,y) in enumerate(points_2d_1):
        print(f"{labels[i]}1({x:.3f}, {y:.3f})")

    # в) видимость граней
    normals = compute_external_normals(A, B, C, D)
    print("\nВнешние нормали граней:")
    for name, val in normals.items():
        print(f"{name}: {val.round(3)}")

    view_dir = n  # направление взгляда
    visible_1 = visible_faces_ortho(normals, view_dir)
    print("\nСкалярные произведения с направлением взгляда:")
    for name, val in normals.items():
        prod = np.dot(view_dir, val)
        print(f"{name}: {prod:.3f}")
    print("Видимые грани:", visible_1)

    plot_projection(points_2d_1, visible_1, "Задание 1: Ортогональная проекция", labels)

    # === Задание 2 ===
    print("\n" + "="*50)
    print("ЗАДАНИЕ 2. Центральное проектирование")
    print("="*50)

    H2 = np.mean(points, axis=0)
    print("Центр тетраэдра H =", H2.round(3))

    n2 = H2 - N  # направление взгляда и нормаль плоскости проекции
    print("Вектор нормали плоскости (NH) =", n2.round(3))

    # а) центральные проекции
    proj2 = central_projection(points, N, n2)
    print("\nЦентральные проекции вершин (мировые координаты):")
    for i, p in enumerate(proj2):
        print(f"{labels[i]}' = {p.round(3)}")

    # б) базис в плоскости проекции
    n2_norm = normalize(n2)
    e1_2, e2_2, e3_2 = basis_from_normal(n2_norm)
    print("\nБазис в плоскости проекции:")
    print(f"e1 = {e1_2.round(3)}")
    print(f"e2 = {e2_2.round(3)}")
    print(f"e3 = {e3_2.round(3)}")

    points_2d_2 = np.array([[np.dot(p, e1_2), np.dot(p, e2_2)] for p in proj2])
    print("\nКоординаты в плоскости (X, Y):")
    for i, (x,y) in enumerate(points_2d_2):
        print(f"{labels[i]}'({x:.3f}, {y:.3f})")

    # в) видимость граней при центральном проектировании
    visible_2 = []
    print("\nПроверка видимости граней (N - вершина)·n_ext >0 => видна)")
    for name, n_ext in normals.items():
        # возьмём первую вершину грани для проверки
        if name == 'ABC' or name == 'ABD' or name == 'ACD':
            vert = A
        else:
            vert = B
        vec = N - vert
        prod = np.dot(vec, n_ext)
        print(f"{name}: {prod:.3f}")
        if prod > 0:
            visible_2.append(name)
    print("Видимые грани:", visible_2)

    plot_projection(points_2d_2, visible_2, "Задание 2: Центральная проекция", labels)

    # === Задание 3 ===
    print("\n" + "="*50)
    print("ЗАДАНИЕ 3. Видовое преобразование (камера)")
    print("="*50)

    H3 = H2
    print("Центр тетраэдра H =", H3.round(3))
    print("Точка наблюдения N =", N)

    # а) базис камеры
    e1_cam, e2_cam, e3_cam = camera_basis(N, H3)
    print("\nБазис камеры:")
    print(f"e1 = {e1_cam.round(3)}")
    print(f"e2 = {e2_cam.round(3)}")
    print(f"e3 = {e3_cam.round(3)}")

    # б) матрица вида
    Mview = view_matrix(e1_cam, e2_cam, e3_cam, N)
    print("\nМатрица вида Mview:")
    print(Mview.round(3))

    # в) координаты вершин в пространстве камеры
    points_cam = []
    for P in points:
        P4 = np.append(P, 1)
        P_cam = Mview @ P4
        points_cam.append(P_cam[:3])
    points_cam = np.array(points_cam)
    print("\nКоординаты вершин в системе камеры (x_cam, y_cam, z_cam):")
    for i, p in enumerate(points_cam):
        print(f"{labels[i]}_cam = {p.round(3)}")

    # г) плоскость проекции: z = -d, где d = |N-H|/2
    d = np.linalg.norm(N - H3) / 2
    print(f"\nРасстояние до плоскости проекции d = {d:.3f}")

    # центральная проекция на эту плоскость
    proj_cam = []
    for p in points_cam:
        x_proj = -d * p[0] / p[2]   # p[2] отрицательно
        y_proj = -d * p[1] / p[2]
        proj_cam.append([x_proj, y_proj])
    proj_cam = np.array(proj_cam)
    print("\nПроекции на плоскость (x_proj, y_proj):")
    for i, (x,y) in enumerate(proj_cam):
        print(f"{labels[i]}_proj = ({x:.3f}, {y:.3f})")

    # д) видимость граней в камере
    R = Mview[:3, :3]  # матрица поворота из мировой в камеру
    normals_cam = {}
    for name, nw in normals.items():
        normals_cam[name] = R @ nw
    print("\nВнешние нормали в камере:")
    visible_3 = []
    for name, nc in normals_cam.items():
        print(f"{name}: {nc.round(3)}  z-компонента = {nc[2]:.3f}")
        if nc[2] > 0:
            visible_3.append(name)
    print("Видимые грани в камере:", visible_3)

    plot_projection(proj_cam, visible_3, "Задание 3: Видовое преобразование", labels)

    # === Задание 4 ===
    print("\n" + "="*50)
    print("ЗАДАНИЕ 4. Поворот объекта")
    print("="*50)

    # а) ось вращения AH
    A4 = A.copy()
    H4 = H3
    axis = H4 - A4
    axis_len = np.linalg.norm(axis)
    p_axis = axis / axis_len
    print(f"Ось вращения AH = {axis.round(3)}, единичный вектор p = {p_axis.round(3)}")

    # б) матрица поворота на 50°
    R_rot = rotation_matrix(p_axis, 50)
    print("\nМатрица поворота на 50° вокруг оси AH:")
    print(R_rot.round(3))

    # в) поворот вершин
    # перенос в начало координат (относительно A)
    points_local = points - A4
    points_rot = (R_rot @ points_local.T).T  # поворот
    points_new = points_rot + A4  # обратный перенос
    print("\nКоординаты вершин после поворота:")
    for i, p in enumerate(points_new):
        print(f"{labels[i]}_new = {p.round(3)}")

    # г) повторяем задание 1 для новых вершин
    A_new, B_new, C_new, D_new = points_new
    points_new_arr = np.array([A_new, B_new, C_new, D_new])

    # проекция на ту же плоскость (с нормалью n)
    proj1_new, n_norm_new = orthogonal_projection(points_new_arr, n)
    print("\nПроекции после поворота (мировые координаты):")
    for i, p in enumerate(proj1_new):
        print(f"{labels[i]}1_new = {p.round(3)}")

    # базис в плоскости (тот же, что и в задании 1, т.к. плоскость не меняется)
    e1_new, e2_new, e3_new = basis_from_normal(n_norm)
    points_2d_new = np.array([[np.dot(p, e1_new), np.dot(p, e2_new)] for p in proj1_new])
    print("\nКоординаты в плоскости (X, Y) после поворота:")
    for i, (x,y) in enumerate(points_2d_new):
        print(f"{labels[i]}1_new({x:.3f}, {y:.3f})")

    # внешние нормали для нового тетраэдра
    normals_new = compute_external_normals(A_new, B_new, C_new, D_new)
    print("\nВнешние нормали после поворота:")
    for name, val in normals_new.items():
        print(f"{name}: {val.round(3)}")

    # видимость
    visible_4 = visible_faces_ortho(normals_new, n)
    print("\nСкалярные произведения с направлением взгляда:")
    for name, val in normals_new.items():
        prod = np.dot(n, val)
        print(f"{name}: {prod:.3f}")
    print("Видимые грани после поворота:", visible_4)

    plot_projection(points_2d_new, visible_4, "Задание 4: После поворота", labels)

if __name__ == "__main__":
    main()