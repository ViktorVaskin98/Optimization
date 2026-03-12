import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Tuple

class OptimizationVisualizer:
    def __init__(
        self,
        save_dir: str = "Plots"
    ) -> None:
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        plt.style.use('seaborn-v0_8-whitegrid') 

        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

    def _get_grid(
        self,
        path: List[List[float]],
        padding: float = 0.5,
        resolution: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        path_np = np.array(path)
        x_min, x_max = path_np[:, 0].min(), path_np[:, 0].max()
        y_min, y_max = path_np[:, 1].min(), path_np[:, 1].max()

        x = np.linspace(x_min - padding, x_max + padding, resolution)
        y = np.linspace(y_min - padding, y_max + padding, resolution)
        return np.meshgrid(x, y)

    def plot_3d_surface(
        self,
        func_np: Callable, 
        x_range: Tuple[float, float], 
        y_range: Tuple[float, float], 
        filename: str = "surface_3d.png"
    ) -> None:
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = func_np(X, Y)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_title("Ландшафт целевой функции (3D)", pad=20)
        ax.set_xlabel("Ось X")
        ax.set_ylabel("Ось Y")
        ax.set_zlabel("Значение функции (Loss)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Visualizer] Сохранен 3D график: {filepath}")

    def plot_optimization_path(
        self,
        point_history: List[List[float]], 
        func_np: Callable, true_optimum: List[float], 
        filename: str = "path_2d.png"
    ) -> None:
        if len(point_history[0]) > 2:
            print("Внимание: алгоритм работал в >2 мерном пространстве. Отрисована проекция на X и Y")

        X, Y = self._get_grid(point_history, padding=0.5)
        Z = func_np(X, Y)

        plt.figure(figsize=(10, 8))

        levels = np.logspace(0, 4, 30)

        plt.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.6, locator=plt.matplotlib.ticker.LogLocator())
        plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)

        path_np = np.array(point_history)
        xs = path_np[:, 0]
        ys = path_np[:, 1]

        plt.plot(xs, ys, color='crimson', marker='o', linestyle='-', linewidth=2, markersize=4, label='Траектория оптимизатора')
        plt.plot(xs[0], ys[0], 'go', markersize=10, label='Старт', markeredgecolor='black')
        plt.plot(xs[-1], ys[-1], 'r*', markersize=15, label='Финиш (Оптимум)', markeredgecolor='black')
        plt.plot(true_optimum[0], true_optimum[1], '*', color='gold', markersize=18, label='Истинный минимум', markeredgecolor='black')


        plt.title("Траектория спуска (Контурный график)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


    def plot_loss_history(
        self,
        loss_history: List[float], 
        filename: str = "loss_history.png"
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, color='darkorange', linewidth=2.5)
        
        plt.title("Сходимость функции (Loss vs Iteration)")
        plt.xlabel("Итерация")
        plt.ylabel("Значение функции f(x)")
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--", alpha=0.5)

        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_distance_history(
        self,
        point_history: List[List[float]], 
        true_optimum: List[float], 
        filename: str = "distance_history.png"
    ) -> None:
        path_np = np.array(point_history)
        optimum_np = np.array(true_optimum)
    
        distances = np.linalg.norm(path_np - optimum_np, axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(distances, color='purple', linewidth=2.5)
        
        plt.title("Расстояние до истинного оптимума")
        plt.xlabel("Итерация")
        plt.ylabel("Евклидово расстояние")
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)

        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_full_report(
        self, 
        opt_result: Dict, 
        func_np: Callable, 
        optimizer_name: str, 
        true_optimum: List[float]
    ) -> None:
        prefix = optimizer_name.lower().replace(" ", "_")
        
        print(f"\n[Visualizer] Генерация отчета для: {optimizer_name}...")
        
        self.plot_optimization_path(
            point_history=opt_result["point_history"],
            func_np=func_np,
            true_optimum=true_optimum,
            filename=f"{prefix}_path_2d.png"
        )
        
        self.plot_loss_history(
            opt_result["loss_history"], 
            filename=f"{prefix}_loss.png"
        )
        
        self.plot_distance_history(
            opt_result["point_history"], 
            true_optimum, 
            filename=f"{prefix}_distance.png"
        )
        print(f"[Visualizer] Отчет успешно сохранен в папку '{self.save_dir}'!")