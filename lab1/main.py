import numpy as np
import decimal
from Core.functions import create_variables, get_rosenbrock, get_dixon_price_nd
from Optimization.optimizers import (
    GradientDescent, MomentumGradientDescent, AdamOptimizer, KieferWolfowitzOptimizer
)
from Visuals.visualizer import OptimizationVisualizer

def run_experiment(
    func_builder, 
    func_np, 
    optimizers_dict, 
    true_optimum, 
    experiment_name
) -> None:
    print(f"\n{'='*50}\nЗАПУСК ЭКСПЕРИМЕНТА: {experiment_name.upper()}\n{'='*50}")
    
    viz = OptimizationVisualizer(save_dir=f"Results_{experiment_name}")
    viz.plot_3d_surface(func_np, x_range=(-2, 2), y_range=(-1, 3))

    for opt_name, optimizer in optimizers_dict.items():
        print(f"\n--- Тестируем {opt_name} ---")
        
        variables = create_variables(2, initial_val=0.0)
        target_tree = func_builder(variables)

        result = optimizer.optimize(target_tree, variables, verbose=True)
        
        print(f"Финишная точка: {[float(p) for p in result['final_point']]}")
        print(f"Итераций затрачено: {len(result['loss_history']) - 1}")

        viz.generate_full_report(
            opt_result=result,
            func_np=func_np,
            optimizer_name=opt_name,
            true_optimum=true_optimum
        )

if __name__ == "__main__":
    decimal.getcontext().prec = 25

    # ========================================================
    # ЭКСПЕРИМЕНТ 1: ФУНКЦИЯ РОЗЕНБРОКА
    # ========================================================
    rosenbrock_np = lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2
    rosenbrock_true_opt = [1.0, 1.0]

    rosen_optimizers = {
        "Adam": AdamOptimizer(learning_rate=0.05, max_iter=5000, tolerance=0),
        "Momentum": MomentumGradientDescent(learning_rate=0.01, momentum=0.9, max_iter=5000, tolerance=0),
        "Vanilla_GD": GradientDescent(learning_rate=0.005, max_iter=5000, tolerance=0),
        "Kiefer_Wolfowitz": KieferWolfowitzOptimizer(learning_rate=0.05, c=0.01, max_iter=5000, tolerance=0) 
    }

    run_experiment(
        func_builder=get_rosenbrock,
        func_np=rosenbrock_np,
        optimizers_dict=rosen_optimizers,
        true_optimum=rosenbrock_true_opt,
        experiment_name="Rosenbrock"
    )

    # ========================================================
    # ЭКСПЕРИМЕНТ 2: ФУНКЦИЯ ДИКСОНА-ПРАЙСА
    # ========================================================

    dixon_np = lambda x, y: (x - 1)**2 + 2 * (2 * y**2 - x)**2
    dixon_true_opt = [1.0, 0.707106]

    dixon_optimizers = {
    "Adam": AdamOptimizer(learning_rate=0.01, max_iter=5000, tolerance=0),
    "Momentum": MomentumGradientDescent(learning_rate=0.005, momentum=0.95, max_iter=5000, tolerance=0)
    }

    run_experiment(
        func_builder=get_dixon_price_nd,
        func_np=dixon_np,
        optimizers_dict=dixon_optimizers,
        true_optimum=dixon_true_opt,
        experiment_name="Dixon_Price"
    )

    print("\nВсе эксперименты завершены")