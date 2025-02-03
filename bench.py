import time
import numpy as np
from sim import DoublePendulum, IntegrationMethod, GravitationalSystem

class IntegratorBenchmark:
    def __init__(self):
        self.total_time = 10.0
        self.max_energy = 1e6
        # For timestep analysis
        self.timesteps = np.logspace(-4, 0, num=30)
        # For step count analysis
        self.step_counts = np.logspace(1, 5, num=30, dtype=int)
        self.pendulum = GravitationalSystem()
        self.results = {
            'verlet': {
                'timesteps': [], 
                'step_counts': [],
                'timestep_energy_error': [], 
                'stepcount_energy_error': [],
                'timestep_ops': [],
                'stepcount_ops': [],
                'timestep_error_std': [],
                'timestep_error_min': [],
                'timestep_error_max': [],
                'stepcount_error_std': [],
                'stepcount_error_min': [],
                'stepcount_error_max': []
            },
            'rk4': {
                'timesteps': [],
                'step_counts': [],
                'timestep_energy_error': [],
                'stepcount_energy_error': [],
                'timestep_ops': [],
                'stepcount_ops': [],
                'timestep_error_std': [],
                'timestep_error_min': [],
                'timestep_error_max': [],
                'stepcount_error_std': [],
                'stepcount_error_min': [],
                'stepcount_error_max': []
            },
            'euler': {
                'timesteps': [],
                'step_counts': [],
                'timestep_energy_error': [],
                'stepcount_energy_error': [],
                'timestep_ops': [],
                'stepcount_ops': [],
                'timestep_error_std': [],
                'timestep_error_min': [],
                'timestep_error_max': [],
                'stepcount_error_std': [],
                'stepcount_error_min': [],
                'stepcount_error_max': []
            }
        }
    
    def is_state_valid(self, energy):
        return (np.isfinite(energy) and 
                abs(energy) < self.max_energy)
        
    def count_derivative_flops(self):
        if isinstance(self.pendulum, DoublePendulum):
            # Double pendulum derivative includes:
            # - 6 trigonometric operations (sin, cos)
            # - ~15 multiplications
            # - ~10 additions/subtractions
            # - 2 divisions
            return 6 * 10 + 15 + 10 + 2  # trig ops are ~10 FLOPs each
        else:  # GravitationalSystem
            # Gravitational system derivative includes:
            # - ~10 multiplications for distance calculations
            # - ~8 additions/subtractions
            # - 2 square roots
            # - 3 divisions
            return 10 + 8 + 2 * 20 + 3  # sqrt is ~20 FLOPs
    
    def run_timestep_benchmark(self):
        for dt in self.timesteps:
            for method in ['verlet', 'rk4', 'euler']:
                try:
                    self.pendulum = GravitationalSystem()
                    initial_energy = self.pendulum.calculate_energy()
                    steps = int(self.total_time / dt)
                    energy_errors = []
                    
                    # Count operations
                    deriv_flops = self.count_derivative_flops()
                    total_ops = steps * (deriv_flops + 5 if method == 'verlet' else 4 * deriv_flops + 10)
                    
                    # Run simulation with fixed dt
                    for _ in range(steps):
                        try:
                            if method == 'verlet':
                                self.pendulum.verlet_step(dt)
                            elif method == 'rk4':
                                self.pendulum.rk4_step(dt)
                            else:
                                self.pendulum.euler_step(dt)
                            
                            current_energy = self.pendulum.calculate_energy()
                            if self.is_state_valid(current_energy):
                                energy_errors.append(abs(current_energy - initial_energy))
                                
                        except (RuntimeWarning, OverflowError):
                            continue
                    
                    # Store results if enough valid steps
                    if len(energy_errors) > steps // 4:
                        self.results[method]['timesteps'].append(dt)
                        self.results[method]['timestep_ops'].append(total_ops)
                        self.results[method]['timestep_energy_error'].append(np.mean(energy_errors))
                        self.results[method]['timestep_error_std'].append(np.std(energy_errors))
                        self.results[method]['timestep_error_min'].append(np.min(energy_errors))
                        self.results[method]['timestep_error_max'].append(np.max(energy_errors))
                        
                except Exception as e:
                    print(f"Error in {method} at dt={dt:.6f}: {e}")
                    continue
              
    def run_stepcount_benchmark(self):
        for steps in self.step_counts:
            for method in ['verlet', 'rk4', 'euler']:
                try:
                    self.pendulum = GravitationalSystem()
                    initial_energy = self.pendulum.calculate_energy()
                    dt = self.total_time / steps
                    energy_errors = []
                    
                    deriv_flops = self.count_derivative_flops()
                    if (method == 'verlet'):
                        total_ops = steps * (deriv_flops + 5)
                    elif (method == 'euler'):
                        total_ops = steps * (deriv_flops + 14)
                    else: # rk4
                        total_ops = steps * (4 * deriv_flops + 10)
                    
                    for _ in range(steps):
                        try:
                            if method == 'verlet':
                                self.pendulum.verlet_step(dt)
                            elif method == 'rk4':
                                self.pendulum.rk4_step(dt)
                            else:
                                self.pendulum.euler_step(dt)
                            
                            current_energy = self.pendulum.calculate_energy()
                            if self.is_state_valid(current_energy):
                                energy_errors.append(abs(current_energy - initial_energy))
                                
                        except (RuntimeWarning, OverflowError):
                            continue
                    
                    if len(energy_errors) > steps//2:
                        self.results[method]['step_counts'].append(steps)
                        self.results[method]['stepcount_ops'].append(total_ops)
                        self.results[method]['stepcount_energy_error'].append(np.mean(energy_errors))
                        self.results[method]['stepcount_error_std'].append(np.std(energy_errors))
                        self.results[method]['stepcount_error_min'].append(np.min(energy_errors))
                        self.results[method]['stepcount_error_max'].append(np.max(energy_errors))
                        
                except Exception as e:
                    print(f"Error in {method} at steps={steps}: {e}")
                    continue
                
                
    def plot_results(self):
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        colors = {'verlet': 'blue', 'rk4': 'red', 'euler': 'green'}
        methods = ['verlet', 'rk4', 'euler']
        
        # Timestep analysis plots
        for method in methods:
            ax1.plot(self.results[method]['timesteps'], 
                    self.results[method]['timestep_ops'], 
                    color=colors[method], 
                    label=method.upper())
            ax2.semilogy(self.results[method]['timesteps'],
                        self.results[method]['timestep_energy_error'],
                        color=colors[method], 
                        label=method.upper())
            ax2.fill_between(self.results[method]['timesteps'],
                   self.results[method]['timestep_error_min'],
                   self.results[method]['timestep_error_max'],
                   alpha=0.2,
                   color=colors[method])
        
        ax1.set_xlabel('Timestep (s)')
        ax1.set_ylabel('Operation Count')
        ax1.set_title('Timestep vs Computational Cost')
        ax1.legend()
        
        ax2.set_xlabel('Timestep (s)')
        ax2.set_ylabel('Energy Error (J)')
        ax2.set_title('Timestep vs Energy Error')
        ax2.legend()
        
        # Step count analysis plots
        for method in methods:
            ax3.plot(self.results[method]['step_counts'], 
                    self.results[method]['stepcount_ops'], 
                    color=colors[method], 
                    label=method.upper())
            ax4.semilogy(self.results[method]['step_counts'],
                        self.results[method]['stepcount_energy_error'],
                        color=colors[method], 
                        label=method.upper())
            ax4.fill_between(self.results[method]['step_counts'],
                   self.results[method]['stepcount_error_min'],
                   self.results[method]['stepcount_error_max'],
                   alpha=0.2,
                   color=colors[method])
            
        
        ax3.set_xlabel('Number of Steps')
        ax3.set_ylabel('Operation Count')
        ax3.set_title('Steps vs Computational Cost')
        ax3.legend()
        
        ax4.set_xlabel('Number of Steps')
        ax4.set_ylabel('Energy Error (J)')
        ax4.set_title('Steps vs Energy Error')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    benchmark = IntegratorBenchmark()
    benchmark.run_timestep_benchmark()
    benchmark.run_stepcount_benchmark()
    benchmark.plot_results()