from collections import deque
from matplotlib import pyplot as plt
import pygame
import numpy as np
from enum import Enum
from dataclasses import dataclass
import math

DT = 1/60

class IntegrationMethod(Enum):
    RK4 = "rk4"
    VERLET = "verlet"
    EULER = "euler"

@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    mass: float
    
class DoublePendulum:
    def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.l1, self.l2 = l1, l2
        self.m1, self.m2 = m1, m2
        self.g = g
        
        # Initial conditions (angles and angular velocities)
        self.theta1 = np.pi/2
        self.theta2 = np.pi/2
        self.omega1 = 0.0
        self.omega2 = 0.0
        
        self.old_theta1 = self.theta1 - self.omega1 * DT
        self.old_theta2 = self.theta2 - self.omega2 * DT
        
    def calculate_energy(self):
        # Kinetic energy
        v1x = self.l1 * self.omega1 * np.cos(self.theta1)
        v1y = self.l1 * self.omega1 * np.sin(self.theta1)
        v2x = v1x + self.l2 * self.omega2 * np.cos(self.theta2)
        v2y = v1y + self.l2 * self.omega2 * np.sin(self.theta2)
            
        K1 = 0.5 * self.m1 * (v1x**2 + v1y**2)
        K2 = 0.5 * self.m2 * (v2x**2 + v2y**2)
            
        # Potential energy
        y1 = -self.l1 * np.cos(self.theta1)
        y2 = y1 - self.l2 * np.cos(self.theta2)
        U1 = self.m1 * self.g * (y1 + self.l1)  # Reference at height of pivot
        U2 = self.m2 * self.g * (y2 + self.l1 + self.l2)
            
        return K1 + K2 + U1 + U2        

    def derivatives(self, state):
        theta1, theta2, omega1, omega2 = state
        
        # Equations of motion for double pendulum
        delta = theta2 - theta1
        den = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * np.cos(delta) * np.cos(delta)
        
        theta1_dot = omega1
        theta2_dot = omega2
        
        omega1_dot = ((self.m2 * self.l1 * omega1 * omega1 * np.sin(delta) * np.cos(delta)
                      + self.m2 * self.g * np.sin(theta2) * np.cos(delta)
                      + self.m2 * self.l2 * omega2 * omega2 * np.sin(delta)
                      - (self.m1 + self.m2) * self.g * np.sin(theta1)) / den)
        
        omega2_dot = ((-self.m2 * self.l2 * omega2 * omega2 * np.sin(delta) * np.cos(delta)
                      + (self.m1 + self.m2) * (self.g * np.sin(theta1) * np.cos(delta)
                      - self.l1 * omega1 * omega1 * np.sin(delta)
                      - self.g * np.sin(theta2))) / den)
        
        return np.array([theta1_dot, theta2_dot, omega1_dot, omega2_dot])

    def rk4_step(self, dt):
        state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + dt/2 * k1)
        k3 = self.derivatives(state + dt/2 * k2)
        k4 = self.derivatives(state + dt * k3)
        
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        self.theta1, self.theta2, self.omega1, self.omega2 = state
    
    # Standard Euler integration
    def euler_step(self, dt):
        state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        derivatives = self.derivatives(state)
        
        # Update positions and velocities using current state
        self.theta1 += derivatives[0] * dt
        self.theta2 += derivatives[1] * dt
        
        self.omega1 += derivatives[2] * dt
        self.omega2 += derivatives[3] * dt
    
    def verlet_step(self, dt):
        # Classic Verlet integration
        state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        acc = self.derivatives(state)[2:]  # Only angular accelerations
        
        # Store current positions
        next_theta1 = 2 * self.theta1 - self.old_theta1 + acc[0] * dt * dt
        next_theta2 = 2 * self.theta2 - self.old_theta2 + acc[1] * dt * dt
        
        # Update previous and current positions
        self.old_theta1 = self.theta1
        self.old_theta2 = self.theta2
        self.theta1 = next_theta1
        self.theta2 = next_theta2
        
        # Update velocities (only for visualization)
        self.omega1 = (self.theta1 - self.old_theta1) / dt
        self.omega2 = (self.theta2 - self.old_theta2) / dt

    def get_positions(self):
        x1 = self.l1 * np.sin(self.theta1)
        y1 = -self.l1 * np.cos(self.theta1)
        x2 = x1 + self.l2 * np.sin(self.theta2)
        y2 = y1 - self.l2 * np.cos(self.theta2)
        return (x1, y1), (x2, y2)
    
class GravitationalSystem:
    def __init__(self, m1=1.0, m2=1.0, G=1.0):
        self.m1, self.m2 = m1, m2
        self.G = G
        
        # Initial conditions (position and velocity vectors)
        self.pos = np.array([1.0, 0.0])  # Initial position (1 unit from origin)
        self.vel = np.array([0.0, 1.0])  # Initial velocity for circular orbit
        
        # For Verlet integration
        self.old_pos = self.pos - self.vel * DT
        
    def calculate_energy(self):
        # Kinetic energy
        K = 0.5 * self.m2 * np.sum(self.vel**2)
        
        # Potential energy (gravitational)
        r = np.sqrt(np.sum(self.pos**2))
        U = -self.G * self.m1 * self.m2 / r
        
        return K + U

    def derivatives(self, state):
        pos = state[:2]
        vel = state[2:]
        
        # Calculate acceleration due to gravity
        r = np.sqrt(np.sum(pos**2))
        acc = -self.G * self.m1 * pos / (r**3)
        
        return np.array([*vel, *acc])

    def rk4_step(self, dt):
        state = np.array([*self.pos, *self.vel])
        
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + dt/2 * k1)
        k3 = self.derivatives(state + dt/2 * k2)
        k4 = self.derivatives(state + dt * k3)
        
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        self.pos = state[:2]
        self.vel = state[2:]
    
    def euler_step(self, dt):
        state = np.array([*self.pos, *self.vel])
        derivatives = self.derivatives(state)
        
        # Update velocity
        self.vel += derivatives[2:] * dt
        
        # Update position using new velocity
        self.pos += self.vel * dt
    
    def verlet_step(self, dt):
        # Calculate acceleration
        r = np.sqrt(np.sum(self.pos**2))
        acc = -self.G * self.m1 * self.pos / (r**3)
        
        # Update position
        next_pos = 2 * self.pos - self.old_pos + acc * dt * dt
        
        # Update previous and current positions
        self.old_pos = self.pos.copy()
        self.pos = next_pos
        
        # Update velocity (for visualization)
        self.vel = (self.pos - self.old_pos) / dt

    def get_positions(self):
        # Return positions of both bodies (central body at origin)
        return (0, 0), (self.pos[0], self.pos[1])

class Simulator:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Double Pendulum Simulator")
        
        self.clock = pygame.time.Clock()
        self.scale = 200  # Pixels per meter
        self.pendulum = DoublePendulum()
        self.integration_method = IntegrationMethod.VERLET
        self.dt = DT
        
        # Energy tracking
        self.energy_history = deque(maxlen=500)  # Store last 500 points
        self.time_history = deque(maxlen=500)
        self.current_time = 0.0
        
        # Create figure for energy plot
        self.fig = plt.figure(figsize=(8, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Energy (J)')
        self.ax.grid(True)
        
        # pre-created plot surface with fixed size
        self.plot_surface = pygame.Surface((self.width, self.height//3))
        self.energy_update_counter = 0
        
    def update_energy_plot(self):
        self.ax.clear()
        self.ax.plot(list(self.time_history), list(self.energy_history), 'b-')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Energy (J)')
        self.ax.grid(True)
        
        # draw to pre-created surface
        self.fig.canvas.draw()
        temp_surface = pygame.image.frombuffer(
            self.fig.canvas.buffer_rgba(),
            self.fig.canvas.get_width_height(),
            'RGBA')
        # when scaling, store to surface
        pygame.transform.scale(
            temp_surface, 
            (self.width, self.height//3), 
            self.plot_surface
        )
        
    def to_screen_coords(self, x, y):
        return (int(self.width/2 + x * self.scale),
                int(self.height/2 + y * self.scale))
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if (self.integration_method == "verlet"):
                            self.integration_method = IntegrationMethod.EULER
                        elif (self.integration_method == "rk4"):
                            self.integration_method = IntegrationMethod.VERLET
                        else:
                            self.integration_method = IntegrationMethod.RK4
                            
            self.integration_method = IntegrationMethod.VERLET
            
            # Update physics and energy tracking
            if self.integration_method == IntegrationMethod.RK4:
                self.pendulum.rk4_step(self.dt)
            elif self.integration_method == IntegrationMethod.VERLET:
                self.pendulum.verlet_step(self.dt)
            else:
                self.pendulum.euler_step(self.dt)
            
            self.current_time += self.dt
            self.energy_history.append(self.pendulum.calculate_energy())
            self.time_history.append(self.current_time)
            
            # Draw pendulum
            self.screen.fill((255, 255, 255))
            
            # Draw pendulum surface (top 2/3)
            pendulum_surface = pygame.Surface((self.width, self.height * 2//3))
            pendulum_surface.fill((255, 255, 255))
            
            # Draw pendulum on pendulum surface
            (x1, y1), (x2, y2) = self.pendulum.get_positions()
            origin = self.to_screen_coords(0, 0)
            p1 = self.to_screen_coords(x1, -y1)
            p2 = self.to_screen_coords(x2, -y2)
            
            pygame.draw.line(pendulum_surface, (0, 0, 0), origin, p1, 2)
            pygame.draw.line(pendulum_surface, (0, 0, 0), p1, p2, 2)
            pygame.draw.circle(pendulum_surface, (255, 0, 0), p1, 10)
            pygame.draw.circle(pendulum_surface, (255, 0, 0), p2, 10)
            
            self.screen.blit(pendulum_surface, (0, 0))
            
            # Update energy plot every 10 frames
            self.energy_update_counter += 1
            if self.energy_update_counter >= 10:
                self.update_energy_plot()
                self.energy_update_counter = 0
            
            # Always blit the plot surface (bottom 1/3)
            self.screen.blit(self.plot_surface, (0, self.height * 2//3))
            
            # Draw text overlays
            font = pygame.font.Font(None, 36)
            method_text = font.render(
                f"Integration: {self.integration_method.value}", 
                True, (0, 0, 0)
            )
            energy_text = font.render(
                f"Energy: {self.energy_history[-1]:.3f} J", 
                True, (0, 0, 0)
            )
            self.screen.blit(method_text, (10, 10))
            self.screen.blit(energy_text, (10, 50))
            
            pygame.display.flip()
            self.clock.tick(120)
        
        pygame.quit()

if __name__ == "__main__":
    sim = Simulator()
    sim.run()