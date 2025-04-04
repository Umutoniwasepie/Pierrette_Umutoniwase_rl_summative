# environment/rendering.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image
import imageio

class RuralSolarRenderer:
    def __init__(self, env, window_size=608):
        self.env = env
        self.window_size = window_size
        self.cell_size = window_size // (env.grid_size + 1)

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Create a window (visible for interactivity)
        self.window = glfw.create_window(window_size, window_size, "Solar Farm - Enhanced Visualization", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)

        # Set up OpenGL
        glClearColor(0.9, 0.9, 0.9, 1.0)  # Light gray background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Load robot texture
        self.robot_texture = self._load_texture("visualizations/robot.png")

        # Colors for different states
        self.colors = {
            'panel_healthy': (0.2, 0.8, 0.2, 1.0),  # Green
            'panel_dusty': (1.0, 1.0, 0.0, 0.7),    # Bright yellow with transparency
            'panel_damaged': (1.0, 0.0, 0.0, 1.0),  # Bright red
            'battery_high': (0.2, 0.2, 0.8, 1.0),   # Blue
            'battery_low': (0.8, 0.2, 0.8, 1.0),    # Purple
            'weather': [
                (0.7, 0.9, 1.0, 1.0),  # Clear (light blue)
                (0.8, 0.8, 0.8, 0.5),  # Cloudy (gray overlay)
                (0.9, 0.8, 0.6, 0.5)   # Dusty (sandy overlay)
            ],
            'affected_border': (0.0, 0.0, 0.0, 1.0)  # Black border for affected panels
        }

        # State for interactivity
        self.show_legend = True
        self.paused = False

    def _load_texture(self, filename):
        """Load a texture from a file, with a fallback if the file is missing."""
        try:
            img = Image.open(filename).convert("RGBA")
            img_data = np.array(img, dtype=np.uint8)
        except:
            # Fallback: create a simple colored texture
            img_data = np.zeros((64, 64, 4), dtype=np.uint8)
            img_data[:, :, 0:3] = [100, 100, 255]  # Blue
            img_data[:, :, 3] = 255  # Alpha

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_data.shape[1], img_data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture

    def _draw_square(self, x, y, size, color):
        """Draw a square at (x, y) with given size and color."""
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + size, y)
        glVertex2f(x + size, y + size)
        glVertex2f(x, y + size)
        glEnd()

    def _draw_rectangle(self, x, y, width, height, color):
        """Draw a rectangle at (x, y) with given width, height, and color."""
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()

    def _draw_affected_border(self, x, y, size):
        """Draw a border around an affected panel."""
        glColor4f(*self.colors['affected_border'])
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + size, y)
        glVertex2f(x + size, y + size)
        glVertex2f(x, y + size)
        glEnd()

    def _draw_textured_quad(self, x, y, size, texture):
        """Draw a textured quad (e.g., for the robot)."""
        glBindTexture(GL_TEXTURE_2D, texture)
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Neutral color to let texture show
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + size, y)
        glTexCoord2f(1, 1); glVertex2f(x + size, y + size)
        glTexCoord2f(0, 1); glVertex2f(x, y + size)
        glEnd()

    def _draw_weather_background(self):
        """Draw a background indicating the weather state."""
        weather_color = self.colors['weather'][self.env.weather]
        self._draw_rectangle(0, 0, self.window_size, self.window_size, weather_color)

    def _draw_legend(self):
        """Draw a legend explaining the colors."""
        if not self.show_legend:
            return

        legend_x = 10
        legend_y = 10
        legend_width = 150
        legend_height = 120

        # Legend background
        self._draw_rectangle(legend_x, legend_y, legend_width, legend_height, (1.0, 1.0, 1.0, 0.9))

        # Legend items
        items = [
            ("Healthy", self.colors['panel_healthy']),
            ("Dusty", self.colors['panel_dusty']),
            ("Damaged", self.colors['panel_damaged']),
            ("Battery High", self.colors['battery_high']),
            ("Battery Low", self.colors['battery_low'])
        ]

        for i, (label, color) in enumerate(items):
            # Color swatch
            swatch_x = legend_x + 10
            swatch_y = legend_y + 10 + i * 20
            self._draw_rectangle(swatch_x, swatch_y, 20, 15, color)

    def _draw_status_bar(self):
        """Draw a status bar showing robot energy and total power."""
        bar_x = self.window_size - 160
        bar_y = 10
        bar_width = 150
        bar_height = 60

        # Background
        self._draw_rectangle(bar_x, bar_y, bar_width, bar_height, (1.0, 1.0, 1.0, 0.9))

        # Robot energy bar (normalized to max energy of 5.0)
        energy_width = 120 * (self.env.robot_energy / 5.0)
        self._draw_rectangle(bar_x + 20, bar_y + 10, energy_width, 15, (0.2, 0.8, 0.2, 1.0))

        # Total power bar
        power_width = 120 * self.env.total_power
        self._draw_rectangle(bar_x + 20, bar_y + 35, power_width, 15, (0.2, 0.2, 0.8, 1.0))

    def render_static(self, output_path="solar_farm_static.png"):
        """Render a static image of the environment."""
        # Clear all buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.window_size, 0, self.window_size, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw weather background
        self._draw_weather_background()

        # Draw solar panels
        margin = self.cell_size // 2
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                x = margin + j * self.cell_size
                y = margin + i * self.cell_size
                size = self.cell_size - 10

                # Color based on panel state
                health = self.env.panel_health[i, j]
                dust = self.env.panel_dust[i, j]
                is_affected = False
                if health < 0.7:
                    color = self.colors['panel_damaged']
                    is_affected = True
                elif dust > 0.3:
                    color = self.colors['panel_dusty']
                    is_affected = True
                else:
                    color = self.colors['panel_healthy']

                self._draw_square(x, y, size, color)

                if is_affected:
                    self._draw_affected_border(x, y, size)

                batt_height = (size - 20) * self.env.battery_efficiency[i, j]
                batt_color = self.colors['battery_high'] if self.env.battery_efficiency[i, j] > 0.7 else self.colors['battery_low']
                self._draw_rectangle(x + size - 15, y + 5, 10, batt_height, batt_color)
                glColor4f(0.3, 0.3, 0.3, 1.0)
                glLineWidth(1.0)
                glBegin(GL_LINE_LOOP)
                glVertex2f(x + size - 15, y + 5)
                glVertex2f(x + size - 5, y + 5)
                glVertex2f(x + size - 5, y + 5 + (size - 20))
                glVertex2f(x + size - 15, y + 5 + (size - 20))
                glEnd()

        # Draw robot
        robot_x = margin + self.env.robot_pos[1] * self.cell_size
        robot_y = margin + self.env.robot_pos[0] * self.cell_size
        robot_size = self.cell_size - 10
        self._draw_textured_quad(robot_x, robot_y, robot_size, self.robot_texture)

        # Draw legend and status bar
        self._draw_legend()
        self._draw_status_bar()

        # Read the frame buffer and save as image
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.window_size, self.window_size, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (self.window_size, self.window_size), data)
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if output_path:
            image.save(output_path)
            print(f"Static image saved to {output_path}")
        return image

    def render_dynamic(self, episode_func=None, max_steps=100):
        """Render the environment dynamically during an episode.

        Args:
            episode_func: A function that takes the current state and returns an action (optional for static rendering).
            max_steps: Maximum number of steps to render.
        """
        state = self.env._get_obs()  # Get the current state without resetting
        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < max_steps and not glfw.window_should_close(self.window):
            # Clear all buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set up orthographic projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, self.window_size, 0, self.window_size, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Draw the scene
            self._draw_weather_background()
            margin = self.cell_size // 2
            for i in range(self.env.grid_size):
                for j in range(self.env.grid_size):
                    x = margin + j * self.cell_size
                    y = margin + i * self.cell_size
                    size = self.cell_size - 10

                    health = self.env.panel_health[i, j]
                    dust = self.env.panel_dust[i, j]
                    is_affected = False
                    if health < 0.7:
                        color = self.colors['panel_damaged']
                        is_affected = True
                    elif dust > 0.3:
                        color = self.colors['panel_dusty']
                        is_affected = True
                    else:
                        color = self.colors['panel_healthy']

                    self._draw_square(x, y, size, color)

                    if is_affected:
                        self._draw_affected_border(x, y, size)

                    batt_height = (size - 20) * self.env.battery_efficiency[i, j]
                    batt_color = self.colors['battery_high'] if self.env.battery_efficiency[i, j] > 0.7 else self.colors['battery_low']
                    self._draw_rectangle(x + size - 15, y + 5, 10, batt_height, batt_color)
                    glColor4f(0.3, 0.3, 0.3, 1.0)
                    glLineWidth(1.0)
                    glBegin(GL_LINE_LOOP)
                    glVertex2f(x + size - 15, y + 5)
                    glVertex2f(x + size - 5, y + 5)
                    glVertex2f(x + size - 5, y + 5 + (size - 20))
                    glVertex2f(x + size - 15, y + 5 + (size - 20))
                    glEnd()

            # Draw robot
            robot_x = margin + self.env.robot_pos[1] * self.cell_size
            robot_y = margin + self.env.robot_pos[0] * self.cell_size
            robot_size = self.cell_size - 10
            self._draw_textured_quad(robot_x, robot_y, robot_size, self.robot_texture)

            # Draw legend and status bar
            self._draw_legend()
            self._draw_status_bar()

            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            # Update the environment if not paused and episode_func is provided
            if not self.paused and episode_func is not None:
                action = episode_func(state)
                state, _, done, truncated, _ = self.env.step(action)
                step += 1

            # Control frame rate (approximately 10 FPS)
            glfw.wait_events_timeout(1.0 / 10.0)

        return state, done, truncated

    def create_gif(self, output_path="solar_farm.gif", num_frames=50, agent=None):
        """Create a GIF using a heuristic policy to ensure movement."""
        frames = []
        state, _ = self.env.reset()

        # Force some panels to be dusty or damaged to give the heuristic policy a target
        self.env.panel_dust[1, 1] = 0.5  # Make panel at (1,1) dusty
        self.env.panel_health[2, 2] = 0.5  # Make panel at (2,2) damaged
        self.env.weather = 1  # Force cloudy weather to increase dust accumulation

        done = False
        truncated = False
        step = 0

        # Heuristic policy: Move towards the nearest dusty or damaged panel
        def heuristic_policy(state):
            robot_pos = self.env.robot_pos
            min_distance = float('inf')
            target_pos = None

            # Find the nearest dusty or damaged panel
            for i in range(self.env.grid_size):
                for j in range(self.env.grid_size):
                    dust = self.env.panel_dust[i, j]
                    health = self.env.panel_health[i, j]
                    if dust > 0.3 or health < 0.7:
                        distance = np.linalg.norm(np.array(robot_pos) - np.array([i, j]))
                        print(f"Panel [{i}, {j}] - Dust: {dust:.2f}, Health: {health:.2f}, Distance: {distance:.2f}")
                        if distance < min_distance:
                            min_distance = distance
                            target_pos = [i, j]

            # If no affected panels, perform maintenance or inspect
            if target_pos is None:
                print("No affected panels found.")
                x, y = robot_pos
                if self.env.panel_dust[x, y] > 0.3 or self.env.panel_health[x, y] < 0.7:
                    return 5  # Maintain
                return 4  # Inspect

            # Move towards the target panel
            print(f"Target panel: {target_pos}, Min Distance: {min_distance:.2f}")
            dx = target_pos[0] - robot_pos[0]
            dy = target_pos[1] - robot_pos[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    return 2  # Down
                else:
                    return 3  # Up
            else:
                if dy > 0:
                    return 0  # Right
                else:
                    return 1  # Left

        while not (done or truncated) and step < num_frames and not glfw.window_should_close(self.window):
            #Check loop condition states
            print(f"GIF Step {step} - Before Action - Done: {done}, Truncated: {truncated}, Window Should Close: {glfw.window_should_close(self.window)}")

            # Use the heuristic policy to select actions
            action = heuristic_policy(state)
            print(f"GIF Step {step} - Action: {action}, Robot Pos Before: {self.env.robot_pos}, Energy: {self.env.robot_energy}")

            # Step the environment
            state, reward, done, truncated, info = self.env.step(action)
            print(f"GIF Step {step} - After Step - Robot Pos: {self.env.robot_pos}, Energy: {self.env.robot_energy}, Done: {done}, Truncated: {truncated}")
            #action = heuristic_policy(state)
            #print(f"GIF Action: {action}, Robot Pos: {self.env.robot_pos}")

            # Step the environment
            #state, _, done, truncated, _ = self.env.step(action)

            # Render frame
            image = self.render_static(output_path=None)
            frames.append(np.array(image))
            step += 1

            # Update the display in real-time
            #self.render_dynamic(episode_func=None)  # Render without stepping the environment again

        # Save GIF
        imageio.mimsave(output_path, frames, fps=10)
        print(f"GIF saved to {output_path}")

    def key_callback(self, window, key, scancode, action, mods):
        """Handle key events for interactivity."""
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_L:  # Toggle legend
            self.show_legend = not self.show_legend
        elif key == glfw.KEY_P:  # Pause/unpause
            self.paused = not self.paused

    def close(self):
        """Clean up resources."""
        glDeleteTextures([self.robot_texture])
        glfw.terminate()