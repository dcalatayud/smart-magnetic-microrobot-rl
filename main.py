"""
Magnetic Microrobot Control System with Reinforcement Learning

This code implements a real-time control system for magnetic microrobots using reinforcement learning.
It provides both manual control through a GUI interface and automated control using trained RL models.

Hardware Requirements:
- FLIR camera (uses simple_pyspin)
- DAQ device for magnetic field control (uses uldaq)
- Helmholtz coil setup for 3D magnetic field generation

Author: D. Gonzalez-Calatayud (davidgcalatayud@gmail.com)
"""

import os
import cv2
import time
import sys
import datetime
import threading
import queue
import customtkinter as ctk
from simple_pyspin import Camera
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from uldaq import get_daq_device_inventory, DaqDevice, create_float_buffer
from uldaq import Range, ScanOption, AOutScanFlag, InterfaceType
from PIL import Image
import imageio


class OutputRedirector:
    def __init__(self, log_dir="logs"):
        self.terminal = sys.stdout
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"roller_{timestamp}.log")
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if message.startswith('\r'):
            return
        # Only add timestamp for new lines
        if message.startswith('\n') or self.log_file.tell() == 0:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-4]  # Keep 2 decimal places
            self.log_file.write(f"[{timestamp}] {message.lstrip()}")
            if not message.endswith('\n'):  # Add newline if message doesn't end with one
                self.log_file.write('\n')
        else:
            self.log_file.write(message)
            if not message.endswith('\n'):  # Add newline after non-newline messages
                self.log_file.write('\n')
        self.log_file.flush()


    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def cleanup(self):
        self.log_file.close()
        sys.stdout = self.terminal


class RollerGym(gym.Env):
    def __init__(self, RollerGUI):
        # RollerGUI
        self.RollerGUI = RollerGUI
        self.RollerGUI.episode_count = 0
        self.recording_filename = None


        # Magnetic-field actuation
        self.daq_device = None
        self.freq = RollerGUI.freq
        self.freq_max = RollerGUI.freq_max
        self.points_channel = 1000
        self.channels = 3
        self.actuation_time = 1
        if self.freq == 1:
            self.A0 = 2
        else:
            self.A0 = 1
        self.Ay = 0.62
        self.Az = 0.48

        # Bounds for the roller
        self.margins = 0.05
        self.x_min = self.margins * 1920
        self.x_max = (1 - self.margins) * 1920
        self.y_min = self.margins * 1440
        self.y_max = (1 - self.margins) * 1440
        self.x0 = self.RollerGUI.x0
        self.y0 = self.RollerGUI.y0
        self.r0_threshold = self.RollerGUI.r0_threshold
        self.x_goal = self.RollerGUI.x_goal

        # RL Space
        self.max_steps = self.RollerGUI.max_steps
        self.A_max = 1
        self.V_max = 25 # Estimated maximum velocity
        action_high = np.array([self.A_max, self.A_max, self.A_max, np.pi, np.pi], dtype=np.float32)
        obs_high = np.array([self.V_max, self.V_max], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

    def step(self, action):
        self.n_step += 1
        self.last_x, self.last_y = self.x, self.y
        action = np.clip(action, self.action_space.low, self.action_space.high, dtype=np.float32)
        self.take_action(action)
        self.x, self.y = self.get_position()
        
        # Calculate velocities
        self.vx = (self.x - self.last_x)/self.actuation_time
        self.vy = (self.y - self.last_y)/self.actuation_time
        self.state = np.array([self.vx, self.vy], dtype=np.float32)
        
        # Calculate reward
        dx = self.x - self.last_x
        dy = self.y - self.last_y
        reward = 1*dx - 0.5*abs(dy) - 1

        terminated = self.x >= self.x_goal
        if terminated:
            print('Mission accomplished!')
            reward +=1000

    
        truncated = self.n_step >= self.max_steps
        if not self.check_bounds():
            truncated = True
            print('Roller out of bounds')

        if terminated or truncated:
            if self.recording_filename is not None:
                self.RollerGUI.stop_recording(self.recording_filename)
            else:
                self.RollerGUI.stop_recording()

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.daq_device is None:
            self.get_daq_device()
        
        # If no options are passed, normal roller training. Else, resetting of the episode
        if options is None:
            self.RollerGUI.episode_count += 1
            print(f'Episode count: {self.RollerGUI.episode_count:03d}')
            self.reset_position()
            self.recording_filename = None
        else:
            self.RollerGUI.episode_count = 0
            self.recording_filename = options

    
        self.n_step = 0
        self.vx, self.vy = 0, 0
        self.x, self.y = self.get_position()

        # if self.RollerGUI.is_recording:
        #     self.RollerGUI.stop_recording()

        if self.RollerGUI.recording_activated:
            self.RollerGUI.start_recording()
        
        obs = np.array([self.vx, self.vy], dtype=np.float32)
        return obs, {}

    def get_position(self):
        return self.RollerGUI.get_roller()
    
    def manual_move(self, direction, freq_scale = 1):
        self.freq = self.freq * freq_scale

        if direction == 'up':
            action = np.array([1, 0, 1, np.pi/2, 0])
        elif direction == 'down':
            action = np.array([1, 0, 1, -np.pi/2, 0])
        elif direction == 'left':
            action = np.array([0, 1, 1, 0, np.pi/2])
        elif direction == 'right':
            action = np.array([0, 1, 1, 0, -np.pi/2])
        else:
            print(f"Invalid direction: {direction}")
            return

        # _, reward, _, _, _ = self.step(action)
        # self.render(action, reward)
        self.take_action(action)
        self.freq = self.freq / freq_scale


    def directed_move(self, angle_deg=0):
        angle_rad = angle_deg * np.pi/180
        # Calculate phase angles
        phix = -angle_rad
        phiy = -(np.pi/2 - angle_rad)

        Mx = 1
        My = 1
        Mz = 1
        action = np.array([Mx, My, Mz, phix, phiy])
        self.take_action(action)

    def check_bounds(self):
        x_in_bounds = self.x_min <= self.x <= self.x_max
        y_in_bounds = self.y_min <= self.y <= self.y_max
        return x_in_bounds and y_in_bounds
    
    def take_action(self, action):
        signals = self.generate_signals(action)
        channels, points_channel = np.shape(signals)
        rate = points_channel
        ao_device = self.daq_device.get_ao_device()
        self.daq_device.connect(connection_code=0)
        array = create_float_buffer(channels, points_channel)
        for i in range(points_channel):
            for j in range(channels):
                array[i*channels+j] = signals[j,i]
        array_np = np.ctypeslib.as_array(array).copy()
        ao_device.a_out_scan(low_chan=0, high_chan=channels-1, samples_per_channel=points_channel, analog_range = Range.BIP10VOLTS,
                        rate=rate, options = ScanOption.CONTINUOUS, flags = AOutScanFlag.DEFAULT, data = array)
        time.sleep(self.actuation_time)
        self.daq_device.disconnect()

    def get_daq_device(self):
        devices = get_daq_device_inventory(InterfaceType.ANY)
        if len(devices) == 0:
            raise Exception('Error: No DAQ devices found')
        self.daq_device = DaqDevice(devices[0])
    
    def generate_signals(self, action):
        t = self.freq*np.linspace(0,1,self.points_channel,endpoint=False, dtype=np.float64)
        Ax, Ay, Az, phix, phiy = action
        sx = self.A0 * Ax * np.sin(2*np.pi*t + phix)
        sy = self.A0 * self.Ay * Ay * np.sin(2*np.pi*t + phiy)
        sz = self.A0 * self.Az * Az * np.sin(2*np.pi*t)
        signals = np.array([sx, sy, sz])
        return signals
    
    def reset_position(self, freq_scale = 1):
        A0_old = self.A0 
        self.A0 = 1
        freq_old = self.freq
        self.RollerGUI.r0_show = True
        self.freq = self.freq * freq_scale
        max_attempts = 200
        attempts = 0
        distance_old = 0
        n_lost = 0
        while attempts < max_attempts and n_lost < 3:
            current_x, current_y = self.get_position()
            dx = self.x0 - current_x
            dy = self.y0 - current_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Add condition to leave loop if roller is lost
            if distance == distance_old:
                n_lost +=1
            else:
                n_lost = 0

            if distance < self.r0_threshold:
                print(f"Roller reset to initial position: ({current_x}, {current_y})")
                self.freq = self.freq / freq_scale
                self.A0 = A0_old
                return True
            
            # Maximumm speed when far away from r0
            if distance > self.r0_threshold*5:
                self.freq = self.freq_max
            else:
                self.freq = freq_old * freq_scale
            
            angle = np.arctan2(dy, dx)
            angle_deg =  angle*180/np.pi
            print(f"Attempt {attempts}: dr={distance:.1f}, dx={dx}, dy={dy}, angle_deg={angle_deg:.2f}")
            self.directed_move(angle_deg)
            attempts += 1
            distance_old = distance
        
        print("Failed to reset roller position after maximum attempts")
        self.freq = self.freq / freq_scale
        self.A0 = A0_old
        return False    

    def render(self, action=None, reward=0):
        if action is not None:
            Ax, Ay, Az, phix, phiy = action
        else:
            Ax, Ay, Az, phix, phiy = 0., 0., 0., 0.
        print(f'Action: Ax={Ax:.2f}, Ay={Ay:.2f}, Az={Az:.2f}, phix={phix:.2f}, phiy={phiy:.2f} | '
            #   f'Reward: {reward:.2f} | '
              f'Position: ({self.x:.2f}, {self.y:.2f}) | '
              f'Velocity: ({self.vx:.2f}, {self.vy:.2f}) | '
            #   f'Step: {self.n_step}'
              )


class RollerGUI:
    def __init__(self, master):
        self.lock = threading.Lock()
        self.master = master
        self.master.title("Roller Controller")
        self.master.geometry("1000x600")

        # Camera settings
        self.fps = 10
        self.is_recording = False
        self.recording_activated = False
        self.episode_count = 0
        self.frames = []
        self.frames_save = []
        self.total_episodes = 0
        self.camera_width = 1920
        self.camera_height = 1440 # 1200 for small camera
        self.camera_running = True
        self.exit_camera = False
        self.exit_flag = False
        self.exit_event = threading.Event()
        self.save_folder = "episodes"

        # Roller position settings
        self.threshold = 100
        self.min_area = 100
        self.display_roller = False
        self.roller = None
        self.search_radius = 40
        self.search_rollers_flag = True
        self.frame_search = None
        self.background_image = None
        self.subtract_background = False

        # RL settings
        self.x0 = self.camera_width * 0.25
        self.y0 = self.camera_height * 0.5
        self.x_goal = self.camera_width * 0.75
        self.r0_show = False
        self.r0_threshold = 20
        self.freq = 4
        self.A0 = 1
        self.freq_max = 20
        self.max_steps = 800

        self.n_frame = 0
        self.show_frame = True
        self.T = 0

        self.master.after(100, self.update_timer)
        self.save_queue = queue.Queue()
        self.saving_complete = threading.Event()
        
        # Create episodes directory if it doesn't exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # GUI components
        self.setup_gui()
        self.start_time = None

        self.is_manual_moving = False
        # Threading operations
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        self.save_thread = threading.Thread(target=self.saving_loop)
        self.save_thread.daemon = True
        self.save_thread.start()

    def setup_gui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.master)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Left frame for camera feed
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(self.left_frame, text="")
        self.camera_label.pack(expand=True)

        # Right frame for controls and info
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Exit button (at the top and red)
        self.exit_button = ctk.CTkButton(self.right_frame, text="Exit", command=self.exit_program, fg_color="red", hover_color="darkred")
        self.exit_button.pack(pady=(0, 20))

        self.timer_label = ctk.CTkLabel(self.right_frame, text="Time: 00:00:00")
        self.timer_label.pack(pady=5)

        # New frame for manual roller position input and roller search
        self.manual_position_frame = ctk.CTkFrame(self.right_frame)
        self.manual_position_frame.pack(pady=10)

        # Set Background button
        self.set_background_button = ctk.CTkButton(self.manual_position_frame, text="Set Background", command=self.set_background, fg_color="blue", hover_color="darkblue")
        self.set_background_button.grid(row=0, column=0, columnspan=4, padx=5, pady=(5, 10))

        # Threshold input
        self.threshold_label = ctk.CTkLabel(self.manual_position_frame, text="Th.:")
        self.threshold_label.grid(row=1, column=0, padx=5, pady=5)
        self.threshold_entry = ctk.CTkEntry(self.manual_position_frame, width=50)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=5)
        self.threshold_entry.insert(0, "125")  # Default value

        # Min Area input
        self.min_area_label = ctk.CTkLabel(self.manual_position_frame, text="Min Area:")
        self.min_area_label.grid(row=1, column=2, padx=5, pady=5)
        self.min_area_entry = ctk.CTkEntry(self.manual_position_frame, width=50)
        self.min_area_entry.grid(row=1, column=3, padx=5, pady=5)
        self.min_area_entry.insert(0, "100")  # Default value

        # Search Rollers button
        self.search_rollers_button = ctk.CTkButton(self.manual_position_frame, text="Search Rollers", command=self.trigger_roller_search, fg_color="green", hover_color="darkgreen")
        self.search_rollers_button.grid(row=2, column=0, columnspan=4, padx=5, pady=(5, 10))

        # X position input
        self.x_label = ctk.CTkLabel(self.manual_position_frame, text="X:")
        self.x_label.grid(row=3, column=0, padx=5, pady=5)
        self.x_entry = ctk.CTkEntry(self.manual_position_frame, width=50)
        self.x_entry.grid(row=3, column=1, padx=5, pady=5)

        # Y position input
        self.y_label = ctk.CTkLabel(self.manual_position_frame, text="Y:")
        self.y_label.grid(row=3, column=2, padx=5, pady=5)
        self.y_entry = ctk.CTkEntry(self.manual_position_frame, width=50)
        self.y_entry.grid(row=3, column=3, padx=5, pady=5)

        # Set Position button
        self.set_position_button = ctk.CTkButton(self.manual_position_frame, text="Set Position", command=self.set_manual_position)
        self.set_position_button.grid(row=4, column=0, columnspan=4, pady=(5, 0))

        # Update position of the roller
        self.position_label = ctk.CTkLabel(self.manual_position_frame, text="Roller: (-, -)")
        self.position_label.grid(row=5, column=0, columnspan=4, pady=(5, 0))

        # New frame for RL training button
        self.rl_frame = ctk.CTkFrame(self.right_frame)
        self.rl_frame.pack(pady=10)

        # Start Training button
        self.start_training_button = ctk.CTkButton(self.rl_frame, text="Start Training", command=self.start_rl_training, fg_color="orange", hover_color="darkorange")
        self.start_training_button.pack(pady=5)

        self.toggle_recording_button = ctk.CTkButton(self.rl_frame, text="Start Recording", command=self.toggle_recording, fg_color="purple", hover_color="#4B0082")
        self.toggle_recording_button.pack(pady=5)

        # New frame for manual movement buttons
        self.manual_move_frame = ctk.CTkFrame(self.right_frame)
        self.manual_move_frame.pack(pady=10)

        # Create a 3x3 grid for compact arrow buttons
        button_size = 30
        arrow_font = ctk.CTkFont(family="Arial", size=16)

        self.up_button = ctk.CTkButton(self.manual_move_frame, text="↑", font=arrow_font, width=button_size, height=button_size, command=lambda: self.manual_move('up'))
        self.up_button.grid(row=0, column=1, padx=1, pady=1)

        self.left_button = ctk.CTkButton(self.manual_move_frame, text="←", font=arrow_font, width=button_size, height=button_size, command=lambda: self.manual_move('left'))
        self.left_button.grid(row=1, column=0, padx=1, pady=1)

        self.r_button = ctk.CTkButton(self.manual_move_frame, text="R", font=arrow_font, width=0.9*button_size, height=0.9*button_size, command=lambda: self.manual_move('R'))
        self.r_button.grid(row=1, column=1, padx=1, pady=1)

        self.right_button = ctk.CTkButton(self.manual_move_frame, text="→", font=arrow_font, width=button_size, height=button_size, command=lambda: self.manual_move('right'))
        self.right_button.grid(row=1, column=2, padx=1, pady=1)

        self.down_button = ctk.CTkButton(self.manual_move_frame, text="↓", font=arrow_font, width=button_size, height=button_size, command=lambda: self.manual_move('down'))
        self.down_button.grid(row=2, column=1, padx=1, pady=1)
    
    def get_roller(self):
        with self.lock:
            return self.roller
    
    def set_roller(self, x, y):
        with self.lock:
            self.roller = [x, y]
    
    def update_timer(self):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
            self.timer_label.configure(text=f"Time: {formatted_time}")
            
        # Update roller position here
        roller = self.get_roller()
        if roller is not None:
            x, y = roller
            position_text = f"Roller: ({x}, {y})"
            if hasattr(self, 'position_label'):
                self.position_label.configure(text=position_text)
            else:
                self.position_label = ctk.CTkLabel(self.manual_position_frame, text=position_text)
                self.position_label.grid(row=4, column=0, columnspan=4, pady=(5, 0))
        
        self.master.after(100, self.update_timer)  # Update every 100ms for smoother updates
    
    def start_rl_training(self):
        self.rl_thread = threading.Thread(target=self.training_loop)
        self.rl_thread.daemon = True
        self.rl_thread.start()
        self.start_training_button.configure(text="Training...", state="disabled", text_color_disabled="white")

##########################################################################################################################################
    def training_loop(self):
        print("Starting RL training...")
        if self.get_roller() is not None:
            self.freq = 16
            print(f'Freq. set to {self.freq} Hz')
            folder_path = f'freq_{self.freq}'
            self.set_save_folder(folder_path)
            env = RollerGym(self)
            env.reset()
            model = SAC(
                "MlpPolicy", 
                env,
                device ='cuda', 
                verbose=1, 
                gamma=0.99, 
                tau=0.01,
                learning_starts=100, # changed to 500
                tensorboard_log=f"./sac_roller_tensorboard/",
                buffer_size=100_000, 
                learning_rate=3e-3,
                batch_size=256, 
                train_freq=1,
                gradient_steps=1, 
                ent_coef='auto',
                seed=69420
            )
            model.learn(total_timesteps=5000, progress_bar=True, log_interval=1)
            self.stop_recording()
            model.save(f'sac_roller_f{self.freq}.zip')
            del model
            
            # Load trained model and run a trajectory with the trained policy
            env.reset_position()
            print('Starting trained policy...')
            model = SAC.load(f'sac_roller_f{self.freq}.zip')
            obs, info = env.reset(options='trained')
            n = 0
            terminated = False
            truncated = False
            actions = []
            n_values = []
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                n += 1 
                actions.append(action)
                n_values.append(n)
            print(f'Trained policy completed with {n} steps')
            self.stop_recording()
            actions = np.array(actions)
            n_values = np.array(n_values).reshape(-1,1)
            actions_with_n = np.hstack((n_values, actions))
            output_path = os.path.join(folder_path, f"trajectories/actions.txt")
            np.savetxt(output_path, actions_with_n,
                    header='n Ax Ay Az phix phiy',
                    fmt='%d %.3f %.3f %.3f %.2f %.2f',
                    comments='')
            
            # Physics-informed policy
            env.reset_position()
            print('Starting physical-informed policy...')
            env.reset(options='physical_informed')
            action = np.array([0, 1, 1, 0, -np.pi/2]) # move right
            n = 0
            terminated = False
            truncated = False
            actions = []
            n_values = []
            while not terminated or truncated:
                obs, reward, terminated, truncated, info = env.step(action)
                n += 1 
                actions.append(action)
                n_values.append(n)           
            print(f'Physical-informed policy completed with {n} steps')
            n_values = np.array(n_values).reshape(-1,1)
            actions_with_n = np.hstack((n_values, actions))
            output_path = os.path.join(folder_path, f"trajectories/actions_PI.txt")
            np.savetxt(output_path, actions_with_n,
                    header='n Ax Ay Az phix phiy',
                    fmt='%d %.3f %.3f %.3f %.2f %.2f',
                    comments='')
            self.stop_recording()
            
            print("RL training completed!")
            # Update the button on the main thread
            self.master.after(0, self.update_training_button)
        else:
            print('Roller postion not set, training not possible')
            self.master.after(0, self.update_training_button)
##########################################################################################################################################

    def manual_move(self, direction):
        if self.roller is None:
            self.trigger_roller_search()
            while self.search_rollers_flag:
                time.sleep(0.1)
            if self.roller is None:
                print("Since no roller was found, manual move cannot be performed")
            return
        if not self.is_manual_moving:
            threading.Thread(target=self._manual_move_thread, args=(direction,), daemon=True).start()
        else:
            print("A manual move is already in progress. Please wait.") 

    def _manual_move_thread(self, direction):
        self.is_manual_moving = True
        try:
            env = RollerGym(self)
            env.reset(options={})
            if direction == 'R':
                env.reset_position(freq_scale=1)
            else:
                for _ in range(10):
                    env.manual_move(direction, freq_scale = 2)
        finally:
            self.is_manual_moving = False

    def update_training_button(self):
        self.start_training_button.configure(text="Start Training", state="normal")
    
    def trigger_roller_search(self):
        # Update threshold and min_area values
        try:
            self.threshold = int(self.threshold_entry.get())
            self.min_area = int(self.min_area_entry.get())
            print(f"Updated threshold to {self.threshold} and min_area to {self.min_area}")
        except ValueError:
            print("Invalid input for threshold or min_area. Using previous values.")

        self.search_rollers_flag = True
        threading.Thread(target=self.search_rollers_thread, daemon=True).start()
    
    def search_rollers_thread(self):
        self.search_rollers_flag = True
        while self.frame_search is None:
            time.sleep(0.1)  # Wait for frame to be available
        
        centroids = self.find_centroids(self.frame_search, search_whole_frame=True)
        print('Roller search triggered')
        
        if len(centroids) > 1:
            print("Found rollers:")
            largest_centroid = max(centroids, key=lambda c: c[2])  # Find centroid with largest area
            for i, (x, y, area) in enumerate(centroids):
                print(f"Roller {i+1}: Position ({x}, {y}) with area: {area}")
            
            x, y, area = largest_centroid
            self.set_roller(x, y)
            self.display_roller = True
            print(f"Roller position automatically set to: x={x}, y={y} (largest area: {area})")
            
        elif len(centroids) == 1:
            print("Found single roller:")
            x, y, area = centroids[0]
            print(f"Roller {1}: Position ({x}, {y}) with area: {area}")
            self.set_roller(x, y)
            self.display_roller = True
            print(f"Roller position automatically set to: x={x}, y={y}")
        else:
            print('No rollers found')
        
        self.frame_search = None  # Reset the frame
        self.search_rollers_flag = False  # Indicate that search is complete

    def set_background(self):
        if self.frame_search is not None:
            self.background_image = self.frame_search
            self.background_image = cv2.bitwise_not(self.background_image)
            self.search_rollers_flag = False
            self.subtract_background = True
            print("Background set successfully")
        else:
            print("No frame available to set as background")
    
    def set_manual_position(self):
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            if 0 <= x < self.camera_width and 0 <= y < self.camera_height:
                self.set_roller(x, y)
                self.display_roller = True
                print(f"Roller position manually set to: x={x}, y={y}")
            else:
                print("Invalid position. Please enter values within the camera frame dimensions.")
        except ValueError:
            print("Invalid input. Please enter integer values for x and y positions.")

    def camera_loop(self):
        while not self.exit_flag:
            with Camera() as cam:
                cam.AcquisitionFrameRateAuto = 'Off'
                cam.AcquisitionFrameRateEnabled = True
                cam.AcquisitionFrameRate = self.fps
                cam.start()
                self.start_time = time.time()
                ix = 0
                while self.camera_running:
                    frame = cam.get_array() # gray array of shape (1200, 1920)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # rgb array of shape (1200, 1920, 3)
                    self.n_frame += 1
                    
                    if self.subtract_background:
                        frame_rgb = cv2.subtract(self.background_image, frame_rgb)
                        frame_rgb = cv2.bitwise_not(frame_rgb)
                    
                    ix += 1
                    if self.is_recording and ix % 2:
                        self.frames.append(frame)
                    

                    if self.exit_camera:
                        print('Exit camera!')
                        cam.stop()
                        break

                    if self.search_rollers_flag:
                        self.frame_search = frame_rgb.copy()
                        # self.search_rollers_flag = False

                    start_time = time.time()
                    if self.display_roller:
                        centroids = self.find_centroids(frame_rgb)
                        frame_rgb = self.show_roller(frame_rgb, centroids)
                        
                    if self.show_frame:
                        cv2.putText(frame_rgb, f"Frame: {self.n_frame:06d}", (1650, 30),  # Position: top-left corner
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1,  # Font scale
                                (0, 0, 0),  # Text color: black
                                2,  # Line thickness
                                cv2.LINE_AA)
                    
                    photo = self.convert_frame_to_photo(frame_rgb)
                    end_time = time.time()
                    self.T += end_time - start_time
                    self.master.after(0, self.update_camera_feed, photo)

                cam.stop()
                

    def find_centroids(self, frame_rgb, search_whole_frame=False):
        grayscale = frame_rgb[:,:,0]
        roller = self.get_roller()
        if not search_whole_frame and roller is not None:
            mask = np.ones(grayscale.shape, dtype=np.uint8) * 255
            cv2.circle(mask, tuple(roller), self.search_radius, 0, -1)
            grayscale = np.where(mask == 255, 255, grayscale)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(grayscale, (15,15), 0)
        blur = grayscale
        # Threshold the image
        inrange = cv2.inRange(blur, 0, self.threshold)

        # Find contours
        contours, _ = cv2.findContours(inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                M = cv2.moments(contour)
                area = M["m00"]
                cX = int(M["m10"] / area)
                cY = int(M["m01"] / area)
                centroids.append([cX, cY, area])
        
        return centroids

    def show_roller(self, frame_rgb, centroids):
        if len(centroids) == 1:
            x, y, _ = centroids[0]
            self.set_roller(x, y)
            color = (0, 255, 0)  # Green color in BGR
            markerType = cv2.MARKER_CROSS
            markerSize = 20
            thickness = 2
            roller = self.get_roller()
            cv2.drawMarker(frame_rgb, tuple(roller), color, markerType, markerSize, thickness)
            # Draw the search area
            cv2.circle(frame_rgb, tuple(roller), self.search_radius, (255, 0, 0), 1)
            # Draw the r0 position. Float r0 will be truncated when passed to cv2
            if self.r0_show:
                start_color = (0, 165, 255)  # Orange color in BGR
                cv2.drawMarker(frame_rgb, (int(self.x_goal), int(self.y0)), start_color, cv2.MARKER_STAR, markerSize, thickness)
                circle_color = (255, 0, 255)  # Purple color in BGR
                cv2.circle(frame_rgb, (int(self.x0), int(self.y0)), self.r0_threshold, circle_color, thickness)
        else:
            pass
        return frame_rgb
          
    def convert_frame_to_photo(self, frame):
        # Resize the image to fit within a 640x480 bounding box while maintaining the original aspect ratio
        aspect_ratio = self.camera_width / self.camera_height
        
        if aspect_ratio > (640/480):
            new_width = 640
            new_height = int(640 / aspect_ratio)
        else:
            new_height = 480
            new_width = int(480 * aspect_ratio)

        frame = cv2.resize(frame, (new_width, new_height))
        image = Image.fromarray(frame)
        return ctk.CTkImage(light_image=image, size=(new_width, new_height))

    def update_camera_feed(self, photo):
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def toggle_recording(self):
        self.recording_activated = not self.recording_activated
        if self.recording_activated:
            self.toggle_recording_button.configure(text="Deactivate Recording")
            print("Recording activated")
        else:
            self.toggle_recording_button.configure(text="Activate Recording")
            print("Recording dectivated")

    def start_recording(self):
        if self.recording_activated:
            self.frames = []
            self.is_recording = True

    def stop_recording(self, filename = None):
        if self.is_recording:
            self.is_recording = False
            if self.frames:
                self.save_episode(filename)
            # self.wait_for_saving()
            print("Recording done!")
    

    def save_episode(self, filename):
        if self.frames:
            self.save_queue.put((self.episode_count, self.frames.copy(), filename))
            print(f"Queued episode {self.episode_count:03d} for saving")
        else:
            print("No frames to save")
        self.frames = []

    def saving_loop(self):
        while not self.exit_flag:
            try:
                # get() removes the item from the queue
                episode, frames, filename_q = self.save_queue.get(timeout=0.1)
                start_time = time.time()
                if filename_q is None:
                    filename = f"{self.save_folder}/episode{episode:03d}.avi"
                else:
                    filename = f"{self.save_folder}/{filename_q}.avi"
                # with imageio.get_writer(filename, fps=self.fps) as writer:
                #     for frame in frames:
                #         writer.append_data(frame)
                height, width = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height), isColor=False)
                for frame in frames:
                    # No need to convert to BGR as frames are already in grayscale
                    out.write(frame)
                out.release()

                n_frames = len(frames)
                file_size_mb = os.path.getsize(filename) / (1024 * 1024)
                print(f"Saved episode {episode:03d}")
                print(f"Video file size: {file_size_mb:.2f} MB, with {n_frames} frames")
                
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Time taken to save episode: {execution_time:.2f} seconds")
                # Clear the reference to frames in saving thread
                frames = None
                # print('Frames = None')
                self.save_queue.task_done()
            except queue.Empty:
                continue

    def set_save_folder(self, folder_name):
        self.save_folder = folder_name
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        print(f"Save folder set to: {self.save_folder}")
    
    def wait_for_saving(self):
        self.save_queue.join()
        self.saving_complete.set() 
    
    def exit_program(self):
        self.camera_running = False
        self.exit_flag = True
        self.exit_camera = True
        self.wait_for_saving()
        self.master.after(100, self.cleanup_and_exit)
    
    def cleanup_and_exit(self):
        self.master.quit()
        self.master.destroy()
        self.redirector.cleanup()

if __name__ == "__main__":
    redirector = OutputRedirector()
    sys.stdout = redirector
    root = ctk.CTk()
    app = RollerGUI(root)
    app.redirector = redirector
    root.mainloop()