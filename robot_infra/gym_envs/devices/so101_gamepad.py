import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
import multiprocessing
import numpy as np
from typing import Tuple


class GamepadExpert:

    def __init__(self, guid):
        # Initialize gamepad with unique hardware identifier
        self.guid = guid

        # Shared state manager for inter-process communication
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["deltas"] = [0] * 6
        self.latest_data["intervened"] = False

        # Start background polling process (daemon mode to follow main process exit)
        self.process = multiprocessing.Process(target=self._read_gamepad)
        self.process.daemon = True
        self.process.start()

    def _read_gamepad(self):
        # Initialize pygame and joystick subsystem
        pygame.init()
        pygame.joystick.init()
        joystick = None

        # Locate and initialize target gamepad by GUID
        for i in range(pygame.joystick.get_count()):
            curr_joy = pygame.joystick.Joystick(i)
            if curr_joy.get_guid() == self.guid:
                joystick = curr_joy
                joystick.init()
                break

        # Exit if target gamepad not found
        if joystick is None:
            return

        # Define input sensitivity thresholds
        deadzone, threshold = 0.2, 0.5

        # Main polling loop
        while True:
            pygame.event.pump()

            # Read raw axis values from gamepad hardware
            a0 = joystick.get_axis(0)  # Left Stick X (Shoulder)
            a1 = joystick.get_axis(1)  # Left Stick Y (Arm Y)
            a3 = joystick.get_axis(3)  # Right Stick Y (Wrist Roll)
            a4 = joystick.get_axis(4)  # Right Stick X (Wrist Flex)

            # Initialize joint movement deltas
            d_sh, d_y, d_flex, d_roll = 0, 0, 0, 0

            # Exclusive axis logic for shoulder/arm Y (prevent diagonal movement)
            if abs(a0) > abs(a1):
                if abs(a0) > deadzone:
                    d_sh = 1 if a0 > 0 else -1
            else:
                if abs(a1) > deadzone:
                    d_y = 1 if a1 > 0 else -1

            # Exclusive axis logic for wrist joints (prevent diagonal movement)
            if abs(a3) > abs(a4):
                if abs(a3) > deadzone:
                    d_roll = 1 if a3 < 0 else -1
            else:
                if abs(a4) > deadzone:
                    d_flex = 1 if a4 > 0 else -1

            # Z-axis (Elevation) control (Button 4 = up, Axis 2 = down)
            d_z = 0
            if joystick.get_button(4):
                d_z = 1
            elif joystick.get_axis(2) > threshold:
                d_z = -1

            # Gripper control (Axis 5 = open, Button 5 = close)
            d_gr = 0
            if joystick.get_axis(5) > threshold:
                d_gr = 1
            elif joystick.get_button(5):
                d_gr = -1

            # Compile all movement deltas into command vector
            deltas = [d_sh, d_y, d_z, d_flex, d_roll, d_gr]

            try:
                # Update shared state with new command
                self.latest_data["deltas"] = deltas
                # Mark intervention status (active if any delta non-zero)
                self.latest_data["intervened"] = any(d != 0 for d in deltas)
            except (BrokenPipeError, OSError):
                # Exit loop if main process closes communication pipe
                break

            # Small sleep to reduce CPU usage
            pygame.time.wait(10)

    def get_action(self) -> Tuple[np.ndarray, bool]:
        # Return latest command vector and scaled intervention status
        return np.array(self.latest_data["deltas"]), self.latest_data["intervened"] * 0.05
