import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
import multiprocessing
import numpy as np
from typing import Tuple


class GamepadExpert:
    def __init__(self, guid):
        self.guid = guid
        # Use a Manager to share gamepad state across the background and main process
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["deltas"] = [0] * 6
        self.latest_data["intervened"] = False
        # Run gamepad polling in a separate daemon process to prevent blocking the RL loop
        self.process = multiprocessing.Process(target=self._read_gamepad)
        self.process.daemon = True
        self.process.start()

    def _read_gamepad(self):
        pygame.init()
        pygame.joystick.init()
        joystick = None
        # Locate the specific gamepad hardware using the provided GUID
        for i in range(pygame.joystick.get_count()):
            curr_joy = pygame.joystick.Joystick(i)
            if curr_joy.get_guid() == self.guid:
                joystick = curr_joy
                joystick.init()
                break
        if joystick is None:
            return
        deadzone, threshold = 0.2, 0.5
        while True:
            pygame.event.pump()
            # Map raw hardware axes to robot logical components
            a0 = joystick.get_axis(0)  # Left Stick X (Shoulder)
            a1 = joystick.get_axis(1)  # Left Stick Y (Arm Y)
            a3 = joystick.get_axis(3)  # Right Stick Y (Wrist Roll)
            a4 = joystick.get_axis(4)  # Right Stick X (Wrist Flex)
            d_sh, d_y, d_flex, d_roll = 0, 0, 0, 0
            # Mutually exclusive logic: prioritize the axis with the largest deflection
            # This prevents accidental diagonal movement in the shoulder/Y joints
            if abs(a0) > abs(a1):
                if abs(a0) > deadzone:
                    d_sh = 1 if a0 > 0 else -1
            else:
                if abs(a1) > deadzone:
                    d_y = 1 if a1 > 0 else -1
            # Mirroring the exclusive logic for the wrist joints
            if abs(a3) > abs(a4):
                if abs(a3) > deadzone:
                    d_roll = 1 if a3 < 0 else -1
            else:
                if abs(a4) > deadzone:
                    d_flex = 1 if a4 > 0 else -1
            # Z-axis (Elevation) mapping using Button 4 and Analog Axis 2
            d_z = 0
            if joystick.get_button(4):
                d_z = 1
            elif joystick.get_axis(2) > threshold:
                d_z = -1
            # Gripper control mapping using Analog Axis 5 and Button 5
            d_gr = 0
            if joystick.get_axis(5) > threshold:
                d_gr = 1
            elif joystick.get_button(5):
                d_gr = -1
            deltas = [d_sh, d_y, d_z, d_flex, d_roll, d_gr]
            try:
                self.latest_data["deltas"] = deltas
                # Flag intervention if any delta is non-zero (user is actively controlling)
                self.latest_data["intervened"] = any(d != 0 for d in deltas)
            except (BrokenPipeError, OSError):
                # Shutdown the thread if the main application closes the Manager pipe
                break
            pygame.time.wait(10)

    def get_action(self) -> Tuple[np.ndarray, bool]:
        # Return the most recent command vector and the active intervention status
        return np.array(self.latest_data["deltas"]), self.latest_data["intervened"]
