import os
import multiprocessing
import numpy as np
import pygame
from typing import Tuple, List, Dict, Any, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class GamepadExpert:

    def __init__(self, guid: str):
        self.guid = guid
        self.manager = multiprocessing.Manager()

        # Initialize shared state for inter-process communication
        self.shared_state: Dict[str, Any] = self.manager.dict()
        self.shared_state["deltas"] = [0.0] * 6
        self.shared_state["intervened"] = False

        # Start background polling process (daemon process)
        self.poll_process = multiprocessing.Process(target=self._poll_gamepad)
        self.poll_process.daemon = True
        self.poll_process.start()

    def _poll_gamepad(self) -> None:
        # Initialize pygame and joystick module
        pygame.init()
        pygame.joystick.init()
        target_joystick: Optional[pygame.joystick.Joystick] = None

        # Find target gamepad by GUID
        for joy_idx in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(joy_idx)
            if joystick.get_guid() == self.guid:
                target_joystick = joystick
                target_joystick.init()
                break

        # Exit if target gamepad not found
        if target_joystick is None:
            pygame.quit()
            return

        # Input parameters for deadzone and threshold filtering
        deadzone = 0.2
        threshold = 0.5

        try:
            while True:
                pygame.event.pump()

                # Read raw axis values from gamepad
                axis_left_x = target_joystick.get_axis(0)
                axis_left_y = target_joystick.get_axis(1)
                axis_right_y = target_joystick.get_axis(3)
                axis_right_x = target_joystick.get_axis(4)

                # Initialize action delta variables
                d_shoulder = d_arm_y = d_wrist_flex = d_wrist_roll = 0

                # Handle shoulder/arm Y axis (mutually exclusive)
                if abs(axis_left_x) > abs(axis_left_y):
                    if abs(axis_left_x) > deadzone:
                        d_shoulder = 1.0 if axis_left_x > 0 else -1.0
                else:
                    if abs(axis_left_y) > deadzone:
                        d_arm_y = 1.0 if axis_left_y > 0 else -1.0

                # Handle wrist flexion/roll (mutually exclusive)
                if abs(axis_right_y) > abs(axis_right_x):
                    if abs(axis_right_y) > deadzone:
                        d_wrist_roll = 1.0 if axis_right_y < 0 else -1.0
                else:
                    if abs(axis_right_x) > deadzone:
                        d_wrist_flex = 1.0 if axis_right_x > 0 else -1.0

                # Handle Z-axis (lift/lower) control
                d_z = 0.0
                if target_joystick.get_button(4):
                    d_z = 1.0
                elif target_joystick.get_axis(2) > threshold:
                    d_z = -1.0

                # Handle gripper control
                d_gripper = 0.0
                if target_joystick.get_axis(5) > threshold:
                    d_gripper = 1.0
                elif target_joystick.get_button(5):
                    d_gripper = -1.0

                # Update shared state with new action deltas
                deltas: List[float] = [d_shoulder, d_arm_y, d_z, d_wrist_flex, d_wrist_roll, d_gripper]
                self.shared_state["deltas"] = deltas
                self.shared_state["intervened"] = any(abs(d) > 0 for d in deltas)

                # Reduce CPU usage
                pygame.time.wait(10)

        except (BrokenPipeError, OSError, KeyboardInterrupt):
            # Cleanup on process exit
            pygame.quit()
            return

    def get_action(self) -> Tuple[np.ndarray, bool]:
        # Get latest gamepad action deltas and intervention status
        deltas = np.array(self.shared_state["deltas"], dtype=np.float32)
        intervened = self.shared_state["intervened"]
        return deltas, intervened * 0.05
