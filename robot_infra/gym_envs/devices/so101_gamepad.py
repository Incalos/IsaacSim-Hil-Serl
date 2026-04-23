import os
import multiprocessing
import numpy as np
import pygame
from typing import Tuple, List, Dict, Any, Optional

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


class GamepadExpert:
    """Gamepad controller for real-time input polling and inter-process state sharing.

    Manages a background process to poll gamepad inputs, applies deadzone/threshold filtering,
    and provides access to normalized action deltas and intervention status.
    """

    def __init__(self, guid: str):
        """Initialize gamepad controller with target GUID and shared state manager.

        Args:
            guid: Unique GUID identifier for the target gamepad device
        """
        self.guid = guid
        self.manager = multiprocessing.Manager()
        self.shared_state: Dict[str, Any] = self.manager.dict()
        self.shared_state["deltas"] = [0.0] * 6
        self.shared_state["intervened"] = False
        self.poll_process = multiprocessing.Process(target=self._poll_gamepad)
        self.poll_process.daemon = True
        self.poll_process.start()

    def _poll_gamepad(self) -> None:
        """Background process to continuously poll and process gamepad inputs.

        Initializes pygame joystick subsystem, locates target gamepad by GUID,
        applies input filtering (deadzone/threshold), and updates shared state
        with normalized action deltas and intervention status.
        """
        pygame.init()
        pygame.joystick.init()
        target_joystick: Optional[pygame.joystick.Joystick] = None

        for joy_idx in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(joy_idx)
            if joystick.get_guid() == self.guid:
                target_joystick = joystick
                target_joystick.init()
                break

        if target_joystick is None:
            pygame.quit()
            return

        deadzone = 0.2
        threshold = 0.5

        try:
            while True:
                pygame.event.pump()
                axis_left_x = target_joystick.get_axis(0)
                axis_left_y = target_joystick.get_axis(1)
                axis_right_y = target_joystick.get_axis(3)
                axis_right_x = target_joystick.get_axis(4)

                d_shoulder = d_forward = d_wrist_flex = d_wrist_roll = 0

                if abs(axis_left_x) > abs(axis_left_y):
                    if abs(axis_left_x) > deadzone:
                        d_shoulder = 1.0 if axis_left_x > 0 else -1.0
                else:
                    if abs(axis_left_y) > deadzone:
                        d_forward = -1.0 if axis_left_y > 0 else 1.0

                if abs(axis_right_y) > abs(axis_right_x):
                    if abs(axis_right_y) > deadzone:
                        d_wrist_roll = 1.0 if axis_right_y < 0 else -1.0
                else:
                    if abs(axis_right_x) > deadzone:
                        d_wrist_flex = 1.0 if axis_right_x > 0 else -1.0

                d_z = 0.0
                if target_joystick.get_button(4):
                    d_z = 1.0
                elif target_joystick.get_axis(2) > threshold:
                    d_z = -1.0

                d_gripper = 0.0
                if target_joystick.get_axis(5) > threshold:
                    d_gripper = -1.0
                elif target_joystick.get_button(5):
                    d_gripper = 1.0

                deltas: List[float] = [d_shoulder, d_forward, d_z, d_wrist_flex, d_wrist_roll, d_gripper]
                self.shared_state["deltas"] = deltas
                self.shared_state["intervened"] = any(abs(d) > 0 for d in deltas)

                pygame.time.wait(10)

        except (BrokenPipeError, OSError, KeyboardInterrupt):
            pygame.quit()
            return

    def get_action(self) -> Tuple[np.ndarray, bool]:
        """Retrieve latest gamepad action deltas and intervention status.

        Returns:
            Tuple containing:
                - 6-element float32 array of action deltas (shoulder, forward, z, wrist flex, wrist roll, gripper)
                - Boolean flag indicating if any non-zero input was detected (intervention)
        """
        deltas = np.array(self.shared_state["deltas"], dtype=np.float32)
        intervened = self.shared_state["intervened"]
        return deltas, intervened
