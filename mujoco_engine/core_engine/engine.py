#! /usr/bin/env python
""" `mujoco_engine.py`

    @author:  Jack (Jianxiang) Xu
        Contacts    : projectbyjx@gmail.com
        Last edits  : July 27, 2022

    @description:
        This library will initiate an Engine that handles updates and rendering
"""
#===================================#
#  I M P O R T - L I B R A R I E S  #
#===================================#

# python libraries:
import os
import threading
import time
import signal

from enum import Enum
from datetime import timedelta


# python 3rd party libraries:
import numpy as np
import mujoco
import cv2

# custom libraries:
import mujoco_viewer

# local libraries:
from mujoco_engine.core_engine.wrapper.core import MjData, MjModel

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #        ___           ___         ___          ___           ___           ___       # #
# #       /__/\         /__/\       /  /\        /  /\         /  /\         /  /\      # #
# #      |  |::\        \  \:\     /  /:/       /  /::\       /  /:/        /  /::\     # #
# #      |  |:|:\        \  \:\   /__/::\      /  /:/\:\     /  /:/        /  /:/\:\    # #
# #    __|__|:|\:\   ___  \  \:\  \__\/\:\    /  /:/  \:\   /  /:/  ___   /  /:/  \:\   # #
# #   /__/::::| \:\ /__/\  \__\:\    \  \:\  /__/:/ \__\:\ /__/:/  /  /\ /__/:/ \__\:\  # #
# #   \  \:\~~\__\/ \  \:\ /  /:/     \__\:\ \  \:\ /  /:/ \  \:\ /  /:/ \  \:\ /  /:/  # #
# #    \  \:\        \  \:\  /:/      /  /:/  \  \:\  /:/   \  \:\  /:/   \  \:\  /:/   # #
# #     \  \:\        \  \:\/:/      /__/:/    \  \:\/:/     \  \:\/:/     \  \:\/:/    # #
# #      \  \:\        \  \::/       \__\/      \  \::/       \  \::/       \  \::/     # #
# #       \__\/         \__\/                    \__\/         \__\/         \__\/      # #
# #                                                                                     # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#=======================#
#  D E F I N I T I O N  #
#=======================#
class MuJoCo_Engine_InterruptException(Exception):
    pass

class Mujoco_Engine:
    #===================#
    #  C O N S T A N T  #
    #===================#
    _camera_config = {
        "camera/zed/L": {"width": 1280, "height":720, "fps": 60, "id":1},
        "camera/zed/R": {"width": 1280, "height":720, "fps": 60, "id":0},
        "camera/intel/rgb": {"width": 1280, "height":720, "fps": 60, "id":2},
    }
    _camera_views = {}
    _IC_state = None
    _core = None
    
    #===============================#
    #  I N I T I A L I Z A T I O N  #
    #===============================#
    def __init__(self,  
        xml_path, rate_Hz, 
        camera_config=None, 
        name="DEFAULT", 
        CAMERA_V_FACTOR=3
    ):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        ## Init Configs:
        if camera_config:
            self._camera_config = (camera_config) # override if given
        self._name = name
        self._rate_Hz = rate_Hz
        
        ## Initiate MJ
        # self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model = MjModel.from_xml_path(xml_path=xml_path)
        self.mj_data = MjData(self.mj_model)
        
        ## MJ Viewer:
        self.mj_viewer = mujoco_viewer.MujocoViewer(self.mj_model._model, self.mj_data._data, 
            title="Mujoco-Engine", 
            sensor_config=self._camera_config,
            window_size=(1280,720),
        )
        # if len(self._camera_config):
        # self.mj_viewer_off = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data, width=800, height=800, title="camera-view")
        self._t_update = time.time()
        
        # cv2 window
        cv2.startWindowThread()
        self.h_min = np.Infinity
        for camera, config in self._camera_config.items():
            self.h_min = int(min(config["height"]/CAMERA_V_FACTOR, self.h_min))
        
    #==================================#
    #  P U B L I C    F U N C T I O N  #
    #==================================#
    def shutdown(self):
        print("[Job_Engine::{}] Program killed: running cleanup code".format(self._name))
        self.mj_viewer.terminate_safe()
        cv2.destroyAllWindows()
            
    def is_shutdown(self):
        try:
            return False
        except MuJoCo_Engine_InterruptException:
            self.shutdown()
            return True
    
    #====================================#
    #  P R I V A T E    F U N C T I O N  #
    #====================================#    
    def _signal_handler(self, signum, frame):
        raise MuJoCo_Engine_InterruptException
    
    def _internal_engine_update(self):
        self._update()

    def _update(self, if_camera_preview=True):
        delta_t = time.time() - self._t_update
        # print("FPS: {0}".format(1/delta_t))
        # - command joint control angles:
        self.mj_data.actuator("wam/J1/P").ctrl = -1.92
        self.mj_data.actuator("wam/J2/P").ctrl = 1.88

        # - render current view:
        for i in range(10):
            mujoco.mj_step(self.mj_model._model, self.mj_data._data)
        self.mj_viewer.process_safe()
        self.mj_viewer.update_safe()
        self.mj_viewer.render_safe()
        self.mj_viewer.render_sensor_cameras_safe()
        # self.mj_viewer_off.render()

        # - capture view:
        camera_sensor_data = self.mj_viewer.acquire_sensor_camera_frames_safe()
        print(camera_sensor_data["frame_stamp"])
        
        if if_camera_preview:
            cv2_capture_window = []
            for camera_name, camera_buf in camera_sensor_data["frame_buffer"].items():
                img = cv2.cvtColor(camera_buf, cv2.COLOR_RGB2BGR)
                img = cv2.flip(img, 0)
                img = cv2.resize(img, (int(img.shape[1] * self.h_min / img.shape[0]), self.h_min))
                cv2_capture_window.append(img)
                
            cv2.imshow("camera views", cv2.hconcat(cv2_capture_window))
        
        # - update:
        self._t_update = time.time()
        
    
