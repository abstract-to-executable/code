import datetime
import json
import os
import time
from threading import Event

import cv2
import numpy as np
import pyrealsense2 as rs
import transforms3d
from kinematics_helper import SAPIENKinematicsModel
from pynput import keyboard
from record3d import Record3DStream
from xarm.wrapper import XArmAPI

VELOCITY_THRESHOLD = 0.02


class CameraApp:
    def __init__(self, file: str = ""):
        self.rgb_video_file = file
        self.event = Event()
        self.session = None

        self.rgb_stream = None
        self.depth_stream = None
        self.read_count = 0

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx=0):
        if not os.path.exists(self.rgb_video_file):
            print('Searching for devices')
            devs = Record3DStream.get_connected_devices()
            print('{} device(s) found'.format(len(devs)))
            for dev in devs:
                print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

            if len(devs) <= dev_idx:
                raise RuntimeError('Cannot connect to device #{}, try different index.'.format(dev_idx))

            dev = devs[dev_idx]
            self.session = Record3DStream()
            self.session.on_new_frame = self.on_new_frame
            self.session.on_stream_stopped = self.on_stream_stopped
            self.session.connect(dev)  # Initiate connection and start capturing

            # Wait for camera to connected, here we use a hack to check whether we can really get the info from camera
            time_out = 2
            while self.camera_intrinsics.sum() < 10 and time_out >= 0:
                time.sleep(1e-3)
                time_out -= 1e-3
        else:
            print(f"Streaming from video file {os.path.abspath(self.rgb_video_file)}")
            self.rgb_stream = cv2.VideoCapture(self.rgb_video_file)
            self.depth_stream = np.load(self.rgb_video_file.replace("rgb", "depth").replace("mp4", "npz"))['depth']

    @staticmethod
    def _get_intrinsic_mat_from_coeffs(coeffs):
        return np.array([[coeffs.fx, 0, coeffs.tx],
                         [0, coeffs.fy, coeffs.ty],
                         [0, 0, 1]])

    @property
    def camera_intrinsics(self):
        if self.session is None:
            intrinsic_mat = np.array([[804.5928, 0.0, 357.6741333]
                                      [0.0, 804.5928, 474.83026123]
                                      [0., 0., 1.]])
        else:
            intrinsic_mat = self._get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
        return intrinsic_mat

    def fetch_rgb_and_depth(self):
        if not self.rgb_video_file:
            self.event.wait(1)
            depth = np.transpose(self.session.get_depth_frame(), [1, 0])
            rgb = np.transpose(self.session.get_rgb_frame(), [1, 0, 2])

            is_true_depth = depth.shape[0] == 480
            if is_true_depth:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            return rgb, depth
        else:
            while self.rgb_stream.isOpened():
                success_rgb, bgr_frame = self.rgb_stream.read()
                depth_frame = self.depth_stream[self.read_count]
                self.read_count += 1

                if not success_rgb:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), depth_frame


class Recorder:
    def __init__(self, xarm_ip="192.168.1.240",
                 urdf_path="scripts/realworld/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf",
                 use_iphone=False):
        # Robot arm
        self.arm = XArmAPI(xarm_ip)
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.speed = 10
        self.last_qpos = np.zeros(22)
        self.robot_model = SAPIENKinematicsModel(urdf_path)

        # Board Property
        # self.board_size = (5, 4)
        self.board_size = (4, 3)
        self.board_square_size = 10

        # Camera
        self.use_iphone = use_iphone
        if use_iphone:
            self.camera = CameraApp()
            self.camera.connect_to_device()
            # print("Camera Intrinsics of record3d:", self.camera_mat)
        else:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            # Skip 5 first frames to give the Auto-Exposure time to adjust
            for x in range(5):
                pipeline.wait_for_frames()
            self.pipeline = pipeline
            self.profile = profile

        # Camera Info
        if use_iphone:
            self.camera_matrix_k = self.camera.camera_intrinsics
            self.focal_length = self.camera.camera_intrinsics[0, 0]
        else:
            color_stream = profile.get_stream(rs.stream.color)
            camera_info = color_stream.as_video_stream_profile().get_intrinsics()
            self.camera_matrix_k = np.array(
                [[camera_info.fx, 0, camera_info.ppx], [0, camera_info.fy, camera_info.ppy], [0, 0, 1]])
        self.criteria_subpixel = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Motion Info
        self.robot_joints = self.robot_model.robot.get_active_joints()
        self.robot_joint_dof = len(self.robot_joints)
        self.robot_joint_names = [joint.get_name() for joint in self.robot_joints]
        self.board_link_name = "link6"
        self.folder = os.path.abspath("./captured_data")
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def fetch_color_and_depth(self):
        if self.use_iphone:
            color_image, depth_image = self.camera.fetch_rgb_and_depth()
        else:
            depth_frame = None
            while not depth_frame:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def auto_capture(self, name_prefix):
        data_path = os.path.join(self.folder, name_prefix, "all.json")
        if not os.path.exists(data_path):
            raise RuntimeError("No folder with data file exist for {}".format(self.folder))
        selected_data = []

        with open(data_path, "r") as f:
            data = json.load(f)
            joints = [data[i]["joint"] for i in range(len(data))]

        # Get a valid folder name
        while True:
            exist_count = 0
            new_folder_path = os.path.join(self.folder, "{}_{}".format(name_prefix, format(exist_count, "03d")))
            if not os.path.exists(new_folder_path):
                os.mkdir(new_folder_path)
                break
            else:
                exist_count += 1

        try:
            for i in range(len(joints)):
                print("Move robot to {}-th / {} pose".format(i, len(joints)))
                self.arm.set_servo_angle(angle=joints[i], speed=50, is_radian=True, wait=True)
                print(self.arm.get_servo_angle(), self.arm.get_servo_angle(is_radian=True))
                selected_data.append(self.capture_single_data(os.path.join(new_folder_path, str(i))))

        finally:
            if len(selected_data) == 0:
                print("Zero data is collected, do not save anything!")
            json_path = os.path.join(new_folder_path, "all.json")
            with open(json_path, "w") as f:
                json.dump(selected_data, f, sort_keys=True, indent=2)

    def manually_capture(self, name_prefix=None):
        if not os.path.exists(os.path.join(self.folder, name_prefix)):
            os.mkdir(os.path.join(self.folder, name_prefix))
        selected_data = []
        if not name_prefix:
            now = datetime.datetime.now()
            name_prefix = now.strftime("%m_%d_%H_%M_%S")

        print("Move the robot use joystick to the first desired pose.")

        num = 0
        try:
            while True:
                image, _ = self.fetch_color_and_depth()
                cv2.imshow("viz", image)
                cv2.waitKey(1)
                with keyboard.Events() as events:
                    event = events.get(timeout=0.5)
                    if event is None:
                        continue
                    if event.key == event.key.from_char("q"):
                        print("Finishing calibration data collection with {}".format(len(selected_data)))
                        break

                    elif event.key == event.key.from_char("c"):
                        filename = os.path.join(self.folder, name_prefix, str(num))
                        captured_data = self.capture_single_data(filename)
                        if captured_data is None:
                            continue
                        selected_data.append(captured_data)
                        print("Move the robot to next pose.")
                        time.sleep(1)
                        num += 1

        finally:
            if len(selected_data) == 0:
                print("Zero data is collected, do not save anything!")
            json_path = os.path.join(self.folder, name_prefix, "all.json")
            with open(json_path, "w") as f:
                json.dump(selected_data, f, sort_keys=True, indent=2)

    def capture_single_data(self, filename):
        self.wait_to_settle(1.0)
        # self.save_point_cloud("{}.pcd".format(filename))
        rgb, depth = self.fetch_color_and_depth()
        data = dict(rgb=rgb,depth=depth, cam_int=self.camera_matrix_k)
        
        import pickle
        with open("verifypcd", "wb") as f:
            pickle.dump(data, f)

        marker_pose = self.get_marker_pose(filename, rgb)
        if marker_pose is None:
            return None
        xarm_qpos = self.get_xarm_qpos()
        robot_qpos = np.zeros(self.robot_model.robot.dof)
        robot_qpos[:6] = xarm_qpos
        joint_names = self.robot_joint_names
        current_pose = self.robot_model.get_link_pose(robot_qpos, joint_names, self.board_link_name)

        single_data = {"joint": xarm_qpos, "ee_pose": self.pose2dict(current_pose),
                       "marker_pose": marker_pose}
        return single_data

    @staticmethod
    def pose2dict(pose_array):
        pos = pose_array[:3]
        quat = pose_array[3:]
        quat_xyzw = np.array([*quat[1:], quat[0]])
        return dict(position=pos.tolist(), orientation=quat_xyzw.tolist())

    def get_marker_pose(self, filename, image):
        cv2.imwrite(f"{filename}.png", image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f"{filename}_gray.png", grey)
        found, corners = cv2.findChessboardCorners(grey, self.board_size, None)
        if not found:
            print(f"Marker not detected")
            return None
        corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), self.criteria_subpixel)
        obj_point = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1,
                                                                                          2) * self.board_square_size

        # Draw for visualization
        img = cv2.drawChessboardCorners(image, self.board_size, corners, True)
        cv2.imwrite("{}_debug.png".format(filename), img)

        _, rvecs, tvecs = cv2.solvePnP(obj_point, corners, self.camera_matrix_k, None)
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        homo_matrix = np.eye(4)
        homo_matrix[0:3, 0:3] = rotation_matrix
        homo_matrix[0:3, 3:4] = tvecs / 1000.0  # mm to m
        return homo_matrix.tolist()

    def save_point_cloud(self, filename):

        pass

    def wait_to_settle(self, timeout=3.0):
        tic = time.time()
        last_xarm_qpos = self.get_xarm_qpos()
        last_time = time.time()
        while time.time() - tic < timeout:
            xarm_qpos = self.get_xarm_qpos()
            current = time.time()
            xarm_velocity = (np.array(xarm_qpos) - np.array(last_xarm_qpos)) / (current - last_time)
            not_moving = np.max(np.abs(xarm_velocity)) < VELOCITY_THRESHOLD
            last_xarm_qpos = xarm_qpos
            last_time = current
            if not_moving:
                return True

        raise RuntimeError(f'Joint not stop during {timeout}')

    def get_xarm_qpos(self) -> np.ndarray:
        xarm_qpos = self.arm.get_servo_angle(is_radian=True)[1][:6]
        return xarm_qpos


class HandEyeSolver:
    def __init__(self, name, count=None, filtered_idx=[]):
        self.folder = os.path.join("./captured_data", name)
        if count:
            self.folder = os.path.join("./captured_data", "{}_{}".format(name, format(count, "03d")))
        if not os.path.exists(self.folder):
            raise RuntimeError("No calibration data folder {}".format(self.folder))

        with open(os.path.join(self.folder, "all.json"), "r") as f:
            data = json.load(f)
            self.data = []
            for i in range(len(data)):
                if i in filtered_idx:
                    pass
                else:
                    self.data.append(data[i])
            self.data_num = len(self.data)

    def calculate_and_show(self):
        ee_pose = np.ones([4, 4, self.data_num])
        marker_pose = np.ones([4, 4, self.data_num])
        for i in range(self.data_num):
            ee_json = self.data[i]["ee_pose"]
            ee_quat = np.array([ee_json["orientation"][3], ee_json["orientation"][0], ee_json["orientation"][1],
                                ee_json["orientation"][2]])
            ee_in_base = np.eye(4)
            ee_in_base[:3, :3] = transforms3d.quaternions.quat2mat(ee_quat)
            ee_in_base[:3, 3] = np.array(ee_json["position"])
            ee_pose[:, :, i] = ee_in_base
            marker_pose[:, :, i] = np.array(self.data[i]["marker_pose"])
        Hc2g, res = self.__calculate(ee_pose, marker_pose)

        print("*" * 10, "Result", "*" * 10)
        print("The transformation from:")
        print(np.array2string(Hc2g, separator=', '))
        print("Residual Term")
        print(res)

    @staticmethod
    def skew_matrix(v):
        v = v.flatten()
        if len(v) != 3:
            raise RuntimeError("Skew matrix should take a 3-d vector as input")
        a1, a2, a3 = v
        return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    def __calculate(self, Hg2w, Hb2c):
        # This is the eye-to-hand version, not eye-in-hand
        n = Hg2w.shape[2]
        assert n == Hb2c.shape[2], "Input matrix should have same number"

        Hgij_list = []
        Hcij_list = []
        A = np.zeros([3 * n - 3, 3])
        b = np.zeros([3 * n - 3])

        for i in range(n - 1):
            Hgij_list.append(np.dot(Hg2w[:, :, i + 1], np.linalg.inv(Hg2w[:, :, i])))
            Hcij_list.append(np.dot(Hb2c[:, :, i + 1], np.linalg.inv(Hb2c[:, :, i])))

            rgij = cv2.Rodrigues(Hgij_list[i][:3, :3])[0].reshape(3, )
            rcij = cv2.Rodrigues(Hcij_list[i][:3, :3])[0].reshape(3, )
            tgij = np.linalg.norm(rgij)
            tcij = np.linalg.norm(rcij)
            rgij /= tgij
            rcij /= tcij

            # Turn it into modified rodrigues in Tsai
            Pgij = 2 * np.sin(tgij / 2) * rgij
            Pcij = 2 * np.sin(tcij / 2) * rcij

            # Solve equation: skew(Pgij+Pcij)*x = Pcij-Pgij
            # A = skew(Pgij+Pcij) b = Pcij-Pgij
            A[3 * i:3 * i + 3, 0:3] = self.skew_matrix(Pgij + Pcij)
            b[3 * i:3 * i + 3] = Pcij - Pgij

        x = np.dot(np.linalg.pinv(A), b.reshape(3 * n - 3, 1))

        # Compute residue
        err = np.dot(A, x) - b.reshape(3 * n - 3, 1)
        res_rotation = np.sqrt(sum((err * err)) / (n - 1))
        Pcg = 2 * x / (np.sqrt(1 + np.linalg.norm(x) ** 2))
        Rcg = (1 - np.linalg.norm(Pcg) ** 2 / 2) * np.eye(3) + 0.5 * (
                np.dot(Pcg.reshape((3, 1)), Pcg.reshape((1, 3))) + np.sqrt(
            4 - np.linalg.norm(Pcg) ** 2) * self.skew_matrix(Pcg.reshape(3, )))

        # Compute translation from A*Tcg = b
        # (rgij-I)*Tcg=rcg*tcij-tgij
        for i in range(n - 1):
            A[3 * i:3 * i + 3, :] = (Hgij_list[i][:3, :3] - np.eye(3))
            b[3 * i:3 * i + 3] = np.dot(Rcg, Hcij_list[i][:3, 3]) - Hgij_list[i][:3, 3]

        Tcg = np.dot(np.linalg.pinv(A), b)
        err = np.dot(A, Tcg) - b
        res_translation = np.sqrt(sum(err ** 2) / (n - 1))
        Hc2g = np.hstack((Rcg, Tcg.reshape(3, 1)))
        Hc2g = np.vstack((Hc2g, np.array([[0, 0, 0, 1]])))
        error = np.hstack((res_translation, res_rotation))

        return Hc2g, error

    def test(self):
        Hcam2base = np.eye(4)
        Hcam2base[:3, :3] = transforms3d.euler.euler2mat(1, 0.5, 0.2)
        Hcam2base[:3, 3] = np.random.rand(3)
        Hmarker2ee = np.eye(4)
        Hmarker2ee[:3, :3] = transforms3d.euler.euler2mat(1, 0.5, 0.2)
        Hmarker2ee[:3, 3] = np.random.rand(3)

        n = 6
        input1 = np.ones([4, 4, n])
        input2 = np.ones([4, 4, n])
        for i in range(n):
            Hee2base = np.eye(4)
            random_quat = np.random.rand(4)
            random_quat /= np.linalg.norm(random_quat)
            Hee2base[:3, :3] = transforms3d.quaternions.quat2mat(random_quat)
            Hee2base[:3, 3] = np.random.rand(3)
            Hmarker2cam = np.dot(np.dot(np.linalg.inv(Hcam2base), Hee2base), Hmarker2ee)

            input1[:, :, i] = Hee2base
            input2[:, :, i] = Hmarker2cam

        Hc2g, res = self.__calculate(input1, input2)

        print("*" * 10, "Result", "*" * 10)
        print("The transformation from:")
        print(Hc2g)
        print("Ground Truth")
        print(Hcam2base)
        print("Residual Term")
        print(res)


if __name__ == "__main__":
    np.set_printoptions()
    data_name = "2022_06_18_realsense_yz"
    recorder = Recorder()
    # recorder.auto_capture(data_name)
    recorder.manually_capture(data_name)
    # filtered_idx=[2,3,9]
    solver = HandEyeSolver(data_name)
    solver.calculate_and_show()
    # solver.test()


# STEPS
# Initialize robot
# Do small transitions, after each one press c
# after 20 something transitions, press q
# check in captured data there is alignment
# DO NOT PRESS C TWICE IN A ROLL

# ********** Result **********
# The transformation from:
# [[-0.87922824, -0.11325994,  0.4627417 ,  0.32761455],
#  [ 0.39186398,  0.38044942,  0.83767587, -0.1523784 ],
#  [-0.27092493,  0.91784009, -0.29011938,  0.36125163],
#  [ 0.        ,  0.        ,  0.        ,  1.        ]]
# Residual Term
# [0.04100176 0.70363825]

#HOW TO VERIFY?
# Get a PCD, transform to base, draw a XYZ coordinate at base, see if its at the right xyz place.