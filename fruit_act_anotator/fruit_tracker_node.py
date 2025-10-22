# =====================================================================================
# このノードは、2D検出結果、深度情報、TFを統合し、物体の3D位置を追跡します。
# オクルージョンが発生した場合、最後に観測した3D位置と最新のTFから、
# 現在のカメラ画像上での2Dバウンディングボックスを「予測」して配信し続けます。
# ★★★ 中心付近を探索するロバストな深度取得ロジックを実装。 ★★★
# ★★★ さらに、予測したアノテーション(txt)と、その時のRGB画像(png)をペアで保存します。 ★★★
# =====================================================================================
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_point
import image_geometry
import os
import cv2
from std_msgs.msg import Int32
import time


class FruitTrackerNode(Node):
    def __init__(self):
        super().__init__('fruit_tracker_node')
        
        self.declare_parameter('annotation_path', '~/yolo_annotations/labels')
        self.declare_parameter('image_save_path', '~/yolo_annotations/images')
        
        annotation_path_str = self.get_parameter('annotation_path').get_parameter_value().string_value
        self.annotation_path = os.path.expanduser(annotation_path_str)
        if not os.path.exists(self.annotation_path): os.makedirs(self.annotation_path)

        image_save_path_str = self.get_parameter('image_save_path').get_parameter_value().string_value
        self.image_save_path = os.path.expanduser(image_save_path_str)
        if not os.path.exists(self.image_save_path): os.makedirs(self.image_save_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.waypoint_num = 10000 #初期値かぶり防止

        self.last_known_3d_pos_base = None
        self.last_detection_time = self.get_clock().now()
        self.camera_model = None
        self.latest_depth_image = None
        self.latest_color_image_msg = None
        
        self.create_subscription(Detection2DArray, '/detections_2d', self.detection_callback, 1)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 1)
        self.create_subscription(Int32, 'waypoint_num', self.waypoint_callback, 1)



        self.pub_tracked_bbox = self.create_publisher(Detection2DArray, '/tracked_detections', 1)
        self.pub_fruit_point = self.create_publisher(PointStamped, "/detected_fruits", 1)

        self.timer = self.create_timer(0.05, self.predict_and_publish)
        self.get_logger().info("FruitTrackerNode (v3.0 SaveOnDetect) has started.")

    def color_image_callback(self, msg):
        self.latest_color_image_msg = msg

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def get_robust_depth(self, cv_depth, u, v, radius=5):
        h, w = cv_depth.shape
        umin = max(0, u - radius)
        umax = min(w - 1, u + radius)
        vmin = max(0, v - radius)
        vmax = min(h - 1, v + radius)

        for vi in range(vmin, vmax + 1):
            for ui in range(umin, umax + 1):
                depth = cv_depth[vi, ui]
                if depth > 0:
                    return depth / 1000.0, ui, vi
        return 0, u, v
    
    def waypoint_callback(self, msg):
        if self.waypoint_num != msg.data:
            # ★★★ YOLOアノテーションは全ての検出物体を保存する ★★★
            if len(self.latest_bbox.detections) > 0:
                self.save_image_and_annotations(self.latest_color_image_msg, self.latest_bbox.detections, msg.data)
        self.waypoint_num = msg.data


    def detection_callback(self, msg):
        self.latest_bbox = msg

        if self.latest_depth_image is None or self.camera_model is None or self.latest_color_image_msg is None: 
            return

        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
        
        best_detection = max(msg.detections, key=lambda d: d.bbox.size_x * d.bbox.size_y, default=None)
        if not best_detection: 
            return
        
        # ★★★ 最後に検出したバウンディングボックスを保存 ★★★
        self.last_detected_bbox = best_detection.bbox


        u_center = int(best_detection.bbox.center.position.x)
        v_center = int(best_detection.bbox.center.position.y)
        
        depth, u_eff, v_eff = self.get_robust_depth(cv_depth, u_center, v_center)

        if depth == 0:
            self.get_logger().warn(f"No valid depth found around ({u_center}, {v_center}). Discarding.")
            return
        
        ray = self.camera_model.projectPixelTo3dRay((u_eff, v_eff))
        point3d_camera = [val * depth for val in ray]
        pt_camera = PointStamped()
        pt_camera.header = msg.header
        pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = point3d_camera
        try:
            transform = self.tf_buffer.lookup_transform("base_link", pt_camera.header.frame_id, rclpy.time.Time())
            pt_base = do_transform_point(pt_camera, transform)
            self.last_known_3d_pos_base = pt_base
            self.last_detection_time = self.get_clock().now()
            
            # 腕を動かすためのPublishは一度だけで良いかもしれないので、ロジックを工夫する余地あり
            self.pub_fruit_point.publish(pt_base)
            
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")


    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 新しいヘルパー関数：画像とアノテーションの保存処理 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    def save_image_and_annotations(self, image_msg, detections, waypoint_num):
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")#デバック用
        self.get_logger().info(f"bbbbbbbbbbbbbbbbbbbbbbbbbb")#デバック用

        # ファイル名のベース部分を作成
        timestamp = time.strftime("%m%d-%H%M%S")
        # stamp = image_msg.header.stamp
        # filename_base = f"{stamp.sec}_{stamp.nanosec}_waypoint{waypoint_num}"
        filename_base = f"{timestamp}_waypoint{waypoint_num}"
        

        # --- 画像の保存 ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            img_filepath = os.path.join(self.image_save_path, f"{filename_base}.png")
            cv2.imwrite(img_filepath, cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            return # 画像保存に失敗したらアノテーションも保存しない

        # --- TXTファイルの保存 ---
        img_w = float(self.camera_model.width)
        img_h = float(self.camera_model.height)
        yolo_lines = []
        for det in detections:
            bbox = det.bbox
            x_center_norm = bbox.center.position.x / img_w
            y_center_norm = bbox.center.position.y / img_h
            width_norm = bbox.size_x / img_w
            height_norm = bbox.size_y / img_h
            class_id = 0 # クラスIDを0と仮定
            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        if yolo_lines:
            txt_filepath = os.path.join(self.annotation_path, f"{filename_base}.txt")
            try:
                with open(txt_filepath, 'w') as f:
                    f.writelines(yolo_lines)
                self.get_logger().info(f"Saved image/annotation for {filename_base}")
            except IOError as e:
                self.get_logger().error(f"Failed to write annotation file: {e}")

    def predict_and_publish(self):
    #検出できなかった時に、TFを使って物体の位置を予測し、
    #バウンディングボックスを推定して配信する
    
        if self.last_known_3d_pos_base is None or self.camera_model is None:
            return
    
        # 予測が必要か判定(0.5秒以上検出がない場合)
        is_prediction = (self.get_clock().now() - self.last_detection_time) > rclpy.duration.Duration(seconds=0.5)
        if not is_prediction:
            return
    
        # 最後に検出した物体の情報が必要
        if not hasattr(self, 'last_detected_bbox') or self.last_detected_bbox is None:
            return
    
        try:
            # base_link -> camera_color_optical_frameへの変換を取得
            transform = self.tf_buffer.lookup_transform(
                "camera_color_optical_frame",
                "base_link",
                rclpy.time.Time()
            )
        
            # 3D位置をカメラ座標系に変換
            pt_camera = do_transform_point(self.last_known_3d_pos_base, transform)
        
            # カメラ座標系での3D位置
            x_cam = pt_camera.point.x
            y_cam = pt_camera.point.y
            z_cam = pt_camera.point.z
            
            # 深度がゼロまたは負の場合はスキップ
            if z_cam <= 0:
                return
            
            # 3D位置を2D画像座標に投影
            u_pred, v_pred = self.camera_model.project3dToPixel((x_cam, y_cam, z_cam))
            u_pred = int(u_pred)
            v_pred = int(v_pred)
            
            # 画像範囲外ならスキップ
            if u_pred < 0 or u_pred >= self.camera_model.width or v_pred < 0 or v_pred >= self.camera_model.height:
                return
            
            # カメラの傾き角度θを計算
            # Y軸方向のベクトル(カメラの光軸方向)
            camera_y_axis = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            # クォータニオンから回転行列を計算してY軸(前方)ベクトルを取得
            from scipy.spatial.transform import Rotation as R
            rot = R.from_quat(camera_y_axis)
            camera_forward = rot.apply([0, 0, 1])  # カメラ座標系のZ軸が前方
            
            # 水平面に対する角度θを計算
            # カメラが水平のとき、forward[2]=0
            # カメラが下向きのとき、forward[2]<0
            theta = np.arctan2(-camera_forward[2], np.sqrt(camera_forward[0]**2 + camera_forward[1]**2))
            
            # 最後に検出したバウンディングボックスのサイズ
            w_original = self.last_detected_bbox.size_x  # 横幅
            h_original = self.last_detected_bbox.size_y  # 縦幅
            
            # 予測式: h_p = |h*cosθ| + |w*sinθ|
            h_predicted = abs(h_original * np.cos(theta)) + abs(w_original * np.sin(theta))
            
            # 横幅は変わらない(円柱の仮定)
            w_predicted = w_original
            
            # 予測したバウンディングボックスを作成
            predicted_detection = Detection2D()
            predicted_detection.header.frame_id = "camera_color_optical_frame"
            predicted_detection.header.stamp = self.get_clock().now().to_msg()
            
            bbox = BoundingBox2D()
            bbox.center.position.x = float(u_pred)
            bbox.center.position.y = float(v_pred)
            bbox.size_x = float(w_predicted)
            bbox.size_y = float(h_predicted)
            predicted_detection.bbox = bbox
            
            # Detection2DArrayとして配信
            detection_array = Detection2DArray()
            detection_array.header = predicted_detection.header
            detection_array.detections.append(predicted_detection)
            
            self.pub_tracked_bbox.publish(detection_array)
            
            # 予測したバウンディングボックスと画像を保存
            if self.latest_color_image_msg is not None:
                self.save_image_and_annotations(
                    self.latest_color_image_msg,
                    detection_array.detections,
                    self.waypoint_num
                )
            
            self.get_logger().info(
                f"Predicted bbox at ({u_pred}, {v_pred}), "
                f"theta={np.degrees(theta):.1f}deg, "
                f"h={h_predicted:.1f}px"
            )
            
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF prediction failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FruitTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()