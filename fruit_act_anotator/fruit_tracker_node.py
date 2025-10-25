# =====================================================================================
# このノードは、2D検出結果、深度情報、TFを統合し、物体の3D位置を追跡します。
# オクルージョンが発生した場合、最後に観測した3D位置と最新のTFから、
# 現在のカメラ画像上での2Dバウンディングボックスを「予測」して配信し続けます。
# ★★★ さらに、予測したBBoxを赤色で描画し、
#     '/fruit_tracker_node/predicted_image' トピックに配信します。 ★★★
# =====================================================================================
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_point
import image_geometry
import os
import cv2
from std_msgs.msg import Int32
import time

from rclpy.duration import Duration
import tf_transformations 


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

        self.waypoint_num = 10000 

        self.last_known_3d_pos_base = None    
        self.last_detection_time = self.get_clock().now()
        self.camera_model = None
        self.latest_depth_image = None
        self.latest_color_image_msg = None
        
        self.last_known_bbox_width = None     
        self.last_known_bbox_height = None    
        self.last_known_depth = None          
        self.camera_frame_id = None           
        
        self.latest_bbox = None               
        self.current_tracked_detections = []  

        # True = syuukai_node に「動き出せ」とトリガーを送信済み
        self.trigger_sent_to_syuukai = False 
        # True = アームが動き出した後の「接近中」の検出結果を基準として保存済み
        self.reference_locked = False

        self.create_subscription(Detection2DArray, '/detections_2d', self.detection_callback, 1)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 1)
        self.create_subscription(Int32, 'waypoint_num', self.waypoint_callback, 1)

        self.pub_tracked_bbox = self.create_publisher(Detection2DArray, '/tracked_detections', 1)
        self.pub_fruit_point = self.create_publisher(PointStamped, "/detected_fruits", 1)
        
        # ★★★ 予測画像を描画するためのPublisherを追加 ★★★
        self.pub_predicted_image = self.create_publisher(Image, '/fruit_tracker_node/predicted_image', 1)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # self.timer = self.create_timer(0.05, self.predict_and_publish)
        # self.get_logger().info("FruitTrackerNode (v4.1 PredictImage) has started.")
        self.timer = self.create_timer(0.05, self.predict_and_publish)
        self.get_logger().info("FruitTrackerNode (v4.3 ApproachLock) has started.")

    def color_image_callback(self, msg):
        self.latest_color_image_msg = msg

    def info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)
            self.get_logger().info("Camera model initialized.")

    def depth_callback(self, msg):
        self.latest_depth_image = msg

    def get_robust_depth(self, cv_depth, u, v, radius=5):
        h, w = cv_depth.shape
        umin = max(0, int(u - radius))
        umax = min(w - 1, int(u + radius))
        vmin = max(0, int(v - radius))
        vmax = min(h - 1, int(v + radius))
        
        for r in range(radius + 1):
            for vi in range(max(vmin, int(v-r)), min(vmax, int(v+r)) + 1):
                for ui in range(max(umin, int(u-r)), min(umax, int(u+r)) + 1):
                    if (ui-u)**2 + (vi-v)**2 <= r**2:
                        depth = cv_depth[vi, ui]
                        if depth > 0:
                            return depth / 1000.0, ui, vi 
        return 0, u, v
    
    # ★★★ waypoint_callback (予測テスト用) ★★★
    def waypoint_callback(self, msg):
        if self.waypoint_num != msg.data:
            
            if self.last_known_3d_pos_base is None:
                self.get_logger().warn("Waypoint trigger received, but no initial detection yet. Skipping.")
                self.waypoint_num = msg.data
                return

            time_since_last_detection = self.get_clock().now() - self.last_detection_time
            is_prediction = time_since_last_detection > Duration(seconds=0.5)
            
            # ★ 予測モードの時「のみ」保存する ★
            if is_prediction:
                if self.latest_color_image_msg is not None and len(self.current_tracked_detections) > 0:
                    self.get_logger().info(f"Waypoint {msg.data}: [PREDICTION MODE] Saving annotations...")
                    self.save_image_and_annotations(
                        self.latest_color_image_msg, 
                        self.current_tracked_detections, 
                        msg.data
                    )
                else:
                     self.get_logger().warn(f"Waypoint {msg.data}: [PREDICTION MODE] No image or detections to save.")
            else:
                self.get_logger().info(f"Waypoint {msg.data}: [YOLO DETECTED] Skipping save to test prediction.")

        self.waypoint_num = msg.data


    def detection_callback(self, msg):
        # 1. 検出時刻は「常に」更新する
        self.last_detection_time = self.get_clock().now()
        
        # 2. 2D検出結果も常に更新
        self.latest_bbox = msg
        self.current_tracked_detections = msg.detections 

        # 3. 必要なデータが揃っていなければ処理中断
        if self.latest_depth_image is None or self.camera_model is None or self.latest_color_image_msg is None: 
            self.get_logger().warn("Detection callback skipped: missing data.", throttle_duration_sec=5.0)
            return

        # 4. 検出情報から3D座標と深度を計算
        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
        best_detection = max(msg.detections, key=lambda d: d.bbox.size_x * d.bbox.size_y, default=None)
        if not best_detection: 
            self.get_logger().warn("Detection callback: No detections in message.", throttle_duration_sec=5.0)
            return
        
        try:
            # 1. BBox座標を取得
            bbox = best_detection.bbox
            x1 = int(bbox.center.position.x - bbox.size_x / 2)
            y1 = int(bbox.center.position.y - bbox.size_y / 2)
            x2 = int(bbox.center.position.x + bbox.size_x / 2)
            y2 = int(bbox.center.position.y + bbox.size_y / 2)

            # 2. 画像範囲内にクリッピング
            h, w = cv_depth.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x1 >= x2 or y1 >= y2:
                self.get_logger().warn("Invalid BBox dimensions after clipping, skipping.")
                return

            # 3. BBox内の深度パッチと(u, v)座標を取得
            depth_patch = cv_depth[y1:y2, x1:x2]
            v_indices, u_indices = np.indices(depth_patch.shape)
            u_indices += x1
            v_indices += y1

            # 4. 深度が有効な点(>0)のみをフィルタリング
            valid_mask = depth_patch > 0
            if not np.any(valid_mask):
                self.get_logger().warn("No valid depth points found within the BBox.")
                return
                
            u_valid = u_indices[valid_mask]
            v_valid = v_indices[valid_mask]
            d_valid_mm = depth_patch[valid_mask]
            d_valid_m = d_valid_mm.astype(np.float32) / 1000.0 # メートルに変換

            # 5. 有効なピクセルを3D座標に一括変換 (逆投影)
            # x = (u - cx) * d / fx
            # y = (v - cy) * d / fy
            # z = d
            cx = self.camera_model.cx()
            cy = self.camera_model.cy()
            fx = self.camera_model.fx()
            fy = self.camera_model.fy()

            z_points = d_valid_m
            x_points = (u_valid - cx) * z_points / fx
            y_points = (v_valid - cy) * z_points / fy
            
            # 6. 3D点群の「表面」の重心を計算 (背景ピクセルの影響を除去)
            if z_points.size == 0:
                self.get_logger().warn("No valid Z points after unprojection.")
                return

            # 深度の中央値（Median）を計算
            median_z = np.median(z_points)
            
            # 中央値から一定のしきい値（例: 0.05m = 5cm）以内にある点群のみを「表面」とする
            z_threshold = 0.05 
            surface_mask = np.abs(z_points - median_z) < z_threshold
            
            # もし表面点が一つも見つからなければ、安全策として全ての有効な点を使う
            if not np.any(surface_mask):
                self.get_logger().warn(f"No surface points found near median {median_z:.3f}m. Using all points.")
                x_surface = x_points
                y_surface = y_points
                z_surface = z_points
            else:
                x_surface = x_points[surface_mask]
                y_surface = y_points[surface_mask]
                z_surface = z_points[surface_mask]

            # 「表面」の重心（平均）を計算
            center_x = np.mean(x_surface)
            center_y = np.mean(y_surface)
            center_z = np.median(z_surface) # ★これが背景の影響を受けにくい深度になる
            
            point3d_camera_center = [float(center_x), float(center_y), float(center_z)]

            # 予測計算に使う「重心のZ軸深度」を保存
            center_depth_z = center_z

            # 7. TF変換用にPointStampedを作成
            pt_camera = PointStamped()
            pt_camera.header = msg.header
            pt_camera.point.x, pt_camera.point.y, pt_camera.point.z = point3d_camera_center

            transform = self.tf_buffer.lookup_transform("base_link", pt_camera.header.frame_id, rclpy.time.Time())
            pt_base = do_transform_point(pt_camera, transform) # ★ このpt_baseが「重心」になる
            
        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup/Centroid calculation failed in detection_callback: {e}")
            return

        # ★★★ 6. 状態に応じた処理 ★★★

        # ----- 状態A: 初めての検出 (トリガー送信「待ち」) -----
        if not self.trigger_sent_to_syuukai:
            self.get_logger().info(">>> [STATE A] Initial detection. Arm not yet moving. Waiting for approach detection...")
            self.trigger_sent_to_syuukai = True

        # ----- 状態B: トリガー送信済み & 基準ロック待ち -----
        elif self.trigger_sent_to_syuukai and not self.reference_locked:
            self.get_logger().info(">>> [STATE B] Detection during approach. LOCKING reference and SENDING trigger...")
            
            # ★★★ pt_base (推定された中心座標) がロックされ、Publishされる ★★★
            self.last_known_3d_pos_base = pt_base # 予測基準としてロック
            self.camera_frame_id = msg.header.frame_id
            
            self.last_known_bbox_width = best_detection.bbox.size_x
            self.last_known_bbox_height = best_detection.bbox.size_y
            
            # ★★★ 深度も「中心」までのZ軸深度 (center_depth_z) に更新 ★★★
            self.last_known_depth = center_depth_z
            
            self.pub_fruit_point.publish(pt_base) # 軌道生成の中心
            
            self.reference_locked = True
            
            self.get_logger().info(f"★★★ Reference LOCKED at estimated center (depth: {self.last_known_depth:.3f}m) ★★★")

        # ----- 状態C: 基準ロック済み -----
        elif self.trigger_sent_to_syuukai and self.reference_locked:
            self.get_logger().info(">>> [STATE C] Reference locked. Ignoring further detections.", throttle_duration_sec=1.0)
            pass


    def save_image_and_annotations(self, image_msg, detections, waypoint_num):
        timestamp = time.strftime("%m%d-%H%M%S")
        filename_base = f"{timestamp}_waypoint{waypoint_num}"
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            img_filepath = os.path.join(self.image_save_path, f"{filename_base}.png")
            cv2.imwrite(img_filepath, cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")
            return

        if self.camera_model is None:
             self.get_logger().error("Cannot save annotations: camera_model is None.")
             return
             
        img_w = float(self.camera_model.width)
        img_h = float(self.camera_model.height)
        yolo_lines = []
        for det in detections:
            bbox = det.bbox
            x_center_norm = bbox.center.position.x / img_w
            y_center_norm = bbox.center.position.y / img_h
            width_norm = bbox.size_x / img_w
            height_norm = bbox.size_y / img_h
            class_id = 0 
            yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        if yolo_lines:
            txt_filepath = os.path.join(self.annotation_path, f"{filename_base}.txt")
            try:
                with open(txt_filepath, 'w') as f:
                    f.writelines(yolo_lines)
                self.get_logger().info(f"Saved image/annotation for {filename_base}")
            except IOError as e:
                self.get_logger().error(f"Failed to write annotation file: {e}")

    # ★★★ 予測ロジック ★★★
    def predict_and_publish(self):
        if self.last_known_3d_pos_base is None or self.camera_model is None or \
           self.last_known_bbox_width is None or self.last_known_bbox_height is None or \
           self.last_known_depth is None or self.camera_frame_id is None:
            return 

        time_since_last_detection = self.get_clock().now() - self.last_detection_time
        is_prediction = time_since_last_detection > Duration(seconds=0.5)

        if not is_prediction:
            return 
        
        self.get_logger().info("Predicting BBox...", throttle_duration_sec=1.0)

        try:
            trans_cam_to_base = self.tf_buffer.lookup_transform(
                self.camera_frame_id, "base_link", rclpy.time.Time())
            pt_cam = do_transform_point(self.last_known_3d_pos_base, trans_cam_to_base)
            
            trans_base_to_cam = self.tf_buffer.lookup_transform(
                "base_link", self.camera_frame_id, rclpy.time.Time())

        except (LookupException, ExtrapolationException) as e:
            self.get_logger().warn(f"Prediction TF lookup failed: {e}")
            self.current_tracked_detections = [] 
            return

        (u_center, v_center) = self.camera_model.project3dToPixel(
            (pt_cam.point.x, pt_cam.point.y, pt_cam.point.z)
        )

        q = trans_base_to_cam.transform.rotation
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        v_fwd_base = R[0:3, 2] 
        x_fwd, y_fwd, z_fwd = v_fwd_base
        sin_theta = z_fwd
        cos_theta = np.sqrt(x_fwd**2 + y_fwd**2)

        h_orig_px = self.last_known_bbox_height
        w_orig_px = self.last_known_bbox_width
        d_orig = self.last_known_depth
        d_now = pt_cam.point.z
        
        if d_now <= 0.1: 
            self.get_logger().warn("Prediction depth is too small, skipping.")
            self.current_tracked_detections = []
            return

        scale = d_orig / d_now
        h_p_rotated = abs(h_orig_px * cos_theta) + abs(w_orig_px * sin_theta)
        w_p_rotated = w_orig_px
        h_p_scaled = h_p_rotated * scale
        w_p_scaled = w_p_rotated * scale
        
        detection = Detection2D()
        detection.header.stamp = self.get_clock().now().to_msg()
        detection.header.frame_id = self.camera_frame_id
        
        bbox = BoundingBox2D()
        bbox.center.position.x = float(u_center)
        bbox.center.position.y = float(v_center)
        bbox.size_x = float(w_p_scaled)
        bbox.size_y = float(h_p_scaled)
        detection.bbox = bbox
        
        detection_array = Detection2DArray()
        detection_array.header = detection.header
        detection_array.detections.append(detection)
        
        self.pub_tracked_bbox.publish(detection_array)
        self.current_tracked_detections = detection_array.detections
        
        # # ★★★ ここから予測画像の描画と配信処理 ★★★
        # if self.latest_color_image_msg is None:
        #     return # 描画すべき最新画像がない

        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(self.latest_color_image_msg, "bgr8")
        # except Exception as e:
        #     self.get_logger().error(f"Failed to convert prediction image: {e}")
        #     return

        # # 予測BBoxを描画 (赤色)
        # for det in detection_array.detections:
        #     bbox = det.bbox
        #     x1 = int(bbox.center.position.x - bbox.size_x / 2)
        #     y1 = int(bbox.center.position.y - bbox.size_y / 2)
        #     x2 = int(bbox.center.position.x + bbox.size_x / 2)
        #     y2 = int(bbox.center.position.y + bbox.size_y / 2)
            
        #     # 画面外にはみ出ないようにクリッピング
        #     img_h, img_w, _ = cv_image.shape
        #     x1, y1 = max(0, x1), max(0, y1)
        #     x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            
        #     # 赤色 (BGR)
        #     cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     cv2.putText(cv_image, "PREDICTED", (x1, y1 - 10), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # # 描画した画像をROSトピックとして配信
        # try:
        #     predicted_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        #     # タイムスタンプは予測が実行された時間 (detection_arrayと同じ)
        #     predicted_image_msg.header = detection_array.header 
        #     self.pub_predicted_image.publish(predicted_image_msg)
        # except Exception as e:
        #     self.get_logger().error(f"Failed to publish predicted image: {e}")
        # # ★★★ 描画処理ここまで ★★★


def main(args=None):
    rclpy.init(args=None)
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