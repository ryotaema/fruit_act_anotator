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

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
from tf2_geometry_msgs import do_transform_point
import image_geometry
import os
import cv2
from std_msgs.msg import Int32

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
                self.save_image_and_annotations(self.latest_color_image_msg, self.latest_bbox.detections)
        self.waypoint_num = msg.data


    def detection_callback(self, msg):
        self.latest_bbox = msg

        if self.latest_depth_image is None or self.camera_model is None or self.latest_color_image_msg is None: return

        cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth_image, desired_encoding='passthrough')
        
        best_detection = max(msg.detections, key=lambda d: d.bbox.size_x * d.bbox.size_y, default=None)
        if not best_detection: return
        
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
    def save_image_and_annotations(self, image_msg, detections):
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        # ファイル名のベース部分を作成
        stamp = image_msg.header.stamp
        filename_base = f"{stamp.sec}_{stamp.nanosec}"
        
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
        # 予測時にはファイル保存を行わないように変更
        if self.last_known_3d_pos_base is None or self.camera_model is None: return
        
        # is_prediction = (self.get_clock().now() - self.last_detection_time) > rclpy.duration.Duration(seconds=0.5)
        # if not is_prediction: return # 予測タイミングでないなら何もしない

        # (予測トラッキングのロジックは残すが、ファイル保存は行わない)
        # ... (省略) ...
        pass


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