# =====================================================================================
# このノードは、ROSトピック経由で受信したカラー画像に対してYOLOv8を実行し、
# 検出した物体の2Dバウンディングボックス情報を配信する、純粋な「検出器」。
# ★★★ さらに、検出結果を画像に描画し、新しいROSトピックに配信します (rqt_image_viewerなどで確認して)。 ★★★
# =====================================================================================
import rclpy
import torch
import os
import cv2 # ★★★ 画像描画と変換のためにOpenCVをインポート ★★★
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # YOLOモデルをロード
        pkg_dir = os.path.join(os.path.expanduser('~'), 'harvesting_robot', 'avoa_ws', 'src', 'fruit_act_anotator')
        model_path = os.path.join(pkg_dir, 'model', 'best.pt')
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO model loaded from: {model_path}")

        # BridgeとPublisher/Subscriber
        self.bridge = CvBridge()
        self.pub_detections = self.create_publisher(Detection2DArray, '/detections_2d', 1)
        # ★★★ 描画済み画像を配信するPublisherを追加 ★★★
        self.pub_annotated_image = self.create_publisher(Image, '/detection_node/annotated_image', 1)

        self.sub_image = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info("DetectionNode has started. Waiting for images...")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image, verbose=False)

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        # ★★★ 描画用の画像コピーを作成 ★★★
        annotated_image = cv_image.copy() 
        
        # クラス名リスト (YOLOモデルに合わせて調整してください)
        # もしYOLOモデルが 'fruit' などのクラス名を持っている場合、以下のように定義します。
        # self.model.names から取得するのが最も正確です。
        class_names = self.model.names if hasattr(self.model, 'names') else ['fruit'] # 仮のクラス名

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()[0] # [x1, y1, x2, y2]
                confidence = box.conf.cpu().numpy()[0]
                class_id = int(box.cls.cpu().numpy()[0])
                
                # vision_msgs.msg.Detection2Dの作成
                detection = Detection2D()
                detection.header = detection_array.header
                
                # image_callback関数内
                bbox = BoundingBox2D()
                # ★★★ すべての計算結果を float() で囲む ★★★
                bbox.center.position.x = float((xyxy[0] + xyxy[2]) / 2.0)
                bbox.center.position.y = float((xyxy[1] + xyxy[3]) / 2.0)
                bbox.size_x = float(xyxy[2] - xyxy[0])
                bbox.size_y = float(xyxy[3] - xyxy[1])
                detection.bbox = bbox
                detection_array.detections.append(detection)

                # ★★★ 画像にバウンディングボックスとラベルを描画 ★★★
                x1, y1, x2, y2 = map(int, xyxy)
                
                # バウンディングボックス
                color = (0, 255, 0) # 緑色
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # ラベル (クラス名と信頼度)
                label = f"{class_names[class_id]}: {confidence:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 検出結果がある場合のみpublish
        if len(detection_array.detections) > 0:
            self.pub_detections.publish(detection_array)

        # ★★★ 描画済み画像をpublish ★★★
        try:
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_image_msg.header = msg.header # 元の画像のヘッダーを使用
            self.pub_annotated_image.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert or publish annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()