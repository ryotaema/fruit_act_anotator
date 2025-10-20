import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import os
import time

class AnnotatorNode(Node):
    def __init__(self):
        super().__init__('annotator_node')

        # パラメータの宣言と取得
        self.declare_parameter('output_path', '/home/ryota/harvesting_robot/avoa_ws/src/fruit_act_anotator/annotated_images')
        output_path_str = self.get_parameter('output_path').get_parameter_value().string_value
        # '~' をホームディレクトリに展開
        self.output_path = os.path.expanduser(output_path_str)

        # 保存先ディレクトリが存在しない場合は作成
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.get_logger().info(f'Created output directory: {self.output_path}')

        # CvBridgeの初期化
        self.bridge = CvBridge()

        # 最新の画像と検出結果を保持する変数
        self.latest_image = None
        self.latest_detections = None

        # サブスクライバーの作成
        # Realsenseからの画像トピック (トピック名は環境に合わせて変更してください)
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        # 物体検出結果のトピック (これも既存のシステムに合わせて変更してください)
        self.detection_subscriber = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )

        # 処理用のタイマー (0.1秒ごとに実行)
        self.processing_timer = self.create_timer(0.1, self.process_and_save)
        
        self.get_logger().info('Annotator node has been started.')
        self.get_logger().info(f'Images will be saved to: {self.output_path}')


    def image_callback(self, msg):
        """画像トピックのコールバック関数"""
        self.latest_image = msg

    def detection_callback(self, msg):
        """検出結果トピックのコールバック関数"""
        self.latest_detections = msg

    def process_and_save(self):
        """画像と検出結果を処理し、アノテーション画像を保存する関数"""
        # 画像と検出結果の両方が受信されているか確認
        if self.latest_image is None or self.latest_detections is None:
            return

        try:
            # ROS ImageメッセージをOpenCV画像に変換
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # 描画用に画像をコピー
        annotated_image = cv_image.copy()

        # 検出結果（バウンディングボックス）を描画
        for detection in self.latest_detections.detections:
            # バウンディングボックスの中心座標とサイズを取得
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y

            # バウンディングボックスの左上と右下の座標を計算
            pt1_x = int(center_x - size_x / 2)
            pt1_y = int(center_y - size_y / 2)
            pt2_x = int(center_x + size_x / 2)
            pt2_y = int(center_y + size_y / 2)
            
            # バウンディングボックスを描画 (緑色, 太さ2)
            cv2.rectangle(annotated_image, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)

            # クラスIDやスコアなどをラベルとして描画 (オプション)
            if len(detection.results) > 0:
                class_id = detection.results[0].id
                score = detection.results[0].score
                label = f'ID:{class_id} {score:.2f}'
                cv2.putText(annotated_image, label, (pt1_x, pt1_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ファイル名を生成 (現在のUNIXタイムスタンプを使用)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        milliseconds = f"{int(time.time() * 1000) % 1000:03d}"
        filename = f"annotated_{timestamp}_{milliseconds}.png"
        save_path = os.path.join(self.output_path, filename)
        
        # # 画像を保存
        # cv2.imwrite(save_path, annotated_image)
        # self.get_logger().info(f'Saved annotated image to {save_path}')

        # 処理が終わったら、次のデータが来るまでNoneにしておくことで重複保存を防ぐ
        self.latest_image = None
        self.latest_detections = None


def main(args=None):
    rclpy.init(args=args)
    node = AnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()