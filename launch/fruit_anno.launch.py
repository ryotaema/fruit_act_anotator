import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('fruit_act_anotator')
    
    # Realsenseの設定ファイル
    realsense_config_path = os.path.join(pkg_dir, 'config', 'realsense.yaml')

    # 1. Realsenseカメラノード
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[realsense_config_path],
        output='screen'
    )

    # 2. 2D検出ノード (改造したfruit_localizer)
    detection_node = Node(
        package='fruit_act_anotator',
        executable='detection_node',
        name='detection_node',
        output='screen'
    )

    # 3. 追跡・オクルージョン予測ノード (新規)
    fruit_tracker_node = Node(
        package='fruit_act_anotator',
        executable='fruit_tracker_node',
        name='fruit_tracker_node',
        output='screen',
        parameters=[{
            'annotation_path': os.path.join(os.path.expanduser('~'), 'yolo_annotations', 'labels'),
            'image_save_path': os.path.join(os.path.expanduser('~'), 'yolo_annotations', 'images')
        }]
    )
    
    # 4. 周回軌道実行ノード (既存)
    # C++ノードなのでexecutableはsetup.pyではなくCMakeLists.txtで定義されたもの
    # ここでは仮の名前 `syuukai_node_exec` を指定
    syuukai_node = Node(
        package='fruit_executor', # syuukai_nodeが含まれるパッケージ名
        executable='syuukai',  # CMakeLists.txtのadd_executableで指定した名前
        name='syuukai_node',
        output='screen',
    )

    # TODO: URDFをロードし、robot_state_publisherとrviz2を起動する設定を追加すると完璧
    # robot_state_publisher_node = ...
    # rviz_node = ...

    return LaunchDescription([
        realsense_node,
        detection_node,
        fruit_tracker_node,
        syuukai_node, # syuukai_nodeを同時に起動する場合はコメントアウトを外す
        # robot_state_publisher_node,
        # rviz_node
    ])