# rex_omni_ros

[Rex-Omni](https://github.com/IDEA-Research/Rex-Omni)（物体検出・ポインティング・キーポイント検出・OCRなどを行う3BのマルチモーダルLLM）をROSサービスとして提供するパッケージ。単一のコードベースでROS1 NoeticとROS2 Humbleの両方に対応し、推論にはインプロセスのvLLMを用いる。

## 必要環境

- Linux x86_64
- NVIDIA GPU（VRAM 6GB以上, CUDA 12.6対応）
- [Pixi](https://pixi.sh)

## インストール

他のノードからサービスを利用するには、サービス定義パッケージ`rex_omni_msgs`を既存のワークスペースに加えてビルドする。

```sh
cd <ROSワークスペース>/src
git clone https://github.com/0xNOY/rex_omni_ros.git
cd ..
colcon build # ROS1の場合は `catkin_make`
```

## ノードの起動

```sh
cd <ROSワークスペース>/src/rex_omni_ros
pixi run -e ros2 launch-server # ROS1の場合は `-e ros1`
```

起動時にモデルをロードする（数十秒。初回は[~3GBのモデル](https://huggingface.co/0xNOY/Rex-Omni-AWQ-QLMHead)をHugging Faceからダウンロードする）。起動後、以下のサービスを提供する。パラメータは`rex_omni_ros/config/`を参照。

| サービス | 型 | タスク |
|---|---|---|
| `/rex_omni/detect` | `rex_omni_msgs/Detect` | 物体検出・参照表現・GUIグラウンディング |
| `/rex_omni/point` | `rex_omni_msgs/Point` | ポインティング |
| `/rex_omni/detect_with_visual_prompt` | `rex_omni_msgs/DetectWithVisualPrompt` | 参照ボックスに類似した物体の検出 |
| `/rex_omni/detect_keypoints` | `rex_omni_msgs/DetectKeypoints` | 人・動物のキーポイント検出 |
| `/rex_omni/recognize_text` | `rex_omni_msgs/RecognizeText` | OCR |
| `/rex_omni/sleep` | `std_srvs/Trigger` | モデルをVRAMからRAMへ退避 |
| `/rex_omni/wake_up` | `std_srvs/Trigger` | モデルをVRAMへ復帰 |

認識結果を描画したデバッグ画像を`/rex_omni/debug_image`として発行する。描画は購読者がいるときのみ行われるため通常運用ではコストがかからない（`publish_debug_image: false`で完全に無効化できる）。

## 使用例

物体検出の例

```python
# まずはヘルパー関数を定義
from PIL import Image as PILImage
from sensor_msgs.msg import Image

def to_image_msg(img: PILImage.Image) -> Image:
    """PIL Imageを`sensor_msgs/Image`に変換する"""
    img = img.convert("RGB")
    msg = Image()
    msg.height, msg.width = img.height, img.width
    msg.encoding = "rgb8"
    msg.step = img.width * 3
    msg.data = img.tobytes()
    return msg
```

```python
# ROS2の場合
import rclpy
from rex_omni_msgs.srv import Detect

rclpy.init()
node = rclpy.create_node("detect_client")
client = node.create_client(Detect, "/rex_omni/detect")
client.wait_for_service()

request = Detect.Request()
request.image = to_image_msg(PILImage.open("photo.jpg"))
request.categories = ["person", "dog"]

future = client.call_async(request)
rclpy.spin_until_future_complete(node, future)
for d in future.result().detections:
    print(d.category, d.confidence, d.bbox)
```

```python
# ROS1の場合
import rospy
from rex_omni_msgs.srv import Detect, DetectRequest

rospy.init_node("detect_client")
rospy.wait_for_service("/rex_omni/detect")
detect = rospy.ServiceProxy("/rex_omni/detect", Detect)

request = DetectRequest()
request.image = to_image_msg(PILImage.open("photo.jpg"))
request.categories = ["person", "dog"]

for d in detect(request).detections:
    print(d.category, d.confidence, d.bbox)
```

可視化付き機能のクライアントは`rex_omni_ros/examples/detect_client.py`にある。
