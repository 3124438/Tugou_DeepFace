import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av
from deepface import DeepFace # 追加

# =================================================
# ⚙️ 設定エリア
# =================================================
MODEL_FILE_NAME = "best_sign_model.keras" # 手話モデル
CLASS_NAMES = ["Label 1", "Label 2", "Label 3", "Label 4"] # 手話ラベル

# 表情（顔文字とローマ字）の定義
EMOTION_DATA = {
    "neutral":  (" . _ . ", "MAGAO"),
    "happy":    ("^ v ^",   "URESHII"),
    "surprise": ("O . O !", "BIKKURI"),
    "sad":      ("T . T",   "KANASHII"),
    "angry":    ("> _ < #", "OKOTTERU"),
    "fear":     ("; O O ;", "KOWAI"),
    "disgust":  ("...",     "IYA"),
}

# =================================================
# Attention層 (モデル読み込み用)
# =================================================
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE_NAME, custom_objects={'Attention': Attention})

try:
    model = load_model()
except Exception as e:
    st.error(f"エラー: {e}")
    model = None

# MediaPipe設定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =================================================
# 🎛️ UIサイドバー
# =================================================
st.sidebar.title("System Control")
DEBUG_MODE = st.sidebar.checkbox("デバッグモード（詳細表示）", value=False)
st.sidebar.write("---")
st.sidebar.info("チェックを入れると、骨格や詳細データが表示されます。")

# ------------------------------------------------
# 映像処理クラス
# ------------------------------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 手話用変数
        self.sequence = deque(maxlen=30)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.probs = np.zeros(len(CLASS_NAMES)) 
        self.result_label = "Waiting..."
        self.result_conf = 0.0
        self.status_text = "Init..."
        self.debug = DEBUG_MODE
        self.warning_msg = "" 

        # 表情用変数
        self.frame_count = 0
        self.last_emotion_key = "neutral"
        self.kaomoji = " . _ . "
        self.romaji = "MAGAO"

    def transform(self, frame):
        # 1. 画像取得
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ---------------------------------------------------------
        # 2. 表情分析 (DeepFace) - 10フレームに1回実行
        # ---------------------------------------------------------
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            try:
                # actions=['emotion']のみ指定して軽量化
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['emotion'], 
                    enforce_detection=False, # 顔が見つからなくてもエラーにしない
                    detector_backend='opencv' # 軽量な検出器
                )
                self.last_emotion_key = objs[0]['dominant_emotion']
                
                # 辞書から顔文字とローマ字を取得
                data = EMOTION_DATA.get(self.last_emotion_key, ("?", "?"))
                self.kaomoji = data[0]
                self.romaji = data[1]
                
            except Exception:
                pass # エラー時は前の表情を維持

        # ---------------------------------------------------------
        # 3. 手話分析 (MediaPipe + Model)
        # ---------------------------------------------------------
        results = self.holistic.process(img_rgb)
        
        has_pose = results.pose_landmarks is not None
        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None
        
        self.status_text = f"P[{'O' if has_pose else 'X'}] L[{'O' if has_lh else 'X'}] R[{'O' if has_rh else 'X'}]"

        # 張り付き防止対策
        if not has_pose:
            self.warning_msg = "STEP BACK!"
            self.probs = self.probs * 0.9 
            if self.result_conf > 0: self.result_conf *= 0.9
        else:
            self.warning_msg = ""
            
            if model is not None:
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
                if has_lh:
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
                else:
                    lh = np.zeros((21, 3))
                if has_rh:
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
                else:
                    rh = np.zeros((21, 3))

                # 正規化
                if np.sum(pose) != 0:
                    left_shoulder = pose[11]
                    right_shoulder = pose[12]
                    center = (left_shoulder + right_shoulder) / 2.0
                    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                    if shoulder_width < 0.01: shoulder_width = 1.0
                else:
                    center = np.zeros(3)
                    shoulder_width = 1.0

                pose_norm = (pose - center) / shoulder_width
                lh_norm = (lh - center) / shoulder_width
                rh_norm = (rh - center) / shoulder_width

                keypoints = np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])
                self.sequence.append(keypoints)

                if len(self.sequence) == 30:
                    input_data = np.expand_dims(list(self.sequence), axis=0)
                    try:
                        prediction = model.predict(input_data, verbose=0)
                        self.probs = prediction[0]
                        idx = np.argmax(self.probs)
                        self.result_conf = self.probs[idx]
                        
                        if idx < len(CLASS_NAMES):
                            self.result_label = CLASS_NAMES[idx]
                        else:
                            self.result_label = f"Class {idx}"
                    except Exception:
                        pass

        # ---------------------------------------------------------
        # 4. 描画分岐
        # ---------------------------------------------------------
        
        # 【A】デバッグモードの場合（コックピット表示）
        if self.debug:
            # 骨格を描画
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # ダッシュボード作成
            panel_w = 320
            canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            canvas[:h, :w] = img

            # ダッシュボード情報描画
            x_start = w + 10
            y_cursor = 40
            
            # --- 手話セクション ---
            cv2.putText(canvas, "AI Analysis", (x_start, y_cursor), font, 0.8, (255, 255, 255), 2)
            y_cursor += 40
            
            p_color = (0, 255, 0) if has_pose else (0, 0, 255)
            cv2.putText(canvas, self.status_text, (x_start, y_cursor), font, 0.5, p_color, 1)
            y_cursor += 40
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 30

            cv2.putText(canvas, "Sign Result:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 35
            cv2.putText(canvas, self.result_label, (x_start, y_cursor), font, 1.0, (0, 255, 255), 2)
            y_cursor += 30
            
            # --- 表情セクション (追加) ---
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 30
            cv2.putText(canvas, "Face Emotion:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 40
            # 顔文字
            cv2.putText(canvas, self.kaomoji, (x_start, y_cursor), font, 1.0, (255, 255, 255), 2)
            y_cursor += 30
            # ローマ字
            cv2.putText(canvas, self.romaji, (x_start, y_cursor), font, 0.7, (0, 255, 255), 1)
            y_cursor += 30

            # --- 確率バー ---
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 30
            cv2.putText(canvas, "Probabilities:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 20
            bar_max_width = 180
            for i, prob in enumerate(self.probs):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
                y_cursor += 20
                cv2.putText(canvas, f"{class_name}", (x_start, y_cursor), font, 0.5, (255, 255, 255), 1)
                y_bar = y_cursor + 5
                cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_max_width, y_bar + 10), (50, 50, 50), -1)
                bar_w = int(prob * bar_max_width)
                bar_color = (0, 0, 255) if prob == max(self.probs) else (0, 255, 0)
                if bar_w > 0:
                    cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_w, y_bar + 10), bar_color, -1)
                cv2.putText(canvas, f"{prob*100:.0f}%", (x_start + bar_max_width + 10, y_bar + 8), font, 0.4, (200, 200, 200), 1)
                y_cursor += 20

            # 警告表示（中央）
            if self.warning_msg:
                cv2.rectangle(canvas, (50, h//2 - 40), (w-50, h//2 + 40), (0, 0, 255), 2)
                cv2.rectangle(canvas, (52, h//2 - 38), (w-52, h//2 + 38), (0, 0, 0), -1)
                text_size = cv2.getTextSize(self.warning_msg, font, 2.0, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(canvas, self.warning_msg, (text_x, h//2 + 10), font, 2.0, (0, 0, 255), 3)

            return canvas

        # 【B】通常モードの場合（シンプル表示）
        else:
            # 1. 手話の結果（左上）
            cv2.putText(img, f"Result: {self.result_label}", (10, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 2. 表情の結果（右上に顔文字とローマ字）
            # 顔文字
            cv2.putText(img, self.kaomoji, (w - 200, 50), font, 1.0, (0, 0, 0), 4) # 黒フチ
            cv2.putText(img, self.kaomoji, (w - 200, 50), font, 1.0, (255, 255, 255), 2) # 白文字
            
            # ローマ字（その下）
            cv2.putText(img, self.romaji, (w - 200, 90), font, 0.7, (0, 0, 0), 4)
            cv2.putText(img, self.romaji, (w - 200, 90), font, 0.7, (0, 255, 255), 2)

            # 警告表示
            if self.warning_msg:
                 cv2.putText(img, self.warning_msg, (50, h//2), font, 2.0, (0, 0, 255), 3)

            return img

# ------------------------------------------------
# アプリ画面構成
# ------------------------------------------------
st.title("AI 手話 & 表情分析")
st.write("手話モデルと表情(DeepFace)の統合テスト")

if model is None:
    st.error("モデルが読み込めませんでした。")
else:
    webrtc_streamer(
        key=f"sign-language-unified-{DEBUG_MODE}",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
