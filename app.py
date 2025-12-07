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
from deepface import DeepFace

# =================================================
# ⚙️ 設定エリア
# =================================================
MODEL_FILE_NAME = "best_sign_model.keras" 
CLASS_NAMES = ["Label 1", "Label 2", "Label 3", "Label 4"] 

# 表情定義
EMOTION_DATA = {
    "neutral":  (" . _ . ", "MAGAO"),
    "happy":    ("^ v ^",   "URESHII"),
    "surprise":("O . O !", "BIKKURI"),
    "sad":      ("T . T",   "KANASHII"),
    "angry":    ("> _ < #", "OKOTTERU"),
    "fear":     ("; O O ;", "KOWAI"),
    "disgust":  ("...",     "IYA"),
}

# =================================================
# Attention層
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
st.sidebar.info("チェックを入れると、右側にAIの解析データが表示されます。")

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
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        # --- DeepFace (10フレームに1回) ---
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            try:
                objs = DeepFace.analyze(
                    img_path=img, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                self.last_emotion_key = objs[0]['dominant_emotion']
                data = EMOTION_DATA.get(self.last_emotion_key, ("?", "?"))
                self.kaomoji = data[0]
                self.romaji = data[1]
            except Exception:
                pass

        # --- MediaPipe & Model ---
        results = self.holistic.process(img_rgb)
        
        has_pose = results.pose_landmarks is not None
        has_lh = results.left_hand_landmarks is not None
        has_rh = results.right_hand_landmarks is not None
        
        self.status_text = f"P[{'O' if has_pose else 'X'}] L[{'O' if has_lh else 'X'}] R[{'O' if has_rh else 'X'}]"

        if not has_pose:
            self.warning_msg = "STEP BACK!"
            self.probs = self.probs * 0.9 
            if self.result_conf > 0: self.result_conf *= 0.9
        else:
            self.warning_msg = ""
            if model is not None:
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if has_lh else np.zeros((21, 3))
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if has_rh else np.zeros((21, 3))

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
        # 4. 描画分岐 (レイアウト修正版)
        # ---------------------------------------------------------
        
        # 【A】デバッグモード（右パネル表示）
        if self.debug:
            # 骨格描画
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # パネル作成
            panel_w = 320
            canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
            canvas[:h, :w] = img

            x_start = w + 10
            y_cursor = 30 # 初期位置を少し上に修正

            # --- 1. System Status ---
            cv2.putText(canvas, "System Status", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 25 # 行間を詰める
            p_color = (0, 255, 0) if has_pose else (0, 0, 255)
            cv2.putText(canvas, self.status_text, (x_start, y_cursor), font, 0.5, p_color, 1)
            y_cursor += 25
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 25

            # --- 2. Sign Result ---
            cv2.putText(canvas, "Sign Prediction:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 30
            cv2.putText(canvas, self.result_label, (x_start, y_cursor), font, 0.9, (0, 255, 255), 2)
            y_cursor += 25
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 25

            # --- 3. Face Emotion (修正: 横並びで1行に) ---
            cv2.putText(canvas, "Face Emotion:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 30
            
            # 顔文字を左に
            cv2.putText(canvas, self.kaomoji, (x_start, y_cursor), font, 0.8, (255, 255, 255), 2)
            # ローマ字を右に配置（+130pxずらす）
            cv2.putText(canvas, self.romaji, (x_start + 130, y_cursor), font, 0.6, (0, 255, 255), 1)
            
            y_cursor += 25
            cv2.line(canvas, (w, y_cursor), (w+panel_w, y_cursor), (100, 100, 100), 1)
            y_cursor += 25

            # --- 4. Probabilities (修正: ぎゅっと詰める) ---
            cv2.putText(canvas, "Probabilities:", (x_start, y_cursor), font, 0.6, (200, 200, 200), 1)
            y_cursor += 20
            
            bar_max_width = 180
            for i, prob in enumerate(self.probs):
                class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
                
                # 文字を小さく、行間を狭く
                y_cursor += 15 
                cv2.putText(canvas, f"{class_name}", (x_start, y_cursor), font, 0.45, (255, 255, 255), 1)
                
                y_bar = y_cursor + 5
                # バーの背景
                cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_max_width, y_bar + 8), (50, 50, 50), -1)
                # バー本体
                bar_w = int(prob * bar_max_width)
                bar_color = (0, 0, 255) if prob == max(self.probs) else (0, 255, 0)
                if bar_w > 0:
                    cv2.rectangle(canvas, (x_start, y_bar), (x_start + bar_w, y_bar + 8), bar_color, -1)
                
                # パーセント表示
                cv2.putText(canvas, f"{prob*100:.0f}%", (x_start + bar_max_width + 5, y_bar + 7), font, 0.4, (200, 200, 200), 1)
                
                y_cursor += 15 # 次のグラフへの間隔

            # 警告表示
            if self.warning_msg:
                cv2.rectangle(canvas, (50, h//2 - 40), (w-50, h//2 + 40), (0, 0, 255), 2)
                cv2.rectangle(canvas, (52, h//2 - 38), (w-52, h//2 + 38), (0, 0, 0), -1)
                text_size = cv2.getTextSize(self.warning_msg, font, 2.0, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(canvas, self.warning_msg, (text_x, h//2 + 10), font, 2.0, (0, 0, 255), 3)

            return canvas

        # 【B】通常モード（シンプル表示）
        else:
            # 左上：手話結果
            cv2.putText(img, f"Result: {self.result_label}", (10, 50), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 右上：表情結果 (横並びにはしないで見やすく)
            # 顔文字
            cv2.putText(img, self.kaomoji, (w - 200, 50), font, 1.0, (0, 0, 0), 4)
            cv2.putText(img, self.kaomoji, (w - 200, 50), font, 1.0, (255, 255, 255), 2)
            # ローマ字
            cv2.putText(img, self.romaji, (w - 200, 90), font, 0.7, (0, 0, 0), 4)
            cv2.putText(img, self.romaji, (w - 200, 90), font, 0.7, (0, 255, 255), 2)

            if self.warning_msg:
                 cv2.putText(img, self.warning_msg, (50, h//2), font, 2.0, (0, 0, 255), 3)

            return img

# ------------------------------------------------
# アプリ画面構成
# ------------------------------------------------
st.title("AI 手話 & 表情分析")
st.write("統合テスト v2: レイアウト調整版")

if model is None:
    st.error("モデルが読み込めませんでした。")
else:
    webrtc_streamer(
        key=f"sign-language-layout-{DEBUG_MODE}",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
