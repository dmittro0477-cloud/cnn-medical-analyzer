
from flask import Flask, render_template, request, send_file, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import cv2

app = Flask(__name__)

# ==========================================
# НАЛАШТУВАННЯ
# ==========================================

MODEL_PATH = "efficientnetb0_pneumonia_finetuned.keras"
IMG_SIZE = (224, 224)

# ==========================================
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ==========================================

print("Завантаження fine-tuned EfficientNetB0...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Модель успішно завантажена!")


# ==========================================
# ОТРИМУЄМО ВНУТРІШНЮ EfficientNetB0
# ==========================================

efficientnet = model.get_layer("efficientnetb0")

last_conv_layer = efficientnet.get_layer("top_activation")

print("Внутрішня EfficientNetB0 знайдена!")
print("Grad-CAM шар:", last_conv_layer.name)
print("Форма Grad-CAM шару:", last_conv_layer.output.shape)


# ==========================================
# СТВОРЮЄМО ОКРЕМУ МОДЕЛЬ ДЛЯ GRAD-CAM
# ==========================================

gradcam_backbone = tf.keras.Model(
    inputs=efficientnet.input,
    outputs=[
        last_conv_layer.output,
        efficientnet.output
    ],
    name="gradcam_backbone"
)

print("Модель для Grad-CAM створена!")


# ==========================================
# ГОЛОВНА СТОРІНКА
# ==========================================

@app.route("/")
def index():
    return send_file("index.html")


# ==========================================
# GRAD-CAM
# ==========================================

def make_gradcam(image_array):

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = gradcam_backbone(
            image_array,
            training=False
        )

        # Повторюємо ті самі наступні шари,
        # які використовує основна модель.
        x = model.get_layer(
            "global_average_pooling2d"
        )(backbone_output)

        x = model.get_layer(
            "dropout"
        )(x, training=False)

        prediction = model.get_layer(
            "dense"
        )(x)

        score = prediction[:, 0]

    gradients = tape.gradient(
        score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM не зміг отримати градієнти."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ==========================================
# СТВОРЕННЯ ЗОБРАЖЕННЯ GRAD-CAM
# ==========================================

def create_gradcam_image(
    original_image,
    heatmap
):

    width, height = original_image.size

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (width, height)
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_resized,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    original_array = np.array(
        original_image
    )

    overlay = cv2.addWeighted(
        original_array,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    result_image = Image.fromarray(
        overlay
    )

    buffer = BytesIO()

    result_image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ==========================================
# АНАЛІЗ
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "Зображення не завантажено"
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "Файл не вибрано"
        }), 400

    try:

        # ==================================
        # ОРИГІНАЛ
        # ==================================

        image = Image.open(
            file
        ).convert("RGB")


        # ==================================
        # ПІДГОТОВКА ДЛЯ МОДЕЛІ
        # ==================================

        image_resized = image.resize(
            IMG_SIZE
        )

        img_array = np.array(
            image_resized,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # ==================================
        # ЗВИЧАЙНИЙ ПРОГНОЗ
        # ==================================

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        pneumonia_probability = float(
            prediction
        )

        normal_probability = (
            1.0 -
            pneumonia_probability
        )


        if pneumonia_probability >= 0.5:

            result = "PNEUMONIA"
            confidence = pneumonia_probability

        else:

            result = "NORMAL"
            confidence = normal_probability


        # ==================================
        # GRAD-CAM
        # ==================================

        print("Створення Grad-CAM...")

        heatmap = make_gradcam(
            img_array
        )

        gradcam_image = create_gradcam_image(
            image,
            heatmap
        )

        print("Grad-CAM успішно створено!")


        # ==================================
        # РЕЗУЛЬТАТ
        # ==================================

        return jsonify({

            "result": result,

            "pneumonia": round(
                pneumonia_probability * 100,
                2
            ),

            "normal": round(
                normal_probability * 100,
                2
            ),

            "confidence": round(
                confidence * 100,
                2
            ),

            "gradcam": gradcam_image

        })


    except Exception as e:

        print(
            "ПОМИЛКА АНАЛІЗУ:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
