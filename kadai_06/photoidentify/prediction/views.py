import logging  # 追加

# ログ設定
logger = logging.getLogger(__name__)  # 追加
logger.setLevel(logging.INFO)  # 追加

# ログ出力の設定
handler = logging.StreamHandler()  # コンソール出力用のハンドラを追加
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ライブラリのインポート
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from io import BytesIO
import os
import numpy as np


# predict関数の作成
def predict(request):
    # リクエスト処理
    if request.method == "GET":
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        # ログの追加: フォームデータを確認
        logger.info(f"Form data: {request.POST}, {request.FILES}")

        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())

            # 画像の前処理
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # モデルの読み込みと予測
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'templates', 'models', 'vgg16.h5')
            model = load_model(model_path)
            predictions = model.predict(img_array)

            # モデルファイルの存在確認
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'templates', 'models', 'vgg16.h5')
            if not os.path.exists(model_path):
                error_message = f"モデルファイルが見つかりません: {model_path}"
                return render(request, 'home.html', {'form': form, 'error': error_message})
            
            # モデルの読み込みとコンパイル
            model = load_model(model_path)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # 予測
            predictions = model.predict(img_array)

            # 予測結果デコード
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            # 予測結果のログ出力
            logger.info(f'Decoded predictions: {decoded_predictions}')

            # 結果の整形
            results = [
                {"class": pred[1], "probability": f"{pred[2]*100:.2f}%"}
            for pred in decoded_predictions
            ]

            logger.info(f'Formatted results: {results}')
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'results': results, 'img_data': img_data})
        else:
            # フォームが無効な場合の処理
            return render(request, 'home.html', {form: form, 'error': 'Invalid form submission.'})
    else:
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

