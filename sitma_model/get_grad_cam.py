import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def get_gradcam(model, img_path, layer_name):
    # resmi alıyoruz ve senin dediğin gibi iki yüz elli beşe bölüyoruz ve işlemi yaptık
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # aktivasyon sönümlenmesini engellemek için son katmandan önceki ham değerleri alıyoruz ve yaptık
    # modelin son katmanının adını otomatik buluyoruz ve işlemi yapıyoruz
    last_layer = model.layers[-1]
    
    # ara modelimizi kuruyoruz ve yaptık
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, last_layer.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        # eğer model tek çıkışlıysa yani sigmoid ise işlemi ona göre yapıyoruz ve yaptık
        if predictions.shape[-1] == 1:
            # skora göre hangi tarafı görselleştireceğimizi seçiyoruz ve yaptık
            if predictions[0] < 0.5:
                loss = 1.0 - predictions[:, 0]
            else:
                loss = predictions[:, 0]
        else:
            # eğer iki çıkışlıysa yani softmax ise argmax ile en yüksek olanı alıyoruz ve yaptık
            top_class_idx = tf.argmax(predictions[0])
            loss = predictions[:, top_class_idx]

    # gradyanları hesaplıyoruz ve yaptık
    grads = tape.gradient(loss, conv_outputs)
    
    # sadece pozitif etkileri alıyoruz ve yaptık
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # ısı haritasını oluşturuyoruz ve yaptık
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # görseli canlandırmak için normalizasyon yapıyoruz ve yaptık
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
        
    return heatmap, predictions[0].numpy()
