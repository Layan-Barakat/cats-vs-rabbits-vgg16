import argparse, json, numpy as np
from keras.models import load_model
from keras.preprocessing import image

def main(img_path):
    model = load_model("Cats-VS-Rabbits_Output/final.keras")
    with open("Cats-VS-Rabbits_Output/labels.json") as f:
        class_names = json.load(f)

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"Prediction: {class_names[idx]} — probs={probs}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True, help="Path to a JPG/PNG")
    args = p.parse_args()
    main(args.img)
