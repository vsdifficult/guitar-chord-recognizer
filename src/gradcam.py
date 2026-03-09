from interpretability.gradcam import save_overlay


if __name__ == "__main__":
    out = save_overlay("models/guitar_chord_model.keras", "sample.jpg", method="gradcam")
    print(out)
