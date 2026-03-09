from inference.predict import predict


if __name__ == "__main__":
    chord, conf, _ = predict("sample.jpg")
    print(chord, conf)
