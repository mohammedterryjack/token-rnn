from token_rnn.encoder_decoder import ConditionalNLG

model = ConditionalNLG()
model.train(epochs=100)

while True:
    keywords = input("keywords:").split()
    for style in ("question","statement"):
        for sentiment in ("positive","neutral","negative"):
            sentence = model.generate(style,sentiment,keywords)
            print(f"{style} {sentiment} {keywords}:{sentence}")

