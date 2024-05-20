mkdir models
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/pytorch_model.bin?download=true -o models/pytorch_model.bin
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer.json?download=true -o models/tokenizer.json
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/vocab.json?download=true -o models/vocab.json
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/config.json?download=true -o models/config.json
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer_config.json?download=true -o models/tokenizer_config.json
