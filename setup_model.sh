mkdir models
curl -L https://huggingface.co/FacebookAI/roberta-base/resolve/main/pytorch_model.bin?download=true -o models/pytorch_model.bin
curl -L https://huggingface.co/FacebookAI/roberta-base/blob/main/tokenizer.json?download=true -o models/spm.model
curl -L https://huggingface.co/FacebookAI/roberta-base/blob/main/vocab.json?download=true -o models/config.json
curl -L https://huggingface.co/FacebookAI/roberta-base/blob/main/config.json?download=true -o models/config.json
curl -L https://huggingface.co/FacebookAI/roberta-base/blob/main/tokenizer_config.json?download=true -o models/config.json
