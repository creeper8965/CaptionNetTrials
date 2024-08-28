import transformers
from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration
import torch

tokenizer = BartTokenizerFast.from_pretrained('BartTokenizer/')
tokenizer.model_max_length = 512
tokenizer.save_pretrained('En-Tr-Model/')

config = BartConfig(max_position_embeddings=512,
                    encoder_layers=2,
                    encoder_ffn_dim=1024,
                    encoder_attention_heads=4,
                    decoder_layers=2,
                    decoder_ffn_dim=1024,
                    decoder_attention_heads=4,
                    encoder_layerdrop=0,
                    decoder_layerdrop=0,
                    activation_function='gelu_fast',
                    d_model=512,
                    dropout=0.12,
                    pad_token_id=1,
                    bos_token_id=0,
                    eos_token_id=2,
                    is_encoder_decoder=True,
                    decoder_start_token_id=2,
                    forced_eos_token_id=2,
                    )
model = BartForConditionalGeneration(config)
print(model)
model.save_pretrained('En-Tr-Model/')


