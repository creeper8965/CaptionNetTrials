import transformers
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import ViTModel, ViTConfig, ViTImageProcessorFast

#Decoder Choice
Bert = False
Bart = True
#Test Model
Test = True


#Vision Encoder
encoderConf = ViTConfig.from_pretrained('ViT-tiny/')
encoderModel = ViTModel.from_pretrained('ViT-tiny/')
encoderProcessor = ViTImageProcessorFast.from_pretrained('ViT-tiny/')


if Bert:
    from transformers import BertConfig, BertTokenizerFast, BertLMHeadModel
    decoderTokenizer = BertTokenizerFast.from_pretrained('SmallBertUncased/')
    decoderTokenizer.model_max_length = 512
    decoderConf = BertConfig(vocab_size=decoderTokenizer.vocab_size, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048, hidden_act='gelu', bos_token_id=102, eos_token_id=103, max_position_embeddings=512, is_decoder=True)
    decoderModel = BertLMHeadModel(decoderConf)


#testing of BART which is better for decoder only?
if Bart:
    from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration, BartForCausalLM
    decoderTokenizer = BartTokenizerFast.from_pretrained('BartTokenizer/')
    decoderTokenizer.model_max_length = 512 

    decoderConf = BartConfig(
                        vocab_size=decoderTokenizer.vocab_size,
                        max_position_embeddings=512,
                        encoder_layers=2,
                        encoder_ffn_dim=1024,
                        encoder_attention_heads=4,
                        decoder_layers=5,
                        decoder_ffn_dim=2048,
                        decoder_attention_heads=8,
                        encoder_layerdrop=0,
                        decoder_layerdrop=0,
                        activation_function='gelu_fast',
                        d_model=512,
                        dropout=0.12,
                        pad_token_id=1,
                        bos_token_id=0,
                        eos_token_id=2,
                        is_encoder_decoder=False,
                        decoder_start_token_id=2,
                        forced_eos_token_id=2,
                        is_decoder=True
                        )
    decoderModel = BartForCausalLM(decoderConf) #.get_decoder()

config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoderConf, decoder_config=decoderConf)
model = VisionEncoderDecoderModel(config=config, encoder=encoderModel, decoder=decoderModel)

print(model)
print(model.get_memory_footprint() /1000 /1000,'MB')

model.save_pretrained('NewModel/') #,safe_serialization=False)
decoderTokenizer.save_pretrained('NewModel/')
encoderProcessor.save_pretrained('NewModel/')

if Test:
    from PIL import Image
    print('Test Generation')
    img = Image.open('face.jpeg')
    img = encoderProcessor(img, return_tensors='pt').pixel_values
    captionToks = model.generate(img, num_beams=3)
    print([decoderTokenizer.decode(t, skip_special_tokens=False) for t in captionToks])

