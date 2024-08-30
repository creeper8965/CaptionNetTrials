import transformers
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from transformers import ViTModel, ViTConfig, ViTImageProcessorFast

#LEARNT -> BertForMaskedLM also works? 
#Encoder Choice
ViTSMALL = False
FaceDEIT = True
#Decoder Choice
Bert = False
Bart = False
RoBerta = False
TinyBert = False #shit only broke after 2 Ep
BertMLM = True
#Test Model
Test = True


if ViTSMALL:
    #Vision Encoder
    VitPath = 'ViT-small/'
    encoderConf = ViTConfig.from_pretrained(VitPath)
    encoderModel = ViTModel.from_pretrained(VitPath)
    encoderProcessor = ViTImageProcessorFast.from_pretrained(VitPath)

if FaceDEIT:
    DeiTPATH = 'facebook/deit-tiny-patch16-224'
    encoderConf = ViTConfig.from_pretrained(DeiTPATH)
    encoderModel = ViTModel.from_pretrained(DeiTPATH)
    encoderProcessor = ViTImageProcessorFast.from_pretrained(DeiTPATH)

if Bert:
    from transformers import BertConfig, BertTokenizerFast, BertLMHeadModel
    decoderTokenizer = BertTokenizerFast.from_pretrained('BertTokenizer/')
    decoderTokenizer.model_max_length = 324
    decoderConf = BertConfig(vocab_size=decoderTokenizer.vocab_size, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048, hidden_act='gelu', bos_token_id=102, eos_token_id=103, max_position_embeddings=512, is_decoder=True)
    decoderModel = BertLMHeadModel(decoderConf)


#testing of BART which is better for decoder only?
if Bart:
    from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration, BartForCausalLM
    decoderTokenizer = BartTokenizerFast.from_pretrained('BartTokenizer/')
    decoderTokenizer.model_max_length = 324 

    decoderConf = BartConfig(
                        vocab_size=decoderTokenizer.vocab_size,
                        max_position_embeddings=512,
                        encoder_layers=2,
                        encoder_ffn_dim=1024,
                        encoder_attention_heads=4,
                        decoder_layers=6,
                        decoder_ffn_dim=4096,
                        decoder_attention_heads=8,
                        encoder_layerdrop=0,
                        decoder_layerdrop=0,
                        activation_function='swish',
                        d_model=512,
                        dropout=0.12,
                        pad_token_id=1,
                        bos_token_id=0,
                        eos_token_id=2,
                        is_encoder_decoder=False,
                        decoder_start_token_id=2,
                        forced_eos_token_id=2,
                        is_decoder=True,
                        add_cross_attention=False,
                        )
    decoderModel = BartForCausalLM(decoderConf) #.get_decoder()

if RoBerta:
    from transformers import RobertaForCausalLM, RobertaTokenizer, RobertaConfig
    roberta = 'FacebookAI/roberta-base'
    decoderTokenizer = RobertaTokenizer.from_pretrained(roberta)
    decoderTokenizer.model_max_length = 324
    decoderConf = RobertaConfig.from_pretrained(roberta)
    decoderConf.is_decoder=True
    decoderModel = RobertaForCausalLM.from_pretrained(pretrained_model_name_or_path=roberta, config=decoderConf)
    
if TinyBert:
    from transformers import BertConfig, BertTokenizerFast, BertLMHeadModel
    decoderTokenizer = BertTokenizerFast.from_pretrained('BertTokenizer/')
    decoderTokenizer.model_max_length = 324
    decoderConf = BertConfig(vocab_size=decoderTokenizer.vocab_size, hidden_size=324, num_hidden_layers=1, num_attention_heads=9, intermediate_size=768, hidden_act='gelu_fast', bos_token_id=102, eos_token_id=103, max_position_embeddings=324, is_decoder=True)
    decoderModel = BertLMHeadModel(decoderConf)

if BertMLM:
    from transformers import BertForMaskedLM, BertTokenizer, BertConfig
    mobibert = 'google-bert/bert-base-uncased'
    decoderTokenizer = BertTokenizer.from_pretrained(mobibert)
    decoderTokenizer.model_max_length = 324
    decoderConf = BertConfig.from_pretrained(mobibert)
    decoderConf.is_decoder = True
    decoderConf.add_cross_attention = True
    decoderModel = BertForMaskedLM.from_pretrained(mobibert,config=decoderConf)
    ######
    num_decoder_layers = 4
    decoderModel.bert.encoder.layer = decoderModel.bert.encoder.layer[-num_decoder_layers:] #gets last n layers
    decoderConf.num_hidden_layers = num_decoder_layers
    #OPTIONAL - Copies Manga ocr base - 2 layers, FaceDeit


config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=encoderConf, decoder_config=decoderConf)
config.decoder_start_token_id = decoderTokenizer.cls_token_id
config.pad_token_id = decoderTokenizer.pad_token_id
config.tie_word_embeddings = False
model = VisionEncoderDecoderModel(config=config, encoder=encoderModel, decoder=decoderModel)

print(model)
print(model.get_memory_footprint() /1000 /1000,'MB')

model.save_pretrained('NewModel/',safe_serialization=False)
decoderTokenizer.save_pretrained('NewModel/')
encoderProcessor.save_pretrained('NewModel/')

if Test:
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessorFast
    from PIL import Image
    print('Test Generation')
    img = Image.open('face.jpeg')
    model = VisionEncoderDecoderModel.from_pretrained('NewModel/')
    encoderProcessor = ViTImageProcessorFast.from_pretrained('NewModel/')
    decoderTokenizer = AutoTokenizer.from_pretrained('NewModel/')
    img = encoderProcessor(img, return_tensors='pt').pixel_values
    captionToks = model.generate(img, num_beams=3)
    print([decoderTokenizer.decode(t, skip_special_tokens=False) for t in captionToks])

