!pip install bitsandbytes

!pip install accelerate

!pip install langdetect

!pip install translate

from transformers import  (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TextStreamer,pipeline)
import torch
model_name = "Hemanth-thunder/Tamil-Mistral-7B-Instruct-v0.1"

nf4_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',quantization_config=nf4_config,use_cache=False,low_cpu_mem_usage=True )
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
streamer = TextStreamer(tokenizer)

prompt_template ="""சரியான பதிலுடன் வேலையை வெற்றிகரமாக முடிக்க, வழங்கப்பட்ட வழிகாட்டுதல்களைப் பின்பற்றி, தேவையான தகவலை உள்ளிடவும்.

### Instruction:
{}

### Response:"""
def create_prompt(query,prompt_template=prompt_template):
    bos_token = "<s>"
    eos_token = "</s>"
    if query:
        lang_code = detect(query)
        print(lang_code)
        if lang_code != "ta":
            print("Query is in {} language so translating to 'ta'".format(lang_code))
            query = translator.translate(query)
            print("Translated-->",query)
        prompt_template = prompt_template.format(query)
    else:
        raise "Please input with query"
    prompt = bos_token+prompt_template #eos_token
    return prompt

from langdetect import detect

pipe = pipeline("text-generation" ,model=model, tokenizer=tokenizer ,do_sample=True, repetition_penalty=1.15,top_p=0.95,streamer=streamer)
prompt = create_prompt("4 + 5 சேர்க்கவும்")
result=pipe(prompt,max_length=512,pad_token_id=tokenizer.eos_token_id)

from translate import Translator
translator= Translator(to_lang="ta")
prompt = create_prompt("எப்படி இருக்கிறீர்கள்?")
result=pipe(prompt,max_length=512,pad_token_id=tokenizer.eos_token_id)
print("-------")
print("Finished")

from translate import Translator
translator= Translator(to_lang="ta")
prompt = create_prompt("Tell me a story")
result=pipe(prompt,max_length=512,pad_token_id=tokenizer.eos_token_id)
print("-------")
print("Finished")

