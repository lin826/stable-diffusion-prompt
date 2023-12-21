import nltk
import spacy
import torch
from nltk.corpus import stopwords
from spacy.lang.en import English
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
from moviepy.editor import VideoFileClip

sp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo").to("cuda")
pipeline_image2video = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt").to("cuda")

def split_sentences(text):
    # Load the English tokenizer from spaCy
    nlp = English()
    nlp.add_pipe('sentencizer')

    # Process the text
    doc = nlp(text)

    # Extract sentences and store them in an array
    sentences = [sent.text.strip() for sent in doc.sents]

    return sentences

def extract_patterns(sentence):
    #keep the need words
    pos_needed = {"VERB", "ADJ", "ADV", "ADP", "NOUN", "NUM"}
    sentence_tags = []
    for word in sp(sentence):
        if word.text.lower() in stop_words and word.pos_ != "ADP":
            continue
        if word.pos_ == "PROPN":
            word.pos_ = "NOUN"
        if word.pos_ in pos_needed:    
            sentence_tags.append((word.text, word.pos_))
    
    # Initialize the list for results
    res = []

    # Iterate over tagged words in the sentence
    i = 0
    while i < len(sentence_tags):
        word, pos = sentence_tags[i]

        # Check for pattern 1: "ADP" + ... + "NOUN"
        if pos == "ADP":
            for j in range(i + 1, len(sentence_tags)):
                if sentence_tags[j][1] == "NOUN":
                    res.append(" ".join([w for w, p in sentence_tags[i:j+1]]))
                    i = j  # Update the index to the end of the pattern
                    break

        # Check for pattern 2: "ADV" + ("VERB") 
        elif pos == "ADV":
            if i < len(sentence_tags) - 1 and sentence_tags[i+1][1] == "VERB":
                res.append(word + " " + sentence_tags[i+1][0])
                i += 1  # Skip the next word as it's already included in the pattern
            else:
                res.append(word)
                
        elif pos == "VERB":
            if i < len(sentence_tags) - 1 and sentence_tags[i+1][1] == "ADV":
                res.append(word + " " + sentence_tags[i+1][0])
                i += 1  # Skip the next word as it's already included in the pattern
            else:
                res.append(word)
            
        # Check for pattern 3: "ADJ" + ... + "NOUN" or "NOUN"
        elif pos == "ADJ":
            has_noun = False
            for j in range(i + 1, len(sentence_tags)):
                if sentence_tags[j][1] == "NOUN":
                    res.append(" ".join([w for w, p in sentence_tags[i:j+1]]))
                    i = j  # Update the index to the end of the pattern
                    has_noun = True
                    break
            if not has_noun:
                res.append(word)
            
        elif pos == "NOUN":
            res.append(word)

        
        i += 1

    return res

def to_prompt(article):
    general_prompt = "best quality,ultra-detailed,masterpiece,hires,8k,"
    sentences = split_sentences(article)

    return general_prompt + ",".join(set(map(extract_patterns, sentences)))


if __name__ == '__main__':
    article = """
    In Africa's vast savannah, a swift cheetah races, epitomizing nature's splendor. Its effortless sprint highlights not just its remarkable speed but also the urgent need to protect these majestic creatures and their diminishing habitats.
    """

    prompt = to_prompt(article)
    print(prompt)

    topic = "cheetah"
    image = pipeline_text2image(prompt=prompt, guidance_scale=0.0, num_inference_steps=10, height=576, width=1024).images[0]
    image.save(f"{topic}.png")

    image = Image.open(f"{topic}.png")
    image = image.resize((1024, 576))  # Resize image to 1024x576
    # pipeline.enable_model_cpu_offload() #if you can use gpu
    generator = torch.manual_seed(42)
    frames = pipeline_image2video(image, decode_chunk_size=8, generator=generator).frames[0]

    export_to_video(frames, f"{topic}.mp4", fps=7)
    
    video = VideoFileClip(f"{topic}.mp4")
    video.write_gif(f"{topic}.gif")