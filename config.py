from email.mime import image
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os
import yaml

def create(source_folder):
    images = []
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(source_folder, filename)
                images.append(file_path)
    return images
                


def run_captioning(images, concept_sentence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions =[]
    for i, image_path in enumerate(images):
        #print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        #captions[i] = caption_text
        txtfilename = image_path.split(".")[0]
        txtfile_path =  txtfilename + ".txt"
        with open(txtfile_path, "w") as f:
            f.write(caption_text)

        
    model.to("cpu")
    del model
    del processor



# Fonction principale pour lire, modifier, et écrire le fichier YAML
def modify_yaml(input_file, output_file, new_training_folder, new_folder_path):
    # Lire le fichier YAML
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

        # Modifier la valeur de 'training_folder' sous 'config' -> 'process'
    data['config']['process'][0]['training_folder'] = new_training_folder

    # Modifier la valeur de 'folder_path' sous 'config' -> 'process' -> 'datasets'
    #if 'config' in data and 'process' in data['config'] and 'datasets' in data['config']['process']:
    data['config']['process'][0]['datasets'][0]['folder_path'] = new_folder_path


    # Écrire les modifications dans un nouveau fichier YAML
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
    
    print(f"Fichier modifié et sauvegardé sous : {output_file}")



def main():
    # Chemin du dossier contenant les images
    source_folder = "/datasets/rembrands"

    # Créer une liste des chemins des images
    images = create(source_folder)

    # Créer une phrase conceptuelle
    concept_sentence = "rembr0ndts"

    # Générer des légendes pour les images
    run_captioning(images, concept_sentence)

    # Modifier le fichier YAML
    input_file = "/app/ai-toolkit/config/examples/train_lora_flux_24gb.yaml"
    output_file = "/outputs/scripts/rembrandts.yaml"
    new_training_folder = "/outputs"
    new_folder_path = "/datasets/rembrands"
    modify_yaml(input_file, output_file, new_training_folder, new_folder_path)

if __name__ == "__main__":
    main()