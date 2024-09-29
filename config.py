from email.mime import image
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os
import yaml
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict
import time
from azure.storage.queue import QueueServiceClient,QueueMessage,QueueClient
from threading import Thread
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job




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
        print(caption_text)
        txtfilename = image_path.split(".")[0]
        txtfile_path =  txtfilename + ".txt"
        with open(txtfile_path, "w") as f:
            f.write(caption_text)

        
    model.to("cpu")
    del model
    del processor



# Fonction principale pour lire, modifier, et écrire le fichier YAML
def modify_yaml(input_file, output_file, new_training_folder, new_folder_path,concept_sentence,twolayers=False):
    # Lire le fichier YAML
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

        # Modifier la valeur de 'training_folder' sous 'config' -> 'process'
    data['config']['process'][0]['training_folder'] = new_training_folder
    data['config']['process'][0]['concept_sentence'] = concept_sentence
    data['config']['process'][0]['train']['lr'] = 4e-4
    data['config']['process'][0]['train']['disable_sampling']=True
    data['config']['process'][0]['trigger_word'] =concept_sentence
    # Modifier la valeur de 'folder_path' sous 'config' -> 'process' -> 'datasets'
    #if 'config' in data and 'process' in data['config'] and 'datasets' in data['config']['process']:
    data['config']['process'][0]['datasets'][0]['folder_path'] = new_folder_path

    if(twolayers):
        data['config']['process'][0]['network']['linear'] = 128
        data['config']['process'][0]['network']['linear_alpha'] = 128
        data['config']['process'][0]['network']['network_kwargs'] = {'only_if_contains': ["transformer.single_transformer_blocks.7.proj_out","transformer.single_transformer_blocks.20.proj_out"]}

    # linear: 128
    #     linear_alpha: 128
    #     network_kwargs:
    #       only_if_contains:
    #         - "transformer.single_transformer_blocks.7.proj_out"
    #         - "transformer.single_transformer_blocks.20.proj_out"

    # Écrire les modifications dans un nouveau fichier YAML
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
    
    print(f"Fichier modifié et sauvegardé sous : {output_file}")

def update_visibility_periodically(msg:QueueMessage, queue_client:QueueClient, visibility_timeout):
    """
    Fonction pour mettre à jour la visibilité du message périodiquement.
    """
    while True:
        # Attendre la moitié du temps de visibilité avant de renouveler
        time.sleep(visibility_timeout / 2)
        try:
            # Mettre à jour la visibilité pour un autre cycle
            updated_message = queue_client.update_message(
                msg.id, msg.pop_receipt, visibility_timeout=visibility_timeout
            )
            # Mettre à jour le pop_receipt après chaque mise à jour
            msg.pop_receipt = updated_message.pop_receipt
            #print("Visibilité du message mise à jour.")
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la visibilité : {e}")
            break


def main():
    
    # Connexion au compte de stockage Azure
    connect_str =os.environ.get("aiavatarconnstring")
    queue_name = "fluxtrainjob"

    queue_service_client = QueueServiceClient.from_connection_string(connect_str)
    queue_client = queue_service_client.get_queue_client(queue_name)

    # Durée de la visibilité initiale (en secondes)
    visibility_timeout = 120

    # Lire un message dans la file d'attente
    messages = queue_client.receive_messages(visibility_timeout=visibility_timeout, max_messages=1)
    for msg in messages:
        print(msg.content)
    #msg:QueueMessage=messages[0]
    # Démarrer un thread pour mettre à jour la visibilité périodiquement
        visibility_thread = Thread(target=update_visibility_periodically, args=(msg, queue_client, visibility_timeout))
        visibility_thread.start()
        
        try:
            jsmsg=json.loads(msg.content)
            print(jsmsg)
            name=jsmsg["name"]
            # Chemin du dossier contenant les images
            source_folder = "/datasets/"+name#rembrands"

            # Créer une liste des chemins des images
            images = create(source_folder)

            # Créer une phrase conceptuelle
            concept_sentence =jsmsg["concept_sentence"] #"rembr0ndts"
            twolayers=jsmsg.get("twolayers",False)
            # Générer des légendes pour les images
            run_captioning(images, concept_sentence)
            
            # Modifier le fichier YAML
            input_file = "/app/ai-toolkit/config/examples/train_lora_flux_24gb.yaml"
            output_file = "/outputs/scripts/"+name+".yaml"
            new_training_folder = "/outputs/loras"
            new_folder_path = "/datasets/"+name
            modify_yaml(input_file, output_file, new_training_folder, new_folder_path,concept_sentence,twolayers)


            try:
                    job = get_job(output_file, name)
                    job.run()
                    job.cleanup()
            except Exception as e:
                    print(f"Error running job: {e}")

            queue_client.delete_message(msg.id, msg.pop_receipt)
            print("Message supprimé avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'exécution du script : {e}")
    
    # Stopper la mise à jour de la visibilité en terminant le thread
        visibility_thread.join()

if __name__ == "__main__":
    main()