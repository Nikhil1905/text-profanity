import re
import nltk
import pandas as pd
import emoji
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from azure.storage.blob import BlobServiceClient
import torch
import io
import nltk
from nltk.tokenize import sent_tokenize
import logging
import traceback

# Download stopwords list if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords set
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

# Function to remove standalone special characters
def remove_special_characters(text):
    pattern = r'(?<!\w)[^\w\s](?!\w)'
    return re.sub(pattern, '', text)

# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Function to convert emojis to textual representation
def convert_emojis(text):
    return emoji.demojize(text)

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function to preprocess text
def preprocess_text(text):
    final=[]
    sentence = sent_tokenize(text)
    for i in sentence:
        text = remove_special_characters(i)
        text = convert_to_lowercase(text)
        text = convert_emojis(text)
        text = remove_stopwords(text)
        final.append(text)
    return final

# Sample list of profane words
PROFANE_WORDS = [
    '2 girls 1 cup', 'anal', 'anus', 'areole', 'arian', 'arrse', 'arse', 'arsehole', 
    'aryan', 'ass', 'ass-fucker', 'assbang', 'assbanged', 'asses', 'assfuck', 'assfucker', 
    'assfukka', 'asshole', 'assmunch', 'asswhole', 'auto erotic', 'autoerotic', 'ballsack', 
    'bastard', 'bdsm', 'beastial', 'beastiality', 'bellend', 'bestial', 'bestiality', 'bimbo', 
    'bimbos', 'bitch', 'bitches', 'bitchin', 'bitching', 'blow job', 'blowjob', 'blowjobs', 
    'blue waffle', 'bondage', 'boner', 'boob', 'boobs', 'booobs', 'boooobs', 'booooobs', 
    'booooooobs', 'booty call','assshole', 'breasts', 'brown shower', 'brown showers', 'buceta', 
    'bukake', 'bukkake', 'bull shit', 'bullshit', 'busty', 'butthole', 'carpet muncher', 
    'cawk', 'chink', 'cipa', 'clit', 'clitoris', 'clits', 'cnut', 'cock', 'cockface', 'fucker',
    'cockhead', 'cockmunch', 'cockmuncher', 'cocks', 'cocksuck', 'cocksucked', 'cocksucker', 
    'cocksucking', 'cocksucks', 'cokmuncher', 'coon', 'cow-girl', 'cow-girls', 'cowgirl', 
    'cowgirls', 'crap', 'crotch', 'cum', 'cuming', 'cummer', 'cumming', 'cums', 'cumshot', 
    'cunilingus', 'cunillingus', 'cunnilingus', 'cunt', 'cuntlicker', 'cuntlicking', 'cunts', 
    'damn', 'deep throat', 'deepthroat', 'dick', 'dickhead', 'dildo', 'dildos', 'dink', 
    'dinks', 'dlck', 'dog style', 'dog-fucker', 'doggie style', 'doggie-style', 'doggiestyle', 
    'doggin', 'dogging', 'doggy style', 'doggy-style', 'doggystyle', 'dong', 'donkeyribber', 
    'doofus', 'doosh', 'dopey', 'douch3', 'douche', 'douchebag', 'douchebags', 'douchey', 
    'drunk', 'duche', 'dumass', 'dumbass', 'dumbasses', 'dyke', 'dykes', 'eatadick', 
    'eathairpie', 'ejaculate', 'ejaculated', 'ejaculates', 'ejaculating', 'ejaculatings', 
    'ejaculation', 'ejakulate', 'enlargement', 'erect', 'erection', 'erotic', 'erotism', 
    'essohbee', 'extacy', 'extasy', 'f_u_c_k', 'f-u-c-k', 'f.u.c.k', 'f4nny', 'facial', 
    'fack', 'fag', 'fagg', 'fagged', 'fagging', 'faggit', 'faggitt', 'faggot', 'faggs', 
    'fagot', 'fagots', 'fags', 'faig', 'faigt', 'fanny', 'fannybandit', 'fannyflaps', 
    'fannyfucker', 'fanyy', 'fart', 'fartknocker', 'fat', 'fatass', 'fcuk', 'fcuker', 'motherfucker',
    'fcuking', 'feck', 'fecker', 'felch', 'felcher', 'felching', 'fellate', 'fellatio', 
    'feltch', 'feltcher', 'femdom', 'fingerfuck', 'fingerfucked', 'fingerfucker', 
    'fingerfuckers', 'fingerfucking', 'fingerfucks', 'fingering', 'fisted', 'fistfuck', 
    'fistfucked', 'fistfucker', 'fistfuckers', 'fistfucking', 'fistfuckings', 'fistfucks', 
    'fisting', 'fisty', 'flange', 'flogthelog', 'floozy', 'foad', 'fondle', 'foobar', 
    'fook', 'fooker', 'foot job', 'footjob', 'foreskin', 'freex', 'frigg', 'frigga', 
    'fubar', 'fuck', 'fuck-ass', 'fuck-bitch', 'fuck-tard', 'fucka', 'fuckass', 'fucked', 
    'fucker', 'fuckers', 'fuckface', 'fuckhead', 'fuckheads', 'fuckhole', 'fuckin', 
    'fucking', 'fuckings', 'fuckingshitmotherfucker', 'fuckme', 'fuckmeat', 'fucknugget', 
    'fucknut', 'fuckoff', 'fuckpuppet', 'fucks', 'fucktard', 'fucktoy', 'fucktrophy', 
    'fuckup', 'fuckwad', 'fuckwhit', 'fuckwit', 'fuckyomama', 'fudgepacker', 'fuk', 
    'fuker', 'fukker', 'fukkin', 'fukking', 'fuks', 'fukwhit', 'fukwit', 'futanari', 
    'futanary', 'fux', 'fux0r', 'fvck', 'fxck', 'g-spot', 'gae', 'gai', 'gang bang', 
    'gang-bang', 'gangbang', 'gangbanged', 'gangbangs', 'ganja', 'gassyass', 'gay', 
    'gaylord', 'gays', 'gaysex', 'gey', 'gfy', 'ghay', 'ghey', 'gigolo', 'glans', 
    'goatse', 'god', 'god-dam', 'god-damned', 'godamn', 'godamnit', 'goddam', 'goddammit', 
    'goddamn', 'goddamned', 'gokkun', 'golden shower', 'goldenshower', 'gonad', 'gonads', 
    'gook', 'gooks', 'gringo', 'gspot', 'gtfo', 'guido', 'h0m0', 'h0mo', 'hamflap', 
    'hand job', 'handjob', 'hardcoresex', 'hardon', 'he11', 'hebe', 'heeb', 'hell', 
    'hemp', 'hentai', 'heroin', 'herp', 'herpes', 'herpy', 'heshe', 'hitler', 'hiv', 
    'hoar', 'hoare', 'hobag', 'hoer', 'hom0', 'homey', 'homo', 'homoerotic', 'homoey', 
    'honky', 'hooch', 'hookah', 'hooker', 'hoor', 'hootch', 'hooter', 'hooters', 
    'hore', 'horniest', 'horny', 'hotsex', 'howtokill', 'howtomurdep', 'hump', 'humped', 
    'humping', 'hussy', 'hymen', 'inbred', 'incest', 'injun', 'j3rk0ff', 'jack off', 
    'jack-off', 'jackass', 'jackhole', 'jackoff', 'jap', 'japs', 'jerk', 'jerk off', 
    'jerk-off', 'jerk0ff', 'jerked', 'jerkoff', 'jism', 'jiz', 'jizm', 'jizz', 'jizzed', 
    'junkie', 'junky', 'kawk', 'kike', 'kikes', 'kill', 'kinbaku', 'kinky', 'kinkyJesus', 
      'knob', 'kock', 'kondom', 'konk', 'kunt', 'kuntlicker', 'kuntlicking', 'kuntz', 
    'kyke', 'l3tters', 'l33t', 'l3tters', 'leather', 'lesbian', 'lezzie', 'lube', 
    'masturbate', 'masturbation', 'mofo', 'mofos', 'mutha', 'muthafucka', 'muthafuckas', 
    'muthafuckin', 'muthafucking', 'n1gga', 'n1gger', 'nazi', 'nigga', 'nigger', 
    'niggers', 'nutsack', 'orally', 'p0rn', 'p0rn0', 'p3nis', 'p4k', 'paki', 'pano', 
    'panties', 'pecker', 'peeing', 'piss', 'pissed', 'pissin', 'pissin', 'pissing', 
    'playboy', 'poof', 'poon', 'poop', 'porn', 'porn0', 'pornography', 'pr0n', 
    'puss', 'pussy', 'qu33r', 'queer', 'rape', 'rectum', 'retard', 'rimjob', 'roastbeef', 
    'rubandtug', 's3x', 's3x0', 's3xual', 's3xuality', 's3xy', 'sadist', 'shemale', 
    'sh1t', 'sh1tcock', 'sh1tdick', 'sh1tface', 'sh1tfaced', 'sh1thead', 'sh1thole', 
    'sh1ts', 'sh1tster', 'sh1tty', 'sh1z', 'shlong', 'shyte', 'sickfuck', 'slut', 
    'sluts', 'smut', 'snatch', 'sodomy', 'spacelube', 'spunk', 'stfu', 'strapon', 
    'suck', 'sucker', 'sucking', 't1tt1e5', 't1tties', 'teabagger', 'testicle', 
    'threesome', 'titt', 'tits', 'titwank', 'tosser', 'tranny', 'trannies', 'turd', 
    'twat', 'vag', 'vagina', 'wank', 'wanker', 'wetback', 'wh0r3', 'whore', 'w0r3', 
    'wtf', 'yank', 'yiffy', 'z0mg', 'zoophile', 'zoophilia' ,"fuck", "fuck shit", "cock damn", 
    "bitch piss", "crap bitch", "piss crap",
    "dick cock", "asshole bastard", "cunt", "asshole", "slut pussy", "bollocks asshole",
    "dick", "shit", "bitch", "asshole", "pussy", "son of bitch", "Masturbating",
    "Cock sucker", "Fucking dick", "Screw", "Dick", "damn bloody", "bastard",
    "fag bugger", "bastard fag", "fag", "douche bollocks", "bloody slut", "arsehole",
    "bugger darn", "arsehole douche", "damn", "bloody", "Lund", "boor", "gaar",
    "chut", "choot", "crap", "bastard", "darn", "piss", "cock", "bugger", "douche",
    "bollocks", "arsehole", "madarchod", "bhen chod", "bhen k lode", "Mother fucker",
    "Bhosari ke", "chutiya", "Lund", "Haramzada", "Harami", "Haram ka pilla", 
    "Haram ka jana", "Bhadwa", "Madarchod", "Chutia", "Kutiya", "Gandu", "Randi", 
    "Rakhail", "Saali", "Bhosari ke", "Dalla", "Mahesh dalle"
]

# Initialize Blob Service Client
connection_string = 
container_name = ''
blob_name = 'user_inputs.csv'

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def append_to_csv(text, label):
    # Create a DataFrame with the new data
    df = pd.DataFrame({'text': [text], 'label': [label]})

    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)

    # Upload the CSV to Azure Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    
    try:
        # Download the existing CSV
        existing_blob = blob_client.download_blob()
        existing_data = existing_blob.readall().decode('utf-8')
    except:
        # If blob doesn't exist, create it with new data
        existing_data = ''
    
    # Append new data
    updated_csv = existing_data + csv_buffer.getvalue()

    # Upload the updated CSV
    blob_client.upload_blob(updated_csv, overwrite=True)

# Function to detect profanity using regex and return the profane words found
def detect_profanity_with_regex(text, profane_words=PROFANE_WORDS):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, profane_words)) + r')\b', re.IGNORECASE)
    profane_words_found = pattern.findall(text)
    
    if profane_words_found:
        return True, profane_words_found  # Return True and the list of profane words found
    return False, []

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face model and tokenizer
model_name = "parsawar/profanity_model_3.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to classify text
def classify_text(text):
    # Tokenize the preprocessed text
    inputs = tokenizer(text, add_special_tokens=True, truncation=True, max_length=512, return_tensors="pt")

    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted label and confidence score
    predictions = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(predictions, dim=1).item()
    score = predictions[0][label].item()
    
    return label, score

@app.route('/v1/text-profanity/detect', methods=['POST'])
def score():
    # Retrieve text from the request
    text = request.form.get('text')
    
    # Check if text is provided
    if text is None:
        return jsonify({'error': "Please input text."}), 400

    try:
        # Clean and preprocess the input text
        cleaned_text = preprocess_text(text.strip())
    
        # Detect profanity using regex
        profane_detected, profane_words_found = detect_profanity_with_regex(text)  
        if profane_detected:
            profane_words_string = ', '.join(profane_words_found)
            response = {
                'label': "1",  # Profanity detected
                'score': "0.99",
                'profane_words': profane_words_string
            }
        else:
            # If no profane words are found, classify the text using the Hugging Face model
            for i in cleaned_text:
                classified_label, hf_label_score = classify_text(i)
                response = {
                    'label': str(classified_label),
                    'score': str(hf_label_score),
                    'profane_words': None
                }
        
        # Append user input and label to CSV
        append_to_csv(text, response['label'])

        # Return the response as JSON
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}. Line: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
