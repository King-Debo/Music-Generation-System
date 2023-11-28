# Import the required libraries
import os
import requests
import zipfile
import numpy as np
import librosa
import mido
import torch
from torch.utils.data import Dataset, DataLoader

# Define some constants
DATA_DIR = "data" # The directory where the data files will be stored
MIDI_URL = "https://www.kaggle.com/saikayala/maestro-v3-midi/download" # The URL of the MIDI dataset
AUDIO_URL = "https://www.kaggle.com/saikayala/maestro-v3-wav/download" # The URL of the audio dataset
SCORE_URL = "https://www.kaggle.com/saikayala/maestro-v3-score/download" # The URL of the score dataset
LYRICS_URL = "https://www.kaggle.com/saikayala/maestro-v3-lyrics/download" # The URL of the lyrics dataset
MIDI_EXT = ".midi" # The extension of the MIDI files
AUDIO_EXT = ".wav" # The extension of the audio files
SCORE_EXT = ".xml" # The extension of the score files
LYRICS_EXT = ".txt" # The extension of the lyrics files
SAMPLE_RATE = 22050 # The sample rate of the audio files
N_MELS = 128 # The number of mel-frequency bands for the spectrograms
HOP_LENGTH = 512 # The hop length for the spectrograms
WINDOW_SIZE = 2048 # The window size for the spectrograms
MAX_LENGTH = 120 # The maximum length of the songs in seconds
N_PITCHES = 128 # The number of possible pitches for the MIDI files
N_VELOCITIES = 128 # The number of possible velocities for the MIDI files
N_CHORDS = 24 # The number of possible chords for the MIDI files
N_BEATS = 16 # The number of possible beats for the MIDI files
N_GENRES = 10 # The number of possible genres for the songs
N_TEMPOS = 10 # The number of possible tempos for the songs
N_KEYS = 12 # The number of possible keys for the songs
N_MODES = 2 # The number of possible modes for the songs
GENRE_DICT = {"Classical": 0, "Jazz": 1, "Rock": 2, "Pop": 3, "Blues": 4, "Country": 5, "Reggae": 6, "Hip-hop": 7, "Metal": 8, "Electronic": 9} # The dictionary that maps the genres to indices
TEMPO_DICT = {"Largo": 0, "Adagio": 1, "Andante": 2, "Moderato": 3, "Allegro": 4, "Vivace": 5, "Presto": 6, "Prestissimo": 7, "Lento": 8, "Grave": 9} # The dictionary that maps the tempos to indices
KEY_DICT = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11} # The dictionary that maps the keys to indices
MODE_DICT = {"Major": 0, "Minor": 1} # The dictionary that maps the modes to indices
BATCH_SIZE = 32 # The batch size for the data loader

# Define a function to download a dataset from a URL and unzip it
def download_dataset(url, ext):
  # Create the data directory if it does not exist
  if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
  # Get the file name from the URL
  file_name = url.split("/")[-1]
  # Get the file path by joining the data directory and the file name
  file_path = os.path.join(DATA_DIR, file_name)
  # Check if the file already exists
  if not os.path.exists(file_path):
    # Download the file from the URL
    print(f"Downloading {file_name}...")
    response = requests.get(url, stream=True)
    # Save the file to the file path
    with open(file_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=1024):
        if chunk:
          f.write(chunk)
    print(f"Downloaded {file_name}.")
  else:
    print(f"{file_name} already exists.")
  # Check if the file is a zip file
  if file_name.endswith(".zip"):
    # Get the folder name by removing the zip extension
    folder_name = file_name.replace(".zip", "")
    # Get the folder path by joining the data directory and the folder name
    folder_path = os.path.join(DATA_DIR, folder_name)
    # Check if the folder already exists
    if not os.path.exists(folder_path):
      # Unzip the file to the folder path
      print(f"Unzipping {file_name}...")
      with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(folder_path)
      print(f"Unzipped {file_name}.")
    else:
      print(f"{folder_name} already exists.")
    # Return the folder path
    return folder_path
  else:
    # Return the file path
    return file_path

# Define a function to load the MIDI files from a folder and return a list of MIDI objects
def load_midi_files(folder):
  # Initialize an empty list to store the MIDI objects
  midi_files = []
  # Loop through the files in the folder
  for file in os.listdir(folder):
    # Check if the file has the MIDI extension
    if file.endswith(MIDI_EXT):
      # Get the file path by joining the folder and the file name
      file_path = os.path.join(folder, file)
      # Load the MIDI file as a MIDI object
      midi_file = mido.MidiFile(file_path)
      # Append the MIDI object to the list
      midi_files.append(midi_file)
  # Return the list of MIDI objects
  return midi_files

# Define a function to load the audio files from a folder and return a list of audio arrays
def load_audio_files(folder):
  # Initialize an empty list to store the audio arrays
  audio_files = []
  # Loop through the files in the folder
  for file in os.listdir(folder):
    # Check if the file has the audio extension
    if file.endswith(AUDIO_EXT):
      # Get the file path by joining the folder and the file name
      file_path = os.path.join(folder, file)
      # Load the audio file as an audio array
      audio_file, _ = librosa.load(file_path, sr=SAMPLE_RATE)
      # Append the audio array to the list
      audio_files.append(audio_file)
  # Return the list of audio arrays
  return audio_files

# Define a function to load the score files from a folder and return a list of score strings
def load_score_files(folder):
  # Initialize an empty list to store the score strings
  score_files = []
  # Loop through the files in the folder
  for file in os.listdir(folder):
    # Check if the file has the score extension
    if file.endswith(SCORE_EXT):
      # Get the file path by joining the folder and the file name
      file_path = os.path.join(folder, file)
      # Load the score file as a score string
      with open(file_path, "r") as f:
        score_file = f.read()
      # Append the score string to the list
      score_files.append(score_file)
  # Return the list of score strings
  return score_files

# Define a function to load the lyrics files from a folder and return a list of lyrics strings
def load_lyrics_files(folder):
  # Initialize an empty list to store the lyrics strings
  lyrics_files = []
  # Loop through the files in the folder
  for file in os.listdir(folder):
    # Check if the file has the lyrics extension
    if file.endswith(LYRICS_EXT):
      # Get the file path by joining the folder and the file name
      file_path = os.path.join(folder, file)
      # Load the lyrics file as a lyrics string
      with open(file_path, "r") as f:
        lyrics_file = f.read()
      # Append the lyrics string to the list
      lyrics_files.append(lyrics_file)
  # Return the list of lyrics strings
  return lyrics_files

# Define a function to convert a MIDI object to a tensor of shape (MAX_LENGTH * N_PITCHES, N_VELOCITIES + N_CHORDS + N_BEATS)
def midi_to_tensor(midi_file):
  # Initialize an empty list to store the MIDI messages
  midi_messages = []
  # Loop through the MIDI tracks
  for track in midi_file.tracks:
    # Loop through the MIDI messages
    for msg in track:
      # Check if the message is a note on or a note off message
      if msg.type == "note_on" or msg.type == "note_off":
        # Append the message to the list
        midi_messages.append(msg)
  # Initialize a variable to store the current time in ticks
  current_time = 0
  # Initialize a variable to store the current chord in index
  current_chord = 0
  # Initialize a variable to store the current beat in index
  current_beat = 0
  # Initialize an empty dictionary to store the active notes and their velocities
  active_notes = {}
  # Initialize an empty list to store the tensor rows
  tensor_rows = []
  # Loop through the MIDI messages
  for msg in midi_messages:
    # Update the current time by adding the message time
    current_time += msg.time
    # Check if the message is a note on message
    if msg.type == "note_on":
      # Check if the message velocity is positive
      if msg.velocity > 0:
        # Add the note and the velocity to the active notes dictionary
        active_notes[msg.note] = msg.velocity
      else:
        # Remove the note from the active notes dictionary
        active_notes.pop(msg.note, None)
    # Check if the message is a note off message
    elif msg.type == "note_off":
      # Remove the note from the active notes dictionary
      active_notes.pop(msg.note, None)
    # Check if the message is a meta message
    elif msg.is_meta:
      # Check if the message type is key signature
      if msg.type == "key_signature":
        # Get the key and the mode from the message key
        key, mode = msg.key.split(" ")
        # Convert the key and the mode to indices
        key_index = KEY_DICT[key]
        mode_index = MODE_DICT[mode]
        # Compute the chord index by adding the key index and the mode index times 12
        chord_index = key_index + mode_index * 12
        # Update the current chord
        current_chord = chord_index
      # Check if the message type is time signature
      elif msg.type == "time_signature":
        # Get the numerator and the denominator from the message
        numerator = msg.numerator
        denominator = msg.denominator
        # Compute the beat index by dividing the numerator by the denominator and multiplying by 4
        beat_index = int(numerator / denominator * 4)
        # Update the current beat
        current_beat = beat_index
    # Convert the current time from ticks to seconds
    current_time_seconds = mido.tick2second(current_time, midi_file.ticks_per_beat, midi_file.tempo)
    # Check if the current time seconds is greater than the maximum length
    if current_time_seconds > MAX_LENGTH:
      # Break the loop
      break
    # Initialize an empty list to store the tensor row
    tensor_row = []
    # Loop through the possible pitches
    for pitch in range(N_PITCHES):
      # Check if the pitch is in the active notes dictionary
      if pitch in active_notes:
        # Get the velocity from the active notes dictionary
        velocity = active_notes[pitch]
        # Convert the velocity to a one-hot vector of length N_VELOCITIES
        velocity_vector = [0] * N_VELOCITIES
        velocity_vector[velocity] = 1
        # Append the velocity vector to the tensor row
        tensor_row.extend(velocity_vector)
      else:
        # Append a zero vector of length N_VELOCITIES to the tensor row
        tensor_row.extend([0] * N_VELOCITIES)
    # Convert the current chord to a one-hot vector of length N_CHORDS
    chord_vector = [0] * N_CHORDS
    chord_vector[current_chord] = 1
    # Append the chord vector to the tensor row
    tensor_row.extend(chord_vector)
    # Convert the current beat to a one-hot vector of length N_BEATS
    beat_vector = [0] * N_BEATS
    beat_vector[current_beat] = 1
    # Append the beat vector to the tensor row
    tensor_row.extend(beat_vector)
    # Append the tensor row to the tensor rows list
    tensor_rows.append(tensor_row)
  # Convert the tensor rows list to a numpy array
  tensor_array = np.array(tensor_rows)
  # Pad the tensor array with zeros to the shape (MAX_LENGTH * N_PITCHES, N_VELOCITIES + N_CHORDS + N_BEATS)
  tensor_array = np.pad(tensor_array, ((0, MAX_LENGTH * N_PITCHES - tensor_array.shape[0]), (0, 0)), mode="constant")
  # Convert the tensor array to a torch tensor
  tensor = torch.from_numpy(tensor_array)
  # Return the tensor
  return tensor

